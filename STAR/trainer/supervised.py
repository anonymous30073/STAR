import json
import operator
import os
import random
import time
from os.path import exists, join

import torch
from datasets import Dataset
from ray import tune
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from STAR.utils.metrics import calculate_metrics, log_results
from collections import defaultdict

INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"


class RegWeightScheduler:
    # https://github.com/naver/splade/blob/main/splade/losses/regularization.py
    def __init__(self, lambda_=5e-3, T=50000):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


class SupervisedTrainer:
    def __init__(self, config, dataset_config, model, device, logger, exp_dir, test_only=False, tuning=False,
                 save_checkpoint=True,
                 relevance_level=1, users=None, items=None, dataset_eval_neg_sampling=None, to_load_model_name=None,
                 padding_token=None, item_mapping=None, user_mapping=None):
        self.model = model
        self.device = device
        self.logger = logger
        self.padding_token = padding_token
        self.test_only = test_only  # todo used?
        self.tuning = tuning
        self.save_checkpoint = save_checkpoint
        self.relevance_level = relevance_level
        self.valid_metric = config['valid_metric']
        self.patience = config['early_stopping_patience'] if (
                    'early_stopping_patience' in config and config['early_stopping_patience'] != '') else None
        self.do_validation = config["do_validation"]
        self.best_model_train_path = None
        self.last_model_path = None
        self.best_model_path = join(exp_dir, 'best_model.pth')
        self.dataset_config = dataset_config
        self.config = config
        self.exp_dir = exp_dir

        if "save_best_train" in config and config["save_best_train"] is True:
            self.best_model_train_path = join(exp_dir, 'best_model_tr_loss.pth')
        if "save_every_epoch" in config and config["save_every_epoch"] is True:
            self.last_model_path = join(exp_dir, 'last_model.pth')

        if to_load_model_name is not None:
            self.to_load_model = join(exp_dir, f"{to_load_model_name}.pth")
        else:
            self.to_load_model = self.best_model_path
            to_load_model_name = "best_model"

        neg_name = dataset_eval_neg_sampling['test']
        if neg_name.startswith("f:"):
            neg_name = neg_name[len("f:"):]
        self.test_output_path = {"ground_truth": join(exp_dir, f'test_ground_truth_{neg_name}.json'),
                                 "predicted": join(exp_dir, f'test_predicted_{neg_name}_{to_load_model_name}')}

        self.users = users
        self.items = items
        self.sig_output = config["sigmoid_output"]
        self.enable_autocast = False
        self.validation_user_sample_num = None
        if "enable_autocast" in config:
            self.enable_autocast = config["enable_autocast"]
        if "validation_user_sample_num" in config and config["validation_user_sample_num"] != "":
            self.validation_user_sample_num = config["validation_user_sample_num"]

        if config['loss_fn'] == "BCE":
            if self.sig_output is False:
                raise ValueError("cannot have BCE with no sigmoid")
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif config['loss_fn'] == "MRL":
            self.loss_fn = torch.nn.MarginRankingLoss(margin=config["margin"])
        elif config['loss_fn'] == "DISTIL":
            self.subloss_fn_name = config["subloss_fn"]
            tmp = json.load(open(dataset_config["distil_path"]))
            self.distil_scores = defaultdict(dict)
            for kk, vv in tmp.items():
                for k, v in vv.items():
                    self.distil_scores[user_mapping[kk]][item_mapping[k]] = v
            del tmp
            if config["subloss_fn"] == "pointwiseMSE" or config["subloss_fn"] == "pairwiseMSE":
                self.loss_fn = torch.nn.MSELoss()
            elif config["subloss_fn"] == "BCE":
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"loss_fn {config['loss_fn']} is not implemented!")

        self.loss_fn_name = config["loss_fn"]

        self.epochs = config['epochs']
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_saved_valid_metric = np.inf if self.valid_metric == "valid_loss" else -np.inf
        if exists(self.to_load_model):
            checkpoint = torch.load(self.to_load_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['epoch']
            if "best_valid_metric" in checkpoint:
                self.best_saved_valid_metric = checkpoint['best_valid_metric']
            print("last checkpoint restored")
        self.model.to(device)

        self.u_reg_scheduler = RegWeightScheduler(config["u_lambda_flops"] if "u_lambda_flops" in config else 5e-3,
                                                  config["T"] if "T" in config else 50000)
        self.i_reg_scheduler = RegWeightScheduler(config["u_lambda_flops"] if "u_lambda_flops" in config else 5e-3,
                                                  config["T"] if "T" in config else 50000)

        if not test_only:
            opt_params = self.model.parameters()
            if config['optimizer'] == "Adam":
                self.optimizer = Adam(opt_params, lr=config['lr'], weight_decay=config['wd'])
            elif config['optimizer'] == "AdamW":
                self.optimizer = AdamW(opt_params, lr=config['lr'], weight_decay=config['wd'])

            if exists(self.to_load_model):
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    print("optimizer_state_dict was not saved in the checkpoint")

        self.train_doc_num = dataset_config["training_neg_samples"] + 1

    def fit(self, train_dataloader, valid_dataloader):
        early_stopping_cnt = 0
        comparison_op = operator.lt if self.valid_metric == "valid_loss" else operator.gt

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        scaler = GradScaler()

        best_train_loss = np.inf
        for epoch in range(self.start_epoch, self.epochs):
            if self.patience is not None and early_stopping_cnt == self.patience:
                print(f"Early stopping after {self.patience} epochs not improving!")
                break

            self.model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                        disable=True if self.tuning else False)

            train_loss, total_count = 0, 0
            inter_loss, user_flops_loss, item_flops_loss = 0, 0, 0

            for batch_idx, batch in pbar:
                if "user_input_ids" in batch:
                    batch_size = int(len(batch['user_input_ids']) / self.train_doc_num)
                    batch["user_input_ids"] = batch['user_input_ids'][:batch_size]
                    batch['user_attention_mask'] = batch['user_attention_mask'][:batch_size]

                    batch = {k: v.to(self.device) for k, v in batch.items() if
                             k in ["user_input_ids", "user_attention_mask", "item_input_ids", "item_attention_mask",
                                   "label", INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD]}

                elif "user_tokenized_text" in batch:
                    batch = {k: v.to(self.device) for k, v in batch.items() if
                             k in ['user_tokenized_text', 'item_tokenized_text', 'internal_user_id', 'internal_item_id',
                                   'label']}
                else:
                    batch = {k: v.to(self.device) for k, v in batch.items() if
                             k in ["input_ids", "attention_mask", "label", INTERNAL_USER_ID_FIELD]}

                label = batch.pop("label").float()

                self.optimizer.zero_grad()

                with autocast(enabled=self.enable_autocast, device_type='cuda', dtype=torch.float16):
                    output, flops_user, flops_item, pos_scores, neg_scores = self.model(batch)

                    if self.loss_fn_name == "BCE":
                        loss = self.loss_fn(output, label)
                    elif self.loss_fn_name == "DISTIL":
                        target = torch.tensor([self.distil_scores[uu.item()][ii.item()] for uu, ii in
                                               zip(batch[INTERNAL_USER_ID_FIELD], batch[INTERNAL_ITEM_ID_FIELD])]).to(
                            "cuda")
                        if self.subloss_fn_name == "pairwiseMSE":
                            pos_target = target[:batch_size].repeat(self.train_doc_num - 1)
                            neg_target = target[batch_size:]
                            if self.sig_output:
                                pos_scores = torch.sigmoid(pos_scores)
                                neg_scores = torch.sigmoid(neg_scores)
                                pos_target = torch.sigmoid(pos_target)
                                neg_target = torch.sigmoid(neg_target)
                            pos_scores = pos_scores.repeat(self.train_doc_num - 1, 1)
                            diff = pos_scores - neg_scores
                            diff_target = pos_target - neg_target
                            loss = self.loss_fn(diff, diff_target.unsqueeze(1))
                        else:
                            if self.subloss_fn_name == "BCE":
                                target = torch.sigmoid(target)
                            if self.sig_output and self.subloss_fn_name == "pointwiseMSE":
                                output = torch.sigmoid(output)
                                target = torch.sigmoid(target)
                            loss = self.loss_fn(output, target.unsqueeze(1))
                    else:
                        if self.sig_output:
                            pos_scores = torch.sigmoid(pos_scores)
                            neg_scores = torch.sigmoid(neg_scores)
                        if self.loss_fn._get_name() == "MarginRankingLoss":
                            pos_scores = pos_scores.repeat(self.train_doc_num - 1, 1)
                            loss = self.loss_fn(pos_scores, neg_scores,
                                                    torch.ones((len(pos_scores), 1), device=self.device))
                        else:
                            loss = self.loss_fn(output, label)
                    inter_loss += loss
                    flops_user *= self.u_reg_scheduler.step()
                    flops_item *= self.i_reg_scheduler.step()
                    user_flops_loss += flops_user
                    item_flops_loss += flops_item
                    loss += flops_user + flops_item

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss
                total_count += label.size(0)

                pbar.set_description(
                    f'loss: {loss.item():.8f},  epoch: {epoch}/{self.epochs}')

            train_loss /= total_count
            inter_loss /= total_count
            user_flops_loss /= total_count
            item_flops_loss /= total_count
            print(f"Train loss epoch {epoch}: {train_loss}")

            if self.best_model_train_path is not None:
                if train_loss < best_train_loss:
                    checkpoint = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint, f"{self.best_model_train_path}_tmp")
                    os.rename(f"{self.best_model_train_path}_tmp", self.best_model_train_path)
                    best_train_loss = train_loss
            if self.last_model_path is not None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, f"{self.last_model_path}_tmp")
                os.rename(f"{self.last_model_path}_tmp", self.last_model_path)

            self.logger.add_scalar('epoch_metrics/train_loss', train_loss, epoch)
            self.logger.add_scalar('epoch_metrics/inter_loss', inter_loss, epoch)
            self.logger.add_scalar('epoch_metrics/user_flops_loss', user_flops_loss, epoch)
            self.logger.add_scalar('epoch_metrics/item_flops_loss', item_flops_loss, epoch)

            if self.do_validation:
                outputs, ground_truth, valid_loss, users, items, val_inter_loss, val_user_flops_loss, val_item_flops_loss = self.predict(
                    valid_dataloader,
                    num_negs=int(self.dataset_config["validation_neg_sampling_strategy"].split('_')[-1]))

                results = calculate_metrics(ground_truth, outputs, users, items, self.relevance_level)
                results["loss"] = valid_loss
                results["val_inter_loss"] = val_inter_loss
                results["val_user_flops_loss"] = val_user_flops_loss
                results["val_item_flops_loss"] = val_item_flops_loss

                results = {f"valid_{k}": v for k, v in results.items()}
                for k, v in results.items():
                    self.logger.add_scalar(f'epoch_metrics/{k}', v, epoch)
                print(
                    f"Valid loss epoch {epoch}: {valid_loss} - {self.valid_metric} = {results[self.valid_metric]:.6f}\n")

                if comparison_op(results[self.valid_metric], self.best_saved_valid_metric):
                    self.best_saved_valid_metric = results[self.valid_metric]
                    self.best_epoch = epoch
                    if self.save_checkpoint:
                        checkpoint = {
                            'epoch': self.best_epoch,
                            'best_valid_metric': self.best_saved_valid_metric,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                        torch.save(checkpoint, f"{self.best_model_path}_tmp")
                        os.rename(f"{self.best_model_path}_tmp", self.best_model_path)
                    early_stopping_cnt = 0
                else:
                    early_stopping_cnt += 1
                self.logger.add_scalar('epoch_metrics/best_epoch', self.best_epoch, epoch)
                self.logger.add_scalar('epoch_metrics/best_valid_metric', self.best_saved_valid_metric, epoch)
            self.logger.flush()

    def evaluate(self, eval_dataloader):
        eval_output_path = self.test_output_path
        checkpoint = torch.load(self.to_load_model, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.best_epoch = checkpoint['epoch']
        print("best model loaded!")
        outfile = f"{eval_output_path['predicted']}_e-{self.best_epoch}.json"

        outputs, ground_truth, valid_loss, internal_user_ids, internal_item_ids, val_inter_loss, val_user_flops_loss, val_item_flops_loss = self.predict(
            eval_dataloader, num_negs=int(self.dataset_config["test_neg_sampling_strategy"].split('_')[-1]))
        log_results(ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items,
                    eval_output_path['ground_truth'],
                    outfile,
                    f"{eval_output_path['log']}_e-{self.best_epoch}.txt" if "log" in eval_output_path else None)

        # only for cross
        if getattr(self.model, "prec_representations_for_distil", None) and not os.path.exists(os.path.join(self.exp_dir, "distil_scores.json")):
            self.model.prec_representations_for_distil(self.users, self.items, padding_token=self.padding_token)


    def predict(self, eval_dataloader, num_negs=100):
        self.model.eval()

        outputs = []
        ground_truth = []
        user_ids = []
        item_ids = []
        eval_loss, total_count = 0, 0
        inter_loss, user_flops_loss, item_flops_loss = 0, 0, 0
        pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=True if self.tuning else False)

        with torch.no_grad():
            for batch_idx, batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()

                with autocast(enabled=self.enable_autocast, device_type='cuda', dtype=torch.float16):
                    if "user_input_ids" in batch:
                        batch_size = int(len(batch['user_input_ids']) / (num_negs + 1))
                        batch["user_input_ids"] = batch['user_input_ids'][:batch_size]
                        batch['user_attention_mask'] = batch['user_attention_mask'][:batch_size]

                    output, flops_user, flops_item, pp, nn = self.model(batch, mode="eval")

                    if self.loss_fn._get_name() == "BCEWithLogitsLoss":
                        loss = self.loss_fn(output, label)
                        output = torch.sigmoid(output)
                    else:
                        if self.sig_output:
                            output = torch.sigmoid(output)
                        if self.loss_fn._get_name() == "MarginRankingLoss":
                            loss = torch.Tensor([-1]).to(self.device)
                        else:
                            loss = self.loss_fn(output, label)

                inter_loss += loss
                user_flops_loss += flops_user
                item_flops_loss += flops_item
                eval_loss += loss.item()
                total_count += label.size(0)

                ground_truth.extend(label.squeeze(1).tolist())
                outputs.extend(output.squeeze(1).tolist())
                user_ids.extend(batch[
                                    INTERNAL_USER_ID_FIELD].squeeze(1).tolist())
                item_ids.extend(batch[INTERNAL_ITEM_ID_FIELD].squeeze(1).tolist())

                pbar.set_description(
                    f'loss: {loss.item():.8f}')

            eval_loss /= total_count
            inter_loss /= total_count
            user_flops_loss /= total_count
            item_flops_loss /= total_count
        ground_truth = torch.tensor(ground_truth)
        outputs = torch.tensor(outputs)
        return outputs, ground_truth, eval_loss, user_ids, item_ids, inter_loss, user_flops_loss, item_flops_loss
