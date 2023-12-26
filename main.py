import argparse
import json
import os
import random
from os.path import join

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from STAR.trainer.supervised import SupervisedTrainer
from STAR.utils.data_loading_lsr import load_data
from STAR.utils.others import get_model

def main(op, config_file=None, result_folder=None, given_train_neg_num = None,
         given_eval_model=None, given_eval_pos_file=None, given_eval_neg_file=None,
         given_lr=None, given_tbs=None, given_user_text_file_name=None, given_item_text_file_name=None,
         given_lambda_flops = None, given_T = None, given_margin = None, given_eval_batch_size = None,
         given_subloss_fn = None, given_sigmoid_output = None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_only = False
    if op in ["train", "trainonly"]:
        config = json.load(open(config_file, 'r'))
        if given_lr is not None:
            config['trainer']['lr'] = given_lr
        if given_tbs is not None:
            config['dataset']['train_batch_size'] = given_tbs
        if given_user_text_file_name is not None:
            config['dataset']['user_text_file_name'] = given_user_text_file_name
        if given_item_text_file_name is not None:
            config['dataset']['item_text_file_name'] = given_item_text_file_name
        if given_train_neg_num:
            config["dataset"]["training_neg_samples"] = int(given_train_neg_num)
        if given_T:
            config["trainer"]["T"] = given_T
        if given_lambda_flops:
            config["trainer"]["u_lambda_flops"] = given_lambda_flops
            config["trainer"]["i_lambda_flops"] = given_lambda_flops
        if given_margin:
            config["trainer"]["margin"] = given_margin
        if given_subloss_fn:
            config["trainer"]["subloss_fn"] = given_subloss_fn
        if given_sigmoid_output:
            config["trainer"]["sigmoid_output"] = False

        exp_dir_params = []
        for param in config['params_in_exp_dir']:
            p1 = param[:param.index(".")]
            p2 = param[param.index(".")+1:]
            if param == "dataset.test_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
                temp = config[p1][p2]
                temp = temp[temp.index("f:test_neg_")+len("f:test_neg_"):]
                exp_dir_params.append(f"f-{temp}")
            elif isinstance(config[p1][p2], list):
                exp_dir_params.append('-'.join([str(v) for v in config[p1][p2]]))
            else:
                exp_dir_params.append(str(config[p1][p2]))
        exp_dir = join(config['experiment_root'], "_".join(exp_dir_params))

        config["experiment_dir"] = exp_dir
        if os.path.exists(join(exp_dir, "config.json")):
            config2 = json.load(open(join(exp_dir, "config.json"), 'r'))
            if config != config2:
                print(f"GivenConfig: {config}")
                raise ValueError(f"{exp_dir} exists with different config != {config_file}")
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(config, open(join(exp_dir, "config.json"), 'w'), indent=4)
    elif op in ["test"]:
        config = json.load(open(join(result_folder, "config.json"), 'r'))
        exp_dir = config["experiment_dir"]
        test_only = True
        if given_eval_pos_file is not None:
            config["dataset"]["alternative_pos_test_file"] = given_eval_pos_file
        if given_eval_neg_file is not None:
            config["dataset"]["test_neg_sampling_strategy"] = given_eval_neg_file
        if given_eval_batch_size:
            config["dataset"]["eval_batch_size"] = int(given_eval_batch_size)
    else:
        raise ValueError("op not defined!")

    logger = SummaryWriter(exp_dir)
    print(exp_dir)
    print(config)

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token, item_mapping, user_mapping, test_label_list = \
        load_data(config['dataset'],
                  pretrained_model=config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None,
                  joint=True if config["model"]["name"] in ["BertCross"] else False)

    model = get_model(config['model'], device, config['dataset'], exp_dir)

    trainer = SupervisedTrainer(config=config['trainer'], dataset_config = config["dataset"], model=model, device=device, logger=logger, exp_dir=exp_dir,
                                    test_only=test_only, relevance_level=relevance_level,
                                    users=users, items=items,
                                    dataset_eval_neg_sampling=
                                    {"validation": config["dataset"]["validation_neg_sampling_strategy"],
                                    "test": config["dataset"]["test_neg_sampling_strategy"]},
                                    to_load_model_name=given_eval_model,
                                    padding_token=padding_token, item_mapping = item_mapping, user_mapping = user_mapping)
    if op == "train":
        trainer.fit(train_dataloader, valid_dataloader)
        trainer.evaluate(test_dataloader)
    elif op == "trainonly":
        trainer.fit(train_dataloader, valid_dataloader)
    elif op == "test":
        trainer.evaluate(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    parser.add_argument('--eval_model_name', default=None, help='test time model to load, only for op == test')
    parser.add_argument('--eval_pos_file', default=None, help='test positive file, only for op == test')
    parser.add_argument('--eval_neg_file', default=None, help='test neg file, only for op == test')
    parser.add_argument('--trainer_lr', default=None, help='trainer learning rate')
    parser.add_argument('--train_batch_size', default=None, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default=None, help='eval_batch_size')
    parser.add_argument('--user_text_file_name', default=None, help='user_text_file_name')
    parser.add_argument('--item_text_file_name', default=None, help='item_text_file_name')
    parser.add_argument('--train_neg_num', type=int, default=None, help='train neg num')
    parser.add_argument('--T', type=int, default=None, help='T')
    parser.add_argument('--lambda_flops', type=float, default=None, help='lambda flops')
    parser.add_argument('--margin', type=float, default=None, help='MRL margin')
    parser.add_argument('--subloss_fn', type=str, default=None, help='subloss fn name')
    parser.add_argument('--no_sigmoid', action="store_true", default=None, help='sigmoid or not')
    parser.add_argument('--op', type=str, help='operation train/test/trainonly')
    args, _ = parser.parse_known_args()

    if args.op in ["train", "trainonly"]:
        if not os.path.exists(args.config_file):
            raise ValueError(f"Config file does not exist: {args.config_file}")
        if args.result_folder:
            raise ValueError(f"OP==train does not accept result_folder")
        if args.eval_model_name or args.eval_pos_file or args.eval_neg_file:
            raise ValueError(f"OP==train does not accept test-time eval pos/neg/model.")
        main(op=args.op, config_file=args.config_file,
             given_lr=float(args.trainer_lr) if args.trainer_lr is not None else args.trainer_lr,
             given_tbs=int(args.train_batch_size) if args.train_batch_size is not None else args.train_batch_size,
             given_user_text_file_name=args.user_text_file_name, given_item_text_file_name=args.item_text_file_name,
             given_train_neg_num=args.train_neg_num, given_T=args.T, given_lambda_flops = args.lambda_flops, given_margin = args.margin,
             given_eval_batch_size=args.eval_batch_size,
             given_subloss_fn = args.subloss_fn, given_sigmoid_output = args.no_sigmoid)
    elif args.op == "test":
        if not os.path.exists(join(args.result_folder, "config.json")):
            raise ValueError(f"Result folder does not exist: {args.config_file}")
        if args.config_file:
            raise ValueError(f"OP==test does not accept config_file")
        main(op=args.op, result_folder=args.result_folder,
             given_eval_model=args.eval_model_name,
             given_eval_pos_file=args.eval_pos_file,
             given_eval_neg_file=args.eval_neg_file, given_eval_batch_size=args.eval_batch_size)
