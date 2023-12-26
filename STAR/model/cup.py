import torch
import transformers

def dot_product(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(dim=-1)


class Cup(torch.nn.Module):
    def __init__(self, model_config, device):
        super(Cup, self).__init__()

        self.device = device

        self.bert = transformers.AutoModel.from_pretrained(model_config['pretrained_model'])
        self.agg_strategy = model_config['agg_strategy']

        user_layers = [torch.nn.Linear(768, model_config["user_k"][0], device=self.device)]
        for k in range(1, len(model_config["item_k"])):
            user_layers.append(
                torch.nn.Linear(model_config["item_k"][k - 1], model_config["item_k"][k], device=self.device))
        self.user_linear_layers = torch.nn.ModuleList(user_layers)

        item_layers = [torch.nn.Linear(768, model_config["user_k"][0], device=self.device)]
        for k in range(1, len(model_config["item_k"])):
            item_layers.append(
                torch.nn.Linear(model_config["item_k"][k - 1], model_config["item_k"][k], device=self.device))
        self.item_linear_layers = torch.nn.ModuleList(item_layers)

        self.bert.requires_grad_(True)
        freeze_modules = [self.bert.embeddings, self.bert.encoder.layer[:-1]]
        for module in freeze_modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, batch, mode="train"):
        u_input_ids = batch['user_input_ids']
        u_att_mask = batch['user_attention_mask']
        i_input_ids = batch['item_input_ids']
        i_att_mask = batch['item_attention_mask']

        output_u = self.bert(input_ids=u_input_ids, attention_mask=u_att_mask)
        if self.agg_strategy == "CLS":
            user_rep = output_u.last_hidden_state[:, 0]
        elif self.agg_strategy == "mean_last":
            tokens_embeddings = output_u.last_hidden_state
            mask = u_att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
            tokens_embeddings = tokens_embeddings * mask
            sum_tokons = torch.sum(tokens_embeddings, dim=1)
            summed_mask = torch.clamp(u_att_mask.sum(1).type(torch.float), min=1e-9)
            user_rep = (sum_tokons.T / summed_mask).T

        output_i = self.bert(input_ids=i_input_ids, attention_mask=i_att_mask)
        if self.agg_strategy == "CLS":
            item_rep = output_i.last_hidden_state[:, 0]
        elif self.agg_strategy == "mean_last":
            tokens_embeddings = output_i.last_hidden_state
            mask = i_att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
            tokens_embeddings = tokens_embeddings * mask
            sum_tokons = torch.sum(tokens_embeddings, dim=1)
            summed_mask = torch.clamp(i_att_mask.sum(1).type(torch.float), min=1e-9)
            item_rep = (sum_tokons.T / summed_mask).T

        for k in range(len(self.user_linear_layers) - 1):
            user_rep = torch.nn.functional.relu(self.user_linear_layers[k](user_rep))
        user_rep = self.user_linear_layers[-1](user_rep)

        for k in range(len(self.item_linear_layers) - 1):
            item_rep = torch.nn.functional.relu(self.item_linear_layers[k](item_rep))
        item_rep = self.item_linear_layers[-1](item_rep)

        q_num = user_rep.size(0)
        d_num = item_rep.size(0)
        doc_group_size = d_num // q_num
        result = dot_product(
            user_rep.unsqueeze(1), item_rep.view(doc_group_size, q_num, -1).transpose(0, 1)
        ).T.contiguous()

        if mode == "train":
            pos_scores = result[0].view((-1, 1))
            neg_scores = result[1:].view((-1, 1))
            return result.view((-1, 1)), 0, 0, pos_scores, neg_scores
        else:
            result = result.view((-1, 1))
            return result, 0, 0, None, None
