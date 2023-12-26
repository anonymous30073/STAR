from STAR.model.star import STAR
from STAR.model.bert_cross import BertCross
from STAR.model.cup import Cup

def get_model(config, device=None, dataset_config=None, exp_dir=None):
    if config['name'] == "STAR":
        model = STAR(model_config=config,
                       device=device)
    elif config['name'] == "Cup":
        model = Cup(model_config=config,
                       device=device)
    elif config['name'] == "BertCross":
        model = BertCross(model_config=config,
                       device=device,
                       dataset_config=dataset_config, exp_dir=exp_dir)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
