"""Tune Model.
- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import os
import yaml
import optuna
from datetime import datetime
import numpy as np
import timm

import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from optuna.pruners import HyperbandPruner
from subprocess import _args_from_interpreter_flags
import argparse


DATA_PATH = "../data"  # type your data path here that contains test, train and val directories
LOG_PATH = "./exp/latest"
RESULT_MODEL_PATH = "./exp/latest/best.pt" # result model will be saved in this path
os.makedirs(LOG_PATH, exist_ok=True)

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=50, high=50, step=50)
    img_size = trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = trial.suggest_int("batch_size", low=16, high=32, step=16)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
    }

class MyModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)

        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(in_features=n_features, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        return self.model(x)



def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    model_config: Dict[str, Any] = {}
    model_config["input_channel"] = 3
    # img_size = trial.suggest_categorical("img_size", [32, 64, 128])
    img_size = 32
    model_config["INPUT_SIZE"] = [img_size, img_size]
    model_config["depth_multiple"] = trial.suggest_categorical(
        "depth_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["width_multiple"] = trial.suggest_categorical(
        "width_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    #model_config["backbone"], module_info = search_model(trial)
    hyperparams = search_hyperparam(trial)

    model_instance = MyModel('efficientnet_b0', 6)# Model(model_config, verbose=True)
    model_instance.to(device)
    model_instance.model.to(device)

    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = DATA_PATH
    data_config["DATASET"] = "TACO"
    data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TEST"] = "simple_augment_test"
    data_config["AUG_TRAIN_PARAMS"] = {
        "n_select": hyperparams["n_select"],
    }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["VAL_RATIO"] = 0.8
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    mean_time = check_runtime(
        model_instance.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    model_info(model_instance, verbose=True)
    train_loader, val_loader, test_loader = create_dataloader(data_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )

    # 학습 시작전 모델 정보와 파라미터 정보 저장
    with open(os.path.join(LOG_PATH, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(LOG_PATH, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    trainer = TorchTrainer(
        model_instance.model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path=RESULT_MODEL_PATH
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model_instance.model, test_dataloader=val_loader)
    params_nums = count_model_params(model_instance)

    model_info(model_instance, verbose=True)

    # 새로운 학습 시작시 폴더를 옮겨주기
    if os.path.isfile(LOG_PATH + '/best.pt'): 
        modified = datetime.fromtimestamp(os.path.getmtime(LOG_PATH + '/best.pt'))
        new_log_dir = os.path.dirname(LOG_PATH) + '/' + str(trial.number) + '_' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(LOG_PATH, new_log_dir)
        os.makedirs(LOG_PATH, exist_ok=True)

    return f1_score, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, storage: str = None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    from optuna.integration import WeightsAndBiasesCallback
    wandb_kwargs = {"project":'lightweight_model', "entity":"boostcamp-nlp-06", "tags":["AutoML"], "group":"optuna"}
    wandbc = WeightsAndBiasesCallback(metric_name=["f1_score", "params_nums", "mean_time"], wandb_kwargs=wandb_kwargs)
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="tuning_eff_b0",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, device), n_trials=50, callbacks=[wandbc])

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="postgresql://optuna:optuna@nuda.iptime.org:5432/study", type=str, help="Optuna database storage path.")
    args = parser.parse_args()
    tune(args.gpu, storage=args.storage if args.storage != "" else None)