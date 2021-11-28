"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import optuna

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info

import wandb
wandb.login()

class ConvNets(nn.Module):
    def __init__(self, trial):
        super().__init__()
        self.layer = self.define_model(trial)
        self.in_features = None
        self.in_channel = None
        #self.out_layer = self.output_model(trial)

    def define_model(self, trial):
        n_layers = 2#trial.suggest_int("n_layers", 1, 2)
        layers = []
        self.in_features = 224
        self.in_channel = 3

        for i in range(n_layers):
            out_channel = trial.suggest_int(f"conv_c{i}", 3, 128),
            kernel_size = trial.suggest_int(f"conv_k{i}", 3, 5)
            stride = trial.suggest_int(f"conv_s{i}", 1, 2)
            padding = trial.suggest_int(f"conv_p{i}", 0, 2)
            layers.append(nn.Conv2d(self.in_channel, out_channel[0], kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channel[0]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

            out_features = (self.in_features - kernel_size + 2*padding) // (stride) + 1
            out_features = out_features // 2
            print(f"conv_layer_{i}:", self.in_channel, out_channel[0], self.in_features, out_features, kernel_size, stride, padding)
            wandb.config.update({f"l{i}_conv_channel":out_channel[0], f"l{i}_conv_out": out_features, f"l{i}_conv_k": kernel_size, f"l{i}_conv_s": stride, f"l{i}_conv_p": padding})
            self.in_features = out_features
            self.in_channel = out_channel[0]

        print('pow(self.in_features,2)* self.in_channel:',pow(self.in_features,2)* self.in_channel)
        wandb.config.update({"l3_Linear":pow(self.in_features,2)* self.in_channel})
        layers.append(nn.Flatten())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        layers.append(nn.Linear(pow(self.in_features,2)* self.in_channel, 1000))
        layers.append(nn.Dropout(p))
        layers.append(nn.Linear(1000, 6))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x



def train(
    trial,
    #model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    # with open(os.path.join(log_dir, "model.yml"), "w") as f:
    #     yaml.dump(model_config, f, default_flow_style=False)

    # model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    # print(f"Model save path: {model_path}")
    #if os.path.isfile(model_path):
    #    model_instance.model.load_state_dict(
    #       torch.load(model_path, map_location=device)
    #    )
    #model_instance.model.to(device)

    model  = ConvNets(trial).to(device)
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    #optimizer_name = trial.suggest_categorical("opmizer", ["Adam", "RMSprop", "SGD"])
    #lr = trial.suggest_float("lr", 1e-5, data_config["INIT_LR"])
    #optimizer = getattr(optim, optimizer_name)(model_instance.model.parameters(), lr=lr)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=data_config["INIT_LR"]
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=1e-6,
        max_lr=1e-4,
        step_size_up=10,
        step_size_down=None,
        cycle_momentum=False,
        mode='triangular',
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        trial,
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/example.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    def objective(trial):
        wandb.init(
            project='lightweight_model', 
            entity="boostcamp-nlp-06", 
            tags=["optuna"],
            group="Conv2Model",
            allow_val_change=True
            )
        test_loss, test_f1, test_acc = train(
            trial,
            #model_config=model_config,
            data_config=data_config,
            log_dir=log_dir,
            fp16=data_config["FP16"],
            device=device,
        )
        wandb.finish()
        return test_f1

    study = optuna.create_study(directions=["maximize"])
    study.optimize(objective, n_trials=100)
