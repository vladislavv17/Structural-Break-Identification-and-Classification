import json
import os.path

import torch
import torch.nn as nn
from IPython.display import clear_output
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    f1_score
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn, device: torch.device):
    model.train()

    total_loss = 0
    total_correct = 0
    counted_metrics = dict(
        accuracy=0,
        precision=0,
        recall=0,
        auc_roc=0,
        auc_pr=0,
        f1_score=0
    )

    TP, TN, FP, FN = 0, 0, 0, 0

    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)
        loss.backward()

        y_pred_proba = torch.sigmoid(output).detach().to(torch.device("cpu")).numpy()
        y_pred = y_pred_proba.copy()

        y_pred[y_pred >= 0.5] = 1.0
        y_pred[y_pred < 0.5] = 0.0

        y_true = y.detach().to(torch.device("cpu")).numpy()

        TP += y_pred[(y_pred == 1) & (y_true == 1)].shape[0]
        TN += y_pred[(y_pred == 0) & (y_true == 0)].shape[0]
        FP += y_pred[(y_pred == 1) & (y_true == 0)].shape[0]
        FN += y_pred[(y_pred == 0) & (y_true == 1)].shape[0]

        total_correct += (y_pred == y_true).sum()

        counted_metrics['accuracy'] += total_correct
        counted_metrics['precision'] += precision_score(y_true, y_pred, zero_division=0)
        counted_metrics['recall'] += recall_score(y_true, y_pred, zero_division=0)

        try:
            counted_metrics['auc_roc'] += roc_auc_score(y_true, y_pred_proba)
        except:
            pass

        try:
            counted_metrics['auc_pr'] += average_precision_score(y_true, y_pred_proba)
        except:
            pass

        try:
            counted_metrics['f1_score'] += f1_score(y_true, y_pred, zero_division=0)
        except:
            pass

        total_loss += loss.item()

        optimizer.step()

    for metric, value in counted_metrics.items():
        counted_metrics[metric] /= len(data_loader)

    counted_metrics['accuracy'] = total_correct / len(data_loader.dataset)
    counted_metrics['precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    counted_metrics['recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0
    print("Train", f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")
    return total_loss / len(data_loader), counted_metrics


@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn, device: torch.device):
    model.eval()

    total_loss = 0
    total_correct = 0
    counted_metrics = dict(
        accuracy=0,
        precision=0,
        recall=0,
        auc_roc=0,
        auc_pr=0,
        f1_score=0
    )
    TP, TN, FP, FN = 0, 0, 0, 0

    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)

        output = model(x)

        loss = loss_fn(output, y)
        total_loss += loss.item()
        y_pred_proba = torch.sigmoid(output).detach().to(torch.device("cpu")).numpy()
        y_pred = y_pred_proba.copy()

        y_pred[y_pred >= 0.5] = 1.0
        y_pred[y_pred < 0.5] = 0.0

        y_true = y.detach().to(torch.device("cpu")).numpy()

        TP += y_pred[(y_pred == 1) & (y_true == 1)].shape[0]
        TN += y_pred[(y_pred == 0) & (y_true == 0)].shape[0]
        FP += y_pred[(y_pred == 1) & (y_true == 0)].shape[0]
        FN += y_pred[(y_pred == 0) & (y_true == 1)].shape[0]

        total_correct += (y_pred == y_true).sum()

        counted_metrics['accuracy'] += total_correct
        counted_metrics['precision'] += precision_score(y_true, y_pred, zero_division=0)
        counted_metrics['recall'] += recall_score(y_true, y_pred, zero_division=0)
        try:
            counted_metrics['auc_roc'] += roc_auc_score(y_true, y_pred_proba)
        except:
            pass

        try:
            counted_metrics['auc_pr'] += average_precision_score(y_true, y_pred_proba)
        except:
            pass

        try:
            counted_metrics['f1_score'] += f1_score(y_true, y_pred, zero_division=0)
        except:
            pass

    for metric, value in counted_metrics.items():
        counted_metrics[metric] /= len(data_loader)

    counted_metrics['accuracy'] = total_correct / len(data_loader.dataset)
    counted_metrics['precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    counted_metrics['recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("Eval", f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")

    return total_loss / len(data_loader), counted_metrics


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')


def plot_stats(
        train_loss: list[float],
        valid_loss: list[float],
        train_metrics_history,
        valid_metrics_history,
        title: str,
        path_to_save_plot_dir=None
):
    plt.figure(figsize=(16, 8))
    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()

    if path_to_save_plot_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path_to_save_plot_dir, "loss_history.svg"))

    n, m = (len(train_metrics_history) // 2) + 1, 2

    fig, axes = plt.subplots(n, m, figsize=(16, 8))
    metric_names = train_metrics_history.keys()

    for i in range(n):
        for j in range(m):

            if len(metric_names) == (i * m + j):
                break

            metric_name = list(metric_names)[i * m + j]

            axes[i][j].set_title(title + ' ' + metric_name)
            axes[i][j].plot(train_metrics_history[metric_name], label=f'Train {metric_name}')
            axes[i][j].plot(valid_metrics_history[metric_name], label=f'Valid {metric_name}')
            axes[i][j].legend()

    fig.tight_layout()

    if path_to_save_plot_dir is None:
        plt.show()
    else:
        fig.savefig(os.path.join(path_to_save_plot_dir, "metrics_history.svg"))


def fit(model, train_loader, valid_loader, optimizer, loss_fn, device, num_epochs, title, experiment_id):
    train_loss_history, valid_loss_history = [], []

    train_metrics_history = dict(accuracy=[], precision=[], recall=[], auc_roc=[], auc_pr=[], f1_score=[])
    valid_metrics_history = dict(accuracy=[], precision=[], recall=[], auc_roc=[], auc_pr=[], f1_score=[])

    for epoch in range(num_epochs):
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)

        for k, v in train_metrics.items():
            train_metrics_history[k] += [float(v)]

        for k, v in valid_metrics.items():
            valid_metrics_history[k] += [float(v)]

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        clear_output()

    with open(f'experiments/experiment_{experiment_id}/metric_history.json', 'w') as fp:
        json.dump(dict(
            train_loss_history=list(map(float, train_loss_history)),
            valid_loss_history=list(map(float, valid_loss_history)),
            train_metrics_history=train_metrics_history,
            valid_metrics_history=valid_metrics_history
        ), fp, indent=4
        )

    plot_stats(
        train_loss_history, valid_loss_history,
        train_metrics_history, valid_metrics_history,
        title,
        f'experiments/experiment_{experiment_id}'
    )
