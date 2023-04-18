import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


class Dataset:
    def __init__(
            self,
            path,
            n_seq,
            n_batch,
            data_transformer,
            train_idx_coeff=0.7
    ):
        self.path = path
        self.n_seq = n_seq
        self.n_batch = n_batch
        self.data_transformer = data_transformer

        self.data = self.upload_data_from_csv()
        self.train_idx_coeff = train_idx_coeff

    def upload_data_from_csv(self):
        data = pd.read_csv(self.path)[["x", "y"]]
        return data

    def get_dataloaders(self, random_state=0):
        x_prepared, y_prepared = (
            torch.stack([torch.tensor(self.data["x"].shift(i).values) for i in range(1, self.n_seq + 1)], dim=1)
            [self.n_seq:].to(torch.float32),
            torch.tensor(self.data["y"].values)[self.n_seq:].to(torch.float32)
        )

        torch.manual_seed(random_state)
        pos_idx, neg_idx = (y_prepared == 1).nonzero(), (y_prepared == 0).nonzero()
        pos_count = pos_idx.shape[0]
        pos_idx, neg_idx = pos_idx, neg_idx[torch.randperm(neg_idx.shape[0])[:pos_count]]

        idx = torch.randperm(pos_count)
        train_idx, test_idx = (
            idx[:int(pos_count * self.train_idx_coeff)],
            idx[int(pos_count * self.train_idx_coeff):]
        )

        x_train_prepared, y_train_prepared = (
            torch.cat(
                [
                    x_prepared[pos_idx[train_idx]],
                    x_prepared[neg_idx[train_idx]]
                ]
            ),

            torch.cat(
                [
                    y_prepared[pos_idx[train_idx]],
                    y_prepared[neg_idx[train_idx]]
                ]
            )
        )

        permute_idx = torch.randperm(y_train_prepared.size(0))
        x_train_prepared = x_train_prepared[permute_idx] / np.pi
        y_train_prepared = y_train_prepared[permute_idx]

        x_test_prepared, y_test_prepared = (
            torch.cat(
                [
                    x_prepared[pos_idx[test_idx]],
                    x_prepared[neg_idx[test_idx]]
                ]
            ),

            torch.cat(
                [
                    y_prepared[pos_idx[test_idx]],
                    y_prepared[neg_idx[test_idx]]
                ]
            )
        )

        permute_idx = torch.randperm(y_test_prepared.size(0))
        x_test_prepared = x_test_prepared[permute_idx] / np.pi
        y_test_prepared = y_test_prepared[permute_idx]

        train_dataset = [(x_train_prepared[i].T, y_train_prepared[i]) for i in range(len(y_train_prepared))]
        test_dataset = [(x_test_prepared[i].T, y_test_prepared[i]) for i in range(len(y_test_prepared))]

        train_dataloader = DataLoader(train_dataset, batch_size=self.n_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=self.n_batch)

        return train_dataloader, test_dataloader
