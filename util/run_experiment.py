import os
from pathlib import Path

import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Adam

from util.learning_pipeline import fit
from util.model_architecture import BaseLSTM, AdvancedLSTM
from util.noise_generator import pink_noise, white_noise, violet_noise, blue_noise, brownian_noise
from util.prepare_dataset import Dataset
from util.time_series_generator import TSGenerator


def run_pipeline(model_id, dataset_path, noise_type_name, params, experiment_id, model_name='base',
                 device=torch.device("mps")):
    dataset = Dataset(
        path=dataset_path,
        n_seq=params['n_seq'],
        n_batch=params['n_batch'],
        data_transformer=MinMaxScaler
    )

    input_dim = params['input_dim']
    hidden_dim = params['hidden_dim']
    layer_dim = params['layer_dim']
    output_dim = params['output_dim']
    n_epoch = params['n_epoch']
    lr = params['lr']

    train_loader, valid_loader = dataset.get_dataloaders()

    if model_name == 'base':
        model = BaseLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=output_dim
        ).to(device)

    elif model_name == 'advanced':
        model = AdvancedLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=output_dim
        ).to(device)

    else:
        model = BaseLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=output_dim
        ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    fit(
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        device,
        num_epochs=n_epoch,
        title='Model learned on ' + noise_type_name,
        experiment_id=experiment_id
    )

    output_file = f"model_{model_id}.pt"
    output_dir = Path(os.path.join('models'))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_scripted = torch.jit.script(model)
    model_scripted.save(output_dir / output_file)

    return model


def run_single_experiment(
        config
):
    print("Started generation of time series")
    noise_type_mapper = dict(
        pink=pink_noise,
        white=white_noise,
        blue=blue_noise,
        brownian=brownian_noise,
        violet=violet_noise
    )

    ts = TSGenerator(
        data_id=config['experiment']['data_config']['data_id'],
        noise_type_name=config['experiment']['data_config']['noise_type_name'],
        noise_func=noise_type_mapper[config['experiment']['data_config']['noise_type_name']],
        noise_coeff=config['experiment']['data_config']['noise_coeff'],
        size=config['experiment']['data_config']['size'],
        seed=config['experiment']['data_config']['data_id']

    )

    # print(ts.get_stats())
    # ts.visualize()

    while ts.get_stats()['pos_label_count'] < 1:
        print("data_id = ", config['experiment']['data_config']['data_id'], "| Regenerate with noise_func = x1.25")
        ts = TSGenerator(
            data_id=config['experiment']['data_config']['data_id'],
            noise_type_name=config['experiment']['data_config']['noise_type_name'],
            noise_func=noise_type_mapper[config['experiment']['data_config']['noise_type_name']],
            size=config['experiment']['data_config']['size'],
            seed=config['experiment']['data_config']['data_id'],
            noise_coeff=1.5
        )

    ts.publicate_stats(config['experiment']['experiment_id'])

    # ts.visualize()
    ts.save()

    print("Saved time series")
    print("Preparing to model fitting")

    params = dict(
        n_seq=config['experiment']['model_config']['n_seq'],
        n_batch=config['experiment']['model_config']['n_batch'],
        input_dim=config['experiment']['model_config']['input_dim'],
        hidden_dim=config['experiment']['model_config']['hidden_dim'],
        layer_dim=config['experiment']['model_config']['layer_dim'],
        output_dim=config['experiment']['model_config']['output_dim'],
        lr=config['experiment']['model_config']['lr'],
        n_epoch=config['experiment']['model_config']['n_epoch']
    )

    run_pipeline(
        model_id=config['experiment']['model_config']['model_id'],
        dataset_path=f"data/dataset_{config['experiment']['data_config']['data_id']}/{config['experiment']['data_config']['noise_type_name']}.csv",
        noise_type_name=f"{config['experiment']['data_config']['noise_type_name']} noise",
        params=params,
        model_name=config['experiment']['model_config']['model_name'],
        experiment_id=config['experiment']['experiment_id']
    )
