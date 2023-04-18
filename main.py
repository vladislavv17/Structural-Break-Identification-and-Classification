import json
import os
from pathlib import Path

import numpy as np

from util.run_experiment import run_single_experiment


def run_experiments(n_experiments):
    model_params = dict(
        n_seq=[64 for i in range(n_experiments)],
        n_batch=[16 for i in range(n_experiments)],
        input_dim=[1 for i in range(n_experiments)],
        hidden_dim=[2 ** (np.random.randint(4) + 4) for i in range(n_experiments)],
        layer_dim=[2 ** (np.random.randint(3) + 1) for i in range(n_experiments)],
        output_dim=[1 for i in range(n_experiments)],
        lr=[float(1e-4) for i in range(n_experiments)],
        n_epoch=[400 for i in range(n_experiments)],
        model_id=[i for i in range(n_experiments)],
        model_name=[['base', 'base'][np.random.randint(2)] for _ in range(n_experiments)]
    )

    noise_type_names = ['violet']
    mapping_noise_and_coeff = dict(
        white=0.69,
        violet=1.6
    )

    noise_type_name_params = [noise_type_names[np.random.randint(len(noise_type_names))] for i in range(n_experiments)]
    noise_coeff_params = [mapping_noise_and_coeff[noise] for noise in noise_type_name_params]
    data_params = dict(
        data_id=[i for i in range(n_experiments)],
        noise_type_name=noise_type_name_params,
        size=[int(1e6) for _ in range(n_experiments)],
        noise_coeff=noise_coeff_params
    )

    for experiment_id in range(n_experiments):
        model_config = dict()
        data_config = dict()

        for k, v in model_params.items():
            model_config[k] = model_params[k][experiment_id]

        for k, v in data_params.items():
            data_config[k] = data_params[k][experiment_id]

        experiment_config = dict(
            experiment=dict(
                model_config=model_config,
                data_config=data_config,
                experiment_id=experiment_id
            )
        )

        output_file = 'config.json'
        output_dir = Path(os.path.join('experiments', f'experiment_{experiment_id}'))
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / output_file, 'w') as fp:
            json.dump(experiment_config, fp, indent=4)

        print("Starting experiment #", experiment_id)

        run_single_experiment(experiment_config)

        print("Finished experiment #", experiment_id)
        print("All results saved to experiment directory")


if __name__ == '__main__':
    run_experiments(1)
