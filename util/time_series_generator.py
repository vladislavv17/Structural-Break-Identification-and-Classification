import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt


class TSGenerator:
    def __init__(self, data_id, noise_type_name, noise_func, size, noise_coeff=70e-2, seed=0):
        self.size = size
        self.noise_func = noise_func(N=size, seed=seed)
        self.noise_type_name = noise_type_name
        self.data_id = data_id
        self.marked_table = None
        self.noise_coeff = noise_coeff

    def get_data(self):
        x_t = np.pi / 1e6
        data = [x_t]

        errors = self.noise_func
        errors *= self.noise_coeff

        for i in range(errors.size):
            x_t = x_t + math.sin(x_t) + errors[i]
            data += [x_t]

        return data

    def get_marks(self):

        if self.marked_table is not None:
            return self.marked_table

        marked_table = pd.DataFrame()
        data = self.get_data()

        marked_table['x'] = pd.Series(data)
        data = self.apply_rolling_window(data, 10)

        marked_table['level'] = pd.Series([
            sorted(
                (
                    np.pi * (int(data[i] / np.pi) - 1),
                    np.pi * (int(data[i] / np.pi)),
                    np.pi * (int(data[i] / np.pi) + 1)
                ),
                key=lambda t: abs(t - data[i])
            )[0]
            for i in range(len(data))
        ])

        marked_table['y'] = self.label_data(marked_table, mode='kirill')

        self.marked_table = marked_table
        return marked_table

    def apply_rolling_window(self, data, window_size=10):
        return list(pd.Series(data).rolling(window=window_size).mean().fillna(0).values)

    def label_data(self, data, mode='default'):
        data = data.copy()
        if mode == 'default':
            return (data['level'].shift(1, fill_value=0) != data['level']) * 1

        if mode == 'ruptures':
            sigma = math.pi / 2
            model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
            algo = rpt.Window(width=20, model=model).fit(data['x'].values)
            bkps_idx = algo.predict(epsilon=self.size * sigma ** 2)[:-1]
            data.loc[bkps_idx, 'y'] = 1
            data = data.fillna(0)

            return data['y']

        if mode == 'kirill':
            return self.find_change_points(data, msd=20)

        return None

    def find_change_points(self, data, msd=20):
        data = data.copy()
        ts = data['x'].copy()
        last_stable_level = round(ts[0] / np.pi)
        last_unstable_level = 0
        last_stable_ix = 0
        last_unstable_ix = 0

        min_stability_duration = msd
        stability_duration = 0

        change_points = []

        for i, xt in enumerate(ts):
            if i == 0:
                continue
            if round(xt / np.pi) % 2 == 0:
                # unstable уровень - пока здесь ничего не делаю
                last_unstable_level = round(xt / np.pi)
                last_unstable_ix = i
            else:
                # stable
                if last_stable_level != round(xt / np.pi):
                    # прыгнули на новый устойчивый уровень -> был change_point
                    if stability_duration >= min_stability_duration:
                        # добавляем change point только если уровень устойчивости длился больше некоторого T
                        change_points.append(last_stable_ix)
                    stability_duration = 1
                    last_stable_level = round(xt / np.pi)
                    last_stable_ix = i
                else:
                    # остались на прежнем уровне, либо вернулись с неустойчивой точки
                    last_stable_ix = i
                    stability_duration += 1

        data.loc[change_points, 'y'] = 1
        data = data.fillna(0)
        return data['y']

    def get_stats(self):
        if self.marked_table is None:
            self.get_marks()

        distances = pd.DataFrame(self.marked_table['y'].cumsum()).reset_index() \
            .groupby('y') \
            .count() \
            .values \
            .squeeze()

        return dict(
            pos_label_count=float(self.marked_table['y'].sum()),
            neg_label_count=float(self.marked_table['y'].count() - self.marked_table['y'].sum()),
            distance_min=float(distances.min()),
            distance_max=float(distances.max()),
            distance_median=float(np.percentile(distances, 50)),
            distance_95=float(np.percentile(distances, 90)),
            # distances=distances,
        )

    def visualize(self):
        if self.marked_table is None:
            self.get_marks()
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.set_yticks(
            [
                round(i * math.pi, 2)
                for i in range(
                int(self.marked_table['level'].min() / math.pi),
                int(self.marked_table['level'].max() / math.pi) + 1
            )
            ],
            minor=False
        )

        axes.set_yticklabels(
            [
                f"{i} $\pi$"
                for i in range(
                int(self.marked_table['level'].min() / math.pi),
                int(self.marked_table['level'].max() / math.pi) + 1
            )
            ],
            minor=False
        )

        axes.plot(self.marked_table['x'], zorder=0)
        axes.scatter(self.marked_table.loc[self.marked_table.y == 1].index,
                     self.marked_table.loc[self.marked_table.y == 1, 'x'], color='r')
        axes.yaxis.grid(True, which='major', zorder=10)

        plt.show()

    def save(self):
        if self.marked_table is None:
            self.get_marks()

        output_file = self.noise_type_name + '.csv'
        output_dir = Path(os.path.join('data', f'dataset_{self.data_id}'))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.marked_table.to_csv(output_dir / output_file)

    def publicate_stats(self, experiment_id):
        with open(f"experiments/experiment_{experiment_id}/data_stats.json", 'w') as fp:
            json.dump(self.get_stats(), fp, indent=4)
