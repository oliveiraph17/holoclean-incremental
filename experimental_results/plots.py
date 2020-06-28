import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def charts_setup():
    plt.rcParams.update({'font.size': 18})
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('tab10')


def load_quality_log_df(log_path, infer_mode):
    df_full = pd.read_csv(log_dir + log_path + '_quality_log.csv', sep=';')

    df_full.drop_duplicates(keep='last', inplace=True)
    if 'infer_mode' in df_full.columns:
        df = df_full.loc[df_full['infer_mode'] == infer_mode]
    else:
        df = df_full.loc[df_full['batch'] != 'batch']  # Drops redundant headers.

    # Casts first as float and then as int to avoid 'invalid literal' errors.
    df = df.astype({'batch': 'float64', 'dk_cells': 'float64', 'training_cells': 'float64', 'precision': 'float64',
                    'recall': 'float64', 'f1': 'float64',  'total_errors': 'float64', 'correct_repairs': 'float64',
                    'total_repairs_grdt': 'float64', 'repairs_on_correct_cells': 'float64'})
    df = df.astype({'batch': 'int32', 'dk_cells': 'int32', 'training_cells': 'int32', 'total_errors': 'int32',
                    'correct_repairs': 'int32', 'total_repairs_grdt': 'int32', 'repairs_on_correct_cells': 'int32'})
    df = df.sort_values(by=['batch'])
    df['remaining_errors'] = df['total_errors'] - df['correct_repairs'] + df['repairs_on_correct_cells']

    return df


def load_time_log_df(log_path, infer_mode):
    df_full = pd.read_csv(log_dir + log_path + '_time_log.csv', sep=';')

    df_full.drop_duplicates(keep='last', inplace=True)
    if 'infer_mode' in df_full.columns:
        df = df_full.loc[df_full['infer_mode'] == infer_mode]
    else:
        df = df_full.loc[df_full['batch'] != 'batch']  # Drops redundant headers.
    df = df[['batch', 'load_data', 'load_dcs', 'detect_errors', 'setup_domain', 'featurize_data', 'setup_model',
             'fit_model', 'infer_repairs', 'get_inferred_values', 'generate_repaired_dataset',
             'repaired_table_copy_time', 'save_stats_time']]
    df = df.astype('float64')
    df = df.set_index('batch')
    df.sort_index(inplace=True)

    return df


def get_cumulative_quality_df(df, approach):
    cum_df = df.copy()
    metrics = ['dk_cells', 'training_cells', 'correct_repairs', 'total_repairs_grdt',
               'repairs_on_correct_cells', 'total_errors']
    if not approach.startswith('HC-Acc') and not approach.startswith('IHC-Re'):
        metrics += ['remaining_errors']
    for metric in metrics:
        # Gets the accumulated values.
        cum_df[metric] = cum_df[metric].cumsum()

    cum_df['precision'] = cum_df['correct_repairs'] / cum_df['total_repairs_grdt']
    cum_df['recall'] = cum_df['correct_repairs'] / cum_df['total_errors']
    cum_df['f1'] = (2 * cum_df['precision'] * cum_df['recall']) / (cum_df['precision'] + cum_df['recall'])

    return cum_df


def get_cumulative_time_df(df):
    cum_df = df.copy()
    cum_df = cum_df.cumsum()

    return cum_df


def plot_quality_charts(infer_mode, approaches, plot_config):
    for approach in approaches:
        df = load_quality_log_df(approach['log_path'], infer_mode)
        if plot_config['cumulative']:
            df = get_cumulative_quality_df(df, approach['label'])

        for measure in plot_config['measures']:
            marker = approach['marker']
            if measure == 'total_errors':
                if approach['label'] == 'HC-Sep':
                    marker = ''
                else:
                    continue
            if approach['log_path'].startswith('full') or approach['log_path'].startswith('co_full'):
                plt.plot(100, df[measure].values, marker=marker, label=approach['label'] + '|' + measure)
            else:
                y_list = df[measure].values.tolist()
                x_list = [i for i in range(1, 101)]
                plt.plot(x_list, y_list, label=approach['label'] + '|' + measure, linestyle=approach['linestyle'],
                         marker=marker)

    if 'title' in plot_config:
        plt.title(plot_config['title'])
    if 'xlabel' in plot_config:
        plt.xlabel(plot_config['xlabel'])
    if 'ylabel' in plot_config:
        plt.ylabel(plot_config['ylabel'])
    if 'legend' in plot_config:
        plt.legend(**plot_config['legend'])
    if 'ylim' in plot_config:
        plt.ylim(**plot_config['ylim'])
    if 'yscale' in plot_config:
        plt.yscale(plot_config['yscale'])

    plt.show()


def plot_time_charts(infer_mode, approaches, plot_config):
    fig, axes = plt.subplots(nrows=len(approaches), ncols=1, sharex=True, sharey=True, squeeze=False)
    legend_labels = None

    for idx, approach in enumerate(approaches):
        df = load_time_log_df(approach['log_path'], infer_mode)
        if idx == 0:
            legend_labels = df.columns
        if plot_config['cumulative']:
            df = get_cumulative_time_df(df)

        # Reduces the number of bars.
        # batches = list(range(9, int(df.index.max()), 10))
        # if max_idx < 100 and max_idx not in batches:
        #     batches.append(max_idx - 1)
        # df_filtered = df.iloc[batches]

        df_filtered = df
        sum_bottom = [0 for pos in df_filtered.index]
        for measure in df_filtered.columns:
            curr_list = df_filtered[measure].values.tolist()
            axes[idx, 0].bar(df_filtered.index, curr_list, bottom=sum_bottom)
            sum_bottom = [old + curr for (old, curr) in zip(sum_bottom, curr_list)]

        if 'title' in plot_config:
            axes[idx, 0].set_title(plot_config['title'] + ' ' + approach['label'])
        if 'xlabel' in plot_config:
            axes[idx, 0].set_xlabel(plot_config['xlabel'])
        if 'ylabel' in plot_config:
            axes[idx, 0].set_ylabel(plot_config['ylabel'])
        if 'ylim' in plot_config:
            axes[idx, 0].set_ylim(**plot_config['ylim'])
        if 'yscale' in plot_config:
            axes[idx, 0].set_yscale(plot_config['yscale'])
        axes[idx, 0].plot()

    if 'legend' in plot_config:
        plt.legend(legend_labels, **plot_config['legend'])

    plt.show()


def plot_total_time_charts(infer_mode, approaches, plot_config):
    legend_labels = None

    for idx, approach in enumerate(approaches):
        df = load_time_log_df(approach['log_path'], infer_mode)
        if plot_config['cumulative']:
            df = get_cumulative_time_df(df)

        # Reduces the number of bars.
        # batches = list(range(9, int(df.index.max()), 10))
        # if max_idx < 100 and max_idx not in batches:
        #     batches.append(max_idx - 1)
        # df_filtered = df.iloc[batches]

        df = df.tail(1)
        if idx == 0:
            legend_labels = df.columns
            df_all = df.copy()
        else:
            df_all = pd.concat([df_all, df])

    approach_labels = [approach['label'] for approach in approaches]

    if 'yscale' in plot_config and plot_config['yscale'] == 'log':
        plt.bar(approach_labels, df_all.sum(axis=1))
        plt.yscale(plot_config['yscale'])
    else:
        sum_bottom = [0] * len(approach_labels)
        for measure in legend_labels:
            curr_list = df_all[measure].values.tolist()
            plt.bar(approach_labels, curr_list, bottom=sum_bottom)
            sum_bottom = [old + curr for (old, curr) in zip(sum_bottom, curr_list)]

    if 'title' in plot_config:
        plt.title(plot_config['title'])
    if 'xlabel' in plot_config:
        plt.xlabel(plot_config['xlabel'])
    if 'ylabel' in plot_config:
        plt.ylabel(plot_config['ylabel'])
    if 'ylim' in plot_config:
        plt.ylim(**plot_config['ylim'])
    plt.plot()

    if 'legend' in plot_config:
        plt.legend(legend_labels, **plot_config['legend'])

    plt.show()


def plot_memory_chart(approaches, plot_config):
    total_time = []
    approach_labels = [approach['label'] for approach in approaches]
    for approach in approaches:
        df = pd.read_csv(log_dir + approach['log_path'] + '_memory_log.csv', names=['time', 'garbage'], sep='\s+')
        total_time.append(df['time'].mean())

    plt.bar(approach_labels, total_time)

    if 'title' in plot_config:
        plt.title(plot_config['title'])
    if 'xlabel' in plot_config:
        plt.xlabel(plot_config['xlabel'])
    if 'ylabel' in plot_config:
        plt.ylabel(plot_config['ylabel'])
    if 'ylim' in plot_config:
        plt.ylim(**plot_config['ylim'])

    plt.show()


dataset_name = 'hospital_shuffled'
log_dir = os.environ['HOLOCLEANHOME'] + '/experimental_results_husky/' + dataset_name + '/'
infer_mode = 'dk'
approaches = [
    # {'log_path': 'co_full',
    #  'label': 'HC-Full',
    #  'linestyle': '',
    #  'marker': '*'},
    {'log_path': 'full_dk',
     'label': 'HC-Full',
     'linestyle': '',
     'marker': '*'},
    # {'log_path': 'co_a_no_pruning',
    #  'label': 'co_A no pruning',
    #  'linestyle': '-',
    #  'marker': '1'},
    # {'log_path': 'co_a',
    #  'label': 'HC-Sep',
    #  'linestyle': '-',
    #  'marker': '1'},
    {'log_path': 'a_dk_100b',
     'label': 'HC-Sep',
     'linestyle': '-',
     'marker': '1'},
    {'log_path': 'b_dk_100b',
     'label': 'HC-Acc',
     'linestyle': '-',
     'marker': '2'},
    # {'log_path': 'b_num_dk_100b',
    #  'label': 'B_num',
    #  'linestyle': '_',
    #  'marker': '2'},

    # no grouping
    # {'log_path': 'c_dk_100b',
    #  'label': 'IHC',
    #  'linestyle': '-',
    #  'marker': ''},
    # {'log_path': 'c_dk_100b_ikl001',
    #  'label': 'IHC iKL0.01',
    #  'linestyle': '-.',
    #  'marker': ''},
    # {'log_path': 'cn_dk_100b_ikl001',
    #  'label': 'IHCNew iKL0.01',
    #  'linestyle': '-.',
    #  'marker': ''},
    # {'log_path': 'c_dk_100b_ikl005',
    #  'label': 'IHC iKL0.05',
    #  'linestyle': '--',
    #  'marker': ''},
    # {'log_path': 'c_dk_100b_ikl01',
    #  'label': 'IHC iKL0.1',
    #  'linestyle': ':',
    #  'marker': ''},
    # {'log_path': 'c_dk_100b_wkl001',
    #  'label': 'IHC wKL0.01',
    #  'linestyle': '-.',
    #  'marker': ''},
    # {'log_path': 'cn_dk_100b_wkl001',
    #  'label': 'IHCNew wKL0.01',
    #  'linestyle': '-.',
    #  'marker': ''},
    {'log_path': 'c_dk_100b_wkl005',
     'label': 'IHC wKL0.05',
     'linestyle': '--',
     'marker': ''},
    # {'log_path': 'c_dk_100b_wkl01',
    #  'label': 'IHC wKL0.1',
    #  'linestyle': ':',
    #  'marker': ''},

    # pair_corr grouping
    # {'log_path': 'c_dk_100b_pc097_ikl001',
    #  'label': 'IHC pc0.97 iKL0.01',
    #  'linestyle': '-.',
    #  'marker': '.'},
    # {'log_path': 'c_dk_100b_pc097_ikl005',
    #  'label': 'IHC pc0.97 iKL0.05',
    #  'linestyle': '--',
    #  'marker': '.'},
    # {'log_path': 'c_dk_100b_pc097_ikl01',
    #  'label': 'IHC pc0.97 iKL0.1',
    #  'linestyle': ':',
    #  'marker': '.'},
    # {'log_path': 'c_dk_100b_pc097',
    #  'label': 'IHC pc0.97',
    #  'linestyle': '-',
    #  'marker': '.'},
    # {'log_path': 'c_dk_100b_pc097_wkl001',
    #  'label': 'IHC pc0.97 wKL0.01',
    #  'linestyle': '-.',
    #  'marker': '.'},
    # {'log_path': 'c_dk_100b_pc097_wkl005',
    #  'label': 'IHC pc0.97 wKL0.05',
    #  'linestyle': '--',
    #  'marker': '.'},
    # {'log_path': 'c_dk_100b_pc097_wkl01',
    #  'label': 'IHC pc0.97 wKL0.1',
    #  'linestyle': ':',
    #  'marker': '.'},

    # sim_corr grouping
    # {'log_path': 'c_dk_100b_sc0005_ikl001',
    #  'label': 'IHC sc0.005 iKL0.01',
    #  'linestyle': '-.',
    #  'marker': '+'},
    # {'log_path': 'c_dk_100b_sc0005_ikl005',
    #  'label': 'IHC sc0.005 iKL0.05',
    #  'linestyle': '--',
    #  'marker': '+'},
    # {'log_path': 'c_dk_100b_sc0005_ikl01',
    #  'label': 'IHC sc0.005 iKL0.1',
    #  'linestyle': ':',
    #  'marker': '+'},
    # {'log_path': 'c_dk_100b_sc0005',
    #  'label': 'IHC sc0.005',
    #  'linestyle': '-',
    #  'marker': '+'},
    # {'log_path': 'c_dk_100b_sc0005_wkl001',
    #  'label': 'IHC sc0.005 wKL0.01',
    #  'linestyle': '-.',
    #  'marker': '+'},
    # {'log_path': 'c_dk_100b_sc0005_wkl005',
    #  'label': 'IHC sc0.005 wKL0.05',
    #  'linestyle': '--',
    #  'marker': '+'},
    # {'log_path': 'c_dk_100b_sc0005_wkl01',
    #  'label': 'IHC sc0.005 wKL0.1',
    #  'linestyle': ':',
    #  'marker': '+'},

    # C approach repairing previous errors.
    # {'log_path': 'cb_dk_100b_ikl001',
    #  'label': 'IHC-Re ikl0.01',
    #  'linestyle': '-.',
    #  'marker': ''},
    # {'log_path': 'cb_dk_100b_ikl005',
    #  'label': 'IHC-Re ikl0.05',
    #  'linestyle': '--',
    #  'marker': ''},
    # {'log_path': 'cb_dk_100b_ikl01',
    #  'label': 'IHC-Re ikl0.1',
    #  'linestyle': ':',
    #  'marker': ''},
    # {'log_path': 'cb_dk_100b',
    #  'label': 'IHC-Re',
    #  'linestyle': '-.',
    #  'marker': ''},
    # {'log_path': 'cb_dk_100b_wkl001',
    #  'label': 'IHC-Re wkl0.01',
    #  'linestyle': '-.',
    #  'marker': ''},
    {'log_path': 'cb_dk_100b_wkl005',
     'label': 'IHC-Re wkl0.05',
     'linestyle': '--',
     'marker': ''},
    # {'log_path': 'cb_dk_100b_wkl01',
    #  'label': 'IHC-Re wkl0.1',
    #  'linestyle': ':',
    #  'marker': ''},

    # C approach grouping + repairing previous errors.
    # {'log_path': 'cb_dk_100b_pc097_ikl001',
    #  'label': 'IHC-Re pc0.97 ikl0.01',
    #  'linestyle': '-.',
    #  'marker': '.'},
    # {'log_path': 'cb_dk_100b_pc097_ikl005',
    #  'label': 'IHC-Re pc0.97 ikl0.05',
    #  'linestyle': '--',
    #  'marker': '.'},
    # {'log_path': 'cb_dk_100b_pc097',
    #  'label': 'IHC-Re pc0.97',
    #  'linestyle': '-',
    #  'marker': '.'},
    # {'log_path': 'cb_dk_100b_pc097_wkl001',
    #  'label': 'IHC-Re pc0.97 wkl0.01',
    #  'linestyle': '-.',
    #  'marker': '.'},
    # {'log_path': 'cb_dk_100b_pc097_wkl005',
    #  'label': 'IHC-Re pc0.97 wkl0.05',
    #  'linestyle': '--',
    #  'marker': '.'},
    # {'log_path': 'cb_dk_100b_sc0005_ikl001',
    #  'label': 'IHC-Re sc0.005 iKL0.01',
    #  'linestyle': '-.',
    #  'marker': '+'},
    # {'log_path': 'cb_dk_100b_sc0005_ikl005',
    #  'label': 'IHC-Re sc0.005 iKL0.05',
    #  'linestyle': '--',
    #  'marker': '+'},
    # {'log_path': 'cb_dk_100b_sc0005',
    #  'label': 'IHC-Re sc0.005',
    #  'linestyle': '-',
    #  'marker': '+'},
    # {'log_path': 'cb_dk_100b_sc0005_wkl001',
    #  'label': 'IHC-Re sc0.005 wKL0.01',
    #  'linestyle': '-.',
    #  'marker': '+'},
    # {'log_path': 'cb_dk_100b_sc0005_wkl005',
    #  'label': 'IHC-Re sc0.005 wKL0.05',
    #  'linestyle': '--',
    #  'marker': '+'},
]

plot_config_PxR = {
    'cumulative': True,
    'measures': ['precision', 'recall', 'f1'],
    'title': dataset_name,
    'xlabel': '% of the Dataset',
    'ylabel': 'Value',
    'legend': {'loc': 'best'},
}
plot_config_F1 = {
    'cumulative': True,
    'measures': ['f1'],
    'title': dataset_name,
    'xlabel': '% of the Dataset',
    'ylabel': 'F1',
    'legend': {'loc': 'best'},
}
plot_config_training_cells = {
    'cumulative': True,
    'measures': ['training_cells'],
    'title': dataset_name,
    'xlabel': '% of the Dataset',
    'ylabel': '# of Training Instances',
    'legend': {'loc': 'best'},
    'yscale': 'log'
}
plot_config_remaining_errors = {
    'cumulative': True,
    'measures': ['remaining_errors', 'total_errors'],
    'title': dataset_name,
    'xlabel': '% of the Dataset',
    'ylabel': '# of Remaining Errors',
    'legend': {'loc': 'best'},
}
plot_config_time = {
    'cumulative': True,
    'title': dataset_name,
    'xlabel': '% of the Dataset',
    'ylabel': 'Total Time (sec)',
    'legend': {'loc': 'best'},
}
plot_config_total_time = {
    'cumulative': True,
    'title': dataset_name,
    'xlabel': 'Approach',
    'ylabel': 'Total Time (sec)',
    'legend': {'loc': 'best'},
}
plot_config_total_time_log = {
    'cumulative': True,
    'title': dataset_name,
    'xlabel': 'Approach',
    'ylabel': 'Total Time (sec)',
    'yscale': 'log',
    'ylim': {'bottom': 1},
}
plot_config_memory = {
    'title': dataset_name,
    'xlabel': 'Approach',
    'ylabel': 'Memory Consumption (GB)',
    'legend': {'loc': 'best'},
}

charts_setup()
plot_quality_charts(infer_mode, approaches, plot_config_F1)
plot_quality_charts(infer_mode, approaches, plot_config_PxR)
plot_quality_charts(infer_mode, approaches, plot_config_remaining_errors)
plot_quality_charts(infer_mode, approaches, plot_config_training_cells)
plot_time_charts(infer_mode, approaches, plot_config_time)
plot_total_time_charts(infer_mode, approaches, plot_config_total_time)
plot_total_time_charts(infer_mode, approaches, plot_config_total_time_log)
plot_memory_chart(approaches, plot_config_memory)
