#!/usr/bin/env python
import os
from glob import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baselines.eval_task import main as eval_main
from baselines.eval_task import construct_parser as eval_construct_parser


def plot_log_file(log_file, trn_kwargs=None, vld_kwargs=None):
    df = pd.read_csv(log_file)
    trn_df = df.loc[df['mode'] == 'train']
    min_trn_loss = trn_df['avg_loss'].min()
    min_trn_acc = trn_df['avg_acc'].max()
    vld_df = df.loc[df['mode'] == 'test']
    min_vld_loss = vld_df['avg_loss'].min()
    min_vld_acc = vld_df['avg_acc'].max()
    if trn_kwargs is None:
        trn_kwargs = dict(
            label=f'({min_trn_loss:.4f}, {min_trn_acc:.1f})')
    elif 'label' in trn_kwargs:
        trn_kwargs['label'] += f' ({min_trn_loss:.4f}, {min_trn_acc:.1f})'
    if vld_kwargs is None:
        vld_kwargs = dict(label=f'({min_vld_loss:.4f}, {min_vld_acc:.1f})')
    elif 'label' in vld_kwargs:
        vld_kwargs['label'] += f' ({min_vld_loss:.4f}, {min_vld_acc:.1f})'
    plt.plot(trn_df['epoch'], trn_df['avg_loss'], **trn_kwargs)
    plt.plot(vld_df['epoch'], vld_df['avg_loss'], **vld_kwargs)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    return trn_df, vld_df, (min_trn_loss, min_trn_acc, min_vld_loss, min_vld_acc)


def plot_task_losses(output_dir, task_name, settings,
                     setting_names, save_plots=False):
    idx_cols = ['task_name', 'expt_name'] + setting_names + ['repeat']
    val_cols = ['min_trn_loss',  'min_vld_loss',
                'min_trn_acc', 'min_vld_acc']
    res = pd.DataFrame(columns=idx_cols + val_cols, dtype=float)
    res.set_index(idx_cols, drop=True, inplace=True)
    
    summary_fig = plt.figure()
    plt.title(f'{task_name} vld loss curves')
    
    alphas = [.3, .5, .7]
    for setting in settings:
        setting_fig = plt.figure()
        for repeat in range(3):
            expt_name = f'{task_name}__{"_".join(setting)}_{repeat}'
            log_file = f'{output_dir}/{task_name}/{expt_name}.log'
            
            plt.figure(setting_fig.number)
            try:
                trn_df, vld_df, losses = plot_log_file(
                    log_file,
                    trn_kwargs=dict(label=f'train {repeat}',
                                    color='C0', alpha=alphas[repeat]),
                    vld_kwargs=dict(label=f'valid {repeat}',
                                    color='C1', alpha=alphas[repeat])
                )
                    
            except pd.errors.EmptyDataError:
                print(f'{expt_name}.log found but empty')
                continue
            except FileNotFoundError:
                print(f'{expt_name}.log not found')
                continue
            min_trn_loss, min_trn_acc, min_vld_loss, min_vld_acc = losses
            idx = (task_name, expt_name) + setting + (str(repeat),)
            res.loc[idx, :] = [min_trn_loss, min_vld_loss, 
                               min_trn_acc, min_vld_acc]
            
            plt.figure(summary_fig.number)
            plt.plot(vld_df['epoch'], vld_df['avg_loss'],
                     color='C1', alpha=.2)
        
        plt.figure(setting_fig.number) 
        if save_plots:
            plt.savefig(f'{save_plots}/{task_name}__{"_".join(setting)}.png',
                        dpi=300)
            plt.savefig(f'{save_plots}/{task_name}__{"_".join(setting)}.pdf',
                        dpi=300)
        plt.title(f'{task_name}__{"_".join(setting)}')
        setting_fig.show()
    
    plt.figure(summary_fig.number)
    if save_plots:
        plt.savefig(f'{save_plots}/{task_name}__summary.png',
                    dpi=300)
        plt.savefig(f'{save_plots}/{task_name}__summary.pdf',
                    dpi=300)
    summary_fig.show()
    return res


def get_settings(output_dir, task_name):
    files = glob(f'{output_dir}/{task_name}/*')
    settings = [tuple(ff.split('__')[-1].rsplit('_', 1)[0].split('_'))
                for ff in files]
    settings = set(settings)  # easy dedup
    settings = list(settings)
    settings.sort()
    return settings


def round_to_n(x, n=3):
    if pd.isnull(x):
        return np.nan
    elif x == 0:
        return 0
    else:
        return np.round(x, -(np.floor(np.log10(x))).astype(int) + (n - 1))


def plot_confusion(confusion_mat, save_plots=False, ax=None):
    if ax is None:
        ax = plt.gca()
    degs = ['none', 'pitch_shift', 'time_shift', 'onset_shift',
            'offset_shift', 'remove_note', 'add_note', 'split_note',
            'join_notes']
    im = ax.imshow(confusion_mat, cmap='Oranges', interpolation='nearest')
    # We want to show all ticks...
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(degs)))
    ax.set_yticks(np.arange(len(degs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(degs, fontname='serif')
    ax.set_yticklabels(degs, fontname='serif')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")
    for i in range(len(degs)):
        for j in range(len(degs)):
            text = ax.text(j, i, "%.2f" % confusion_mat[i, j],
                           ha="center", va="center", color="black" if confusion_mat[i,j] < 0.35 else "white",
                           fontname='serif')
    plt.ylim(8.5, -0.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{save_plots}.png',
                    dpi=300)
        plt.savefig(f'{save_plots}.pdf',
                    dpi=300)


def construct_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", default='output',
                        help='location of logs and model checkpoints')
    parser.add_argument("--save_plots", default=None,
                        help='location to save plots. By default will '
                        'save to output_dir. If set to none no plots '
                        'saved')
    parser.add_argument("--in_dir", default='acme',
                        help='location of the pianoroll and command '
                        'corpus datasets')
    parser.add_argument("--task_names", nargs='+',
                        help='names of tasks to get results for. '
                        'must correspond to names of dirs in output_dir')
    parser.add_argument("--setting_names", nargs='+',
                        help='list of lists describing the names of '
                        'variables the gridsearches were performed over '
                        'for each task')
    parser.add_argument("--formats", nargs='+',
                        help='format type for each task')
    parser.add_argument("--seq_len", nargs='+',
                        help='seq_len for each task')
    parser.add_argument("--metrics", nargs='+',
                        help='metric for each task')
    parser.add_argument("--task_desc", nargs='+',
                        help='description for each task to use in result '
                        'table')
    parser.add_argument("--splits", nargs='+', default=['train', 'valid', 'test'],
                        help="which splits to evaluate: train, valid, test.")
    return parser
    
    
def main(args):
    task_names = args.task_names
    # TODO: change the way this is done! ATM can't think of
    # better way of handling sending a list of lists
    setting_names = [eval(ll) for ll in args.setting_names]
    in_dir = args.in_dir
    output_dir = args.output_dir
    save_plots = args.save_plots
    if save_plots is None:
        save_plots = output_dir
    elif save_plots.lower == 'none':
        save_plots = None
    formats = dict(zip(task_names, args.formats))
    seq_len = dict(zip(task_names, args.seq_len))
    metrics = dict(zip(task_names, args.metrics))
    task_desc = dict(zip(task_names, args.task_desc))
    splits = args.splits
    
    results = {}
    mean_res = {}
    median_res = {}
    min_idx = {}
    median_min_idx = {}
    
    
    for task_name, setting_name in zip(task_names, setting_names):
        print(f'{task_name} plots {20*"="}')
        settings = get_settings(output_dir, task_name)
        results[task_name] = plot_task_losses(output_dir, task_name,
                                              settings, setting_name)
        plt.show()
    
    
    for task_name, setting_name in zip(task_names, setting_names):
        res = results[task_name]
        if len(res) == 0:
            print(f'No results for {task_name}')
            continue
        mean_res[task_name] = res.mean(level=setting_name)
        median_res[task_name] = res.median(level=setting_name)
        min_idx[task_name] = res.min_vld_loss.idxmin()
        median_min_idx[task_name] = res.min_vld_loss.idxmin()
        df = results[task_name].reset_index()
        df['expt_id'] = (
            df[setting_name]
                .apply(lambda x: x.astype(str), axis=1)
                .apply('_'.join, axis=1)
        )
        df.sort_values('expt_id', inplace=True)
        # df.pivot(index='repeat', columns='expt_id', values='min_vld_loss').T.plot(linestyle=None)
        # plt.xticks(rotation=90)
        plt.figure()
        sns.pointplot(x='expt_id', y='min_vld_loss', estimator=np.median,
                      ci='sd', data=df, linewidth=0)
        plt.xticks(rotation=90)
        plt.title(f'Summary of {task_name} - median over repeats')
        plt.show()
    #     sns.pointplot(x='expt_id', y='min_vld_loss', estimator=np.mean,
    #                   ci='sd', data=df, linewidth=0)
    #     plt.xticks(rotation=90)
    #     plt.title(f'Summary of {task_name} - mean over repeats')
    #     plt.show()
    
    
    best_models = {task: f'{output_dir}/{task}/{val[1]}.checkpoint.best'
               for task, val in min_idx.items()}
    best_logs = {task: f'{output_dir}/{task}/{val[1]}.log'
                 for task, val in min_idx.items()}
    print(f"best models: {best_models}")
    for task_name, log_file in best_logs.items():
        plot_log_file(log_file)
        plt.title(f'{task_name} best model training curve')
        if save_plots:
            plt.savefig(f'{save_plots}/{task_name}__best_model_loss.png',
                        dpi=300)
            plt.savefig(f'{save_plots}/{task_name}__best_model_loss.pdf',
                        dpi=300)
        plt.show()
    
    
    task_eval_log = {}
    for task_name in task_names:
        eval_parser = eval_construct_parser()
        eval_args_str = (
            f"--input {in_dir} "
            f"--model {best_models[task_name]} "
            f"--format {formats[task_name]} "
            f"--task {task_name[4]} "
            f"--seq_len {seq_len[task_name]} "
            f"--splits {' '.join(splits)}"
    #         f"--splits test"
        )
        eval_args = eval_parser.parse_args(eval_args_str.split())
        log_info = eval_main(eval_args)
        task_eval_log[task_name] = log_info
    
    dict_of_df = {k: pd.DataFrame(v).T for k,v in task_eval_log.items()}
    res_df = pd.concat(dict_of_df).drop(['batch', 'epoch', 'mode'], axis=1)
    res_df['avg_acc'] = res_df['avg_acc']/100  # I think this is better for tables
    print(res_df)
    
    confusion = {}
    summary_tab = pd.DataFrame(columns=['Task', 'Model', 'Loss', 'Metric'])
    for ii, task_name in enumerate(task_names):
        print(task_name)
        df = (
            res_df.loc[task_name]
                .dropna(axis=1)  # removes cols with na in them (not a metric for this task)
        )
        if 'confusion_mat' in df.columns:
            confusion[task_name] = df['confusion_mat']
            for split in ['train', 'valid', 'test']:
                confusion_mat = df.loc[split, 'confusion_mat']
                if save_plots:
                    save_plot_loc = f"{save_plots}/{task_name}"
                    if not os.path.exists(save_plot_loc):
                        os.makedirs(save_plot_loc)
                plt.figure(figsize=(5, 5))
                plot_confusion(confusion_mat, save_plots=f"{save_plot_loc}/{split}_confusion")
                plt.show()
            df.drop('confusion_mat', axis=1, inplace=True)
        df = (
            df
                .apply(pd.to_numeric)  # they are strings, convert to int or float
                .applymap(round_to_n)  # round to 3sf
        )
        print(df)
        df.to_latex(f'{output_dir}/{task_name}_table.tex')
        summary_tab.loc[ii] = [task_desc[task_name], f'{task_name} baseline', 
                               df.loc['test', 'avg_loss'],
                               df.loc['test', metrics[task_name]]]
    
    # Hard coded dumb-baseline results
    rule_based = {
        'task1': ['ErrorDetection', 'Rule-based', 0.466, 0.00],
        'task2': ['ErrorClassification', 'Rule-based', 2.197, 0.113],
        'task3': ['ErrorLocation', 'Rule-based', 0.404, 0.00],
        'task4': ['ErrorCorrection', 'Rule-based', 0.690, 0.590],
    }
    rule_based_df = pd.DataFrame.from_dict(
        rule_based, orient='index', 
        columns=['Task', 'Model', 'Loss', 'Metric']
    )
    
    task_cat = ['ErrorDetection', 'ErrorClassification',
                'ErrorLocation', 'ErrorCorrection']
    summary_tab_ = pd.concat((summary_tab, rule_based_df))
    summary_tab_['Task'] = pd.Categorical(summary_tab_['Task'], categories=task_cat,
                                          ordered=True)
    summary_tab_ = summary_tab_.set_index(['Task', 'Model']).sort_index()
    summary_tab_.round(3).to_latex(f'{output_dir}/summary_table.tex')
    print(summary_tab_.round(3))
    # summary_tab_.applymap(round_to_n).to_latex(f'{output_dir}/summary_table.tex')
    # summary_tab_.applymap(round_to_n)
    
    return results, min_idx, task_eval_log, res_df, summary_tab_, confusion



if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)