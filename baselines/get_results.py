#!/usr/bin/env python
import argparse
import logging
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from baselines.eval_task import construct_parser as eval_construct_parser
from baselines.eval_task import main as eval_main


def plot_log_file(log_file, trn_kwargs=None, vld_kwargs=None):
    df = pd.read_csv(log_file)
    trn_df = df.loc[df["mode"] == "train"]
    min_trn_loss = trn_df["avg_loss"].min()
    min_trn_acc = trn_df["avg_acc"].max()
    vld_df = df.loc[df["mode"] == "test"]
    min_vld_loss = vld_df["avg_loss"].min()
    min_vld_acc = vld_df["avg_acc"].max()
    if trn_kwargs is None:
        trn_kwargs = dict(label=f"({min_trn_loss:.4f}, {min_trn_acc:.1f})")
    elif "label" in trn_kwargs:
        trn_kwargs["label"] += f" ({min_trn_loss:.4f}, {min_trn_acc:.1f})"
    if vld_kwargs is None:
        vld_kwargs = dict(label=f"({min_vld_loss:.4f}, {min_vld_acc:.1f})")
    elif "label" in vld_kwargs:
        vld_kwargs["label"] += f" ({min_vld_loss:.4f}, {min_vld_acc:.1f})"
    plt.plot(trn_df["epoch"], trn_df["avg_loss"], **trn_kwargs)
    plt.plot(vld_df["epoch"], vld_df["avg_loss"], **vld_kwargs)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    return trn_df, vld_df, (min_trn_loss, min_trn_acc, min_vld_loss, min_vld_acc)


def plot_task_losses(
    output_dir, task_name, settings, setting_names, save_plots=False, show_plots=True
):
    idx_cols = ["task_name", "expt_name"] + setting_names + ["repeat"]
    val_cols = ["min_trn_loss", "min_vld_loss", "min_trn_acc", "min_vld_acc"]
    res = pd.DataFrame(columns=idx_cols + val_cols, dtype=float)
    res.set_index(idx_cols, drop=True, inplace=True)

    summary_fig = plt.figure()
    plt.title(f"{task_name} vld loss curves")

    alphas = [0.3, 0.5, 0.7]
    for setting in settings:
        setting_fig = plt.figure()
        # TODO: this shouldn't be hardcoded 3
        for repeat in range(3):
            expt_name = f'{task_name}__{"_".join(setting)}_{repeat}'
            log_file = f"{output_dir}/{task_name}/{expt_name}.log"

            plt.figure(setting_fig.number)
            try:
                trn_df, vld_df, losses = plot_log_file(
                    log_file,
                    trn_kwargs=dict(
                        label=f"train {repeat}", color="C0", alpha=alphas[repeat]
                    ),
                    vld_kwargs=dict(
                        label=f"valid {repeat}", color="C1", alpha=alphas[repeat]
                    ),
                )
            except pd.errors.EmptyDataError:
                print(f"{expt_name}.log found but empty")
                continue
            except FileNotFoundError:
                print(f"{expt_name}.log not found")
                continue
            min_trn_loss, min_trn_acc, min_vld_loss, min_vld_acc = losses
            idx = (task_name, expt_name) + setting + (str(repeat),)
            res.loc[idx, :] = [min_trn_loss, min_vld_loss, min_trn_acc, min_vld_acc]

            plt.figure(summary_fig.number)
            plt.plot(vld_df["epoch"], vld_df["avg_loss"], color="C1", alpha=0.2)

        plt.figure(setting_fig.number)
        if save_plots:
            plt.savefig(f'{save_plots}/{task_name}__{"_".join(setting)}.png', dpi=300)
            plt.savefig(f'{save_plots}/{task_name}__{"_".join(setting)}.pdf', dpi=300)
        plt.title(f'{task_name}__{"_".join(setting)}')
        if show_plots:
            plt.show()
        plt.close()

    plt.figure(summary_fig.number)
    if save_plots:
        plt.savefig(f"{save_plots}/{task_name}__all_loss_summary.png", dpi=300)
        plt.savefig(f"{save_plots}/{task_name}__all_loss_summary.pdf", dpi=300)
    if show_plots:
        plt.show()
    plt.close()
    return res


def get_settings(output_dir, task_name):
    files = glob(f"{output_dir}/{task_name}/*")
    settings = [tuple(ff.split("__")[-1].rsplit("_", 1)[0].split("_")) for ff in files]
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
    degs = [
        "none",
        "pitch_shift",
        "time_shift",
        "onset_shift",
        "offset_shift",
        "remove_note",
        "add_note",
        "split_note",
        "join_notes",
    ]
    _ = ax.imshow(confusion_mat, cmap="Oranges", interpolation="nearest")
    # We want to show all ticks...
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(degs)))
    ax.set_yticks(np.arange(len(degs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(degs, fontname="serif")
    ax.set_yticklabels(degs, fontname="serif")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    for i in range(len(degs)):
        array_range = np.max(confusion_mat) - np.min(confusion_mat)
        color_cutoff = np.min(confusion_mat) + array_range / 2
        for j in range(len(degs)):
            _ = ax.text(
                j,
                i,
                "%.2f" % confusion_mat[i, j],
                ha="center",
                va="center",
                color="black" if confusion_mat[i, j] < color_cutoff else "white",
                fontname="serif",
            )
    plt.ylim(8.5, -0.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{save_plots}.png", dpi=300)
        plt.savefig(f"{save_plots}.pdf", dpi=300)


def plot_1d_array_per_deg(array, save_plots=False, ax=None):
    if ax is None:
        ax = plt.gca()
    degs = [
        "none",
        "pitch_shift",
        "time_shift",
        "onset_shift",
        "offset_shift",
        "remove_note",
        "add_note",
        "split_note",
        "join_notes",
    ]
    _ = ax.imshow(array.reshape((-1, 1)), cmap="Oranges", interpolation="nearest",)

    # We want to show all ticks...
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(degs)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(degs, fontname="serif")
    for ii in range(len(degs)):
        array_range = np.max(array) - np.min(array)
        color_cutoff = np.min(array) + array_range / 2
        if array[ii] < color_cutoff:
            color = "black"
        else:
            color = "white"
        _ = ax.text(
            0,
            ii,
            f"{array[ii]:.2f}",
            ha="center",
            va="center",
            color=color,
            fontname="serif",
        )
    plt.ylim(8.5, -0.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{save_plots}.png", dpi=300)
        plt.savefig(f"{save_plots}.pdf", dpi=300)


def construct_parser():
    parser = argparse.ArgumentParser(
        description="Script for summarising "
        "results of experiments run. Expects a "
        "directory of output data, with "
        "subdirectories for the name of the task."
        " These subdirs contain the logs and "
        "checkpoints for models fitted."
    )

    parser.add_argument(
        "--output_dir", default="output", help="location of logs and model checkpoints"
    )
    parser.add_argument(
        "--save_plots",
        default=None,
        help="location to save plots. By default will "
        "save to output_dir. If set to none no plots "
        "saved",
    )
    parser.add_argument(
        "--in_dir",
        default="acme",
        help="location of the pianoroll and command corpus datasets",
    )
    parser.add_argument(
        "--task_names",
        nargs="+",
        required=True,
        help="names of tasks to get results for. "
        "must correspond to names of dirs in output_dir",
    )
    parser.add_argument(
        "--setting_names",
        nargs="+",
        required=True,
        help="A list (with no spaces) describing the names of "
        "variables the gridsearches were performed over "
        "for each task e.g. for --task_names task1 task4 "
        "--setting_names \"['lr','wd','hid']\" "
        "\"['lr','wd','hid','lay']\". You need to be careful "
        "to preserve quotes",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        required=True,
        choices=["pianoroll", "command"],
        help="data format type for each task e.g. "
        "for --task_names task1 task4 --formats command "
        "pianoroll",
    )
    parser.add_argument(
        "--seq_len", nargs="+", required=True, help="seq_len for each task"
    )
    parser.add_argument(
        "--metrics", nargs="+", required=True, help="metric for each task"
    )
    parser.add_argument(
        "--task_desc",
        nargs="+",
        required=True,
        help="description (with no spaces) for each task to "
        "use as the identifier in the results table e.g. for "
        "--task_names task1 task4 --task_desc ErrorDetection "
        "ErrorCorrection",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        default=["train", "valid", "test"],
        help="which splits to evaluate: train, valid, test.",
    )
    parser.add_argument(
        "-s",
        "--show_plots",
        action="store_true",
        help="Whether to use plt.show() or not",
    )
    return parser


def main(args):
    task_names = args.task_names
    nr_tasks = len(task_names)
    # TODO: change the way this is done! ATM can't think of
    # better way of handling sending a list of lists
    setting_names = [eval(ll) for ll in args.setting_names]
    assert len(setting_names) == nr_tasks, (
        "You must submit a list of "
        "parameters being searched for each task --setting_names . "
        "Submit lists of parameter names with no spaces, e.g. "
        "--task_names task1 task4 "
        "--setting_names ['lr','wd','hid'] ['lr','wd','hid','lay'] ."
        f"You submitted {nr_tasks} tasks: {task_names}, but "
        f"{len(setting_names)} setting names: {setting_names}"
    )
    for varname in ["formats", "seq_len", "metrics", "task_desc"]:
        value = getattr(args, varname)
        vlen = len(value)
        assert vlen == nr_tasks, (
            f"You submitted {vlen} {varname}, but need "
            f"to supply {nr_tasks}. task_names: {task_names}, "
            f"{varname}: {value}"
        )

    in_dir = args.in_dir
    output_dir = args.output_dir
    save_plots = args.save_plots
    if save_plots is None:
        save_plots = output_dir
    elif save_plots.lower == "none":
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
        results[task_name] = plot_task_losses(
            output_dir,
            task_name,
            settings,
            setting_name,
            save_plots=f"{save_plots}",
            show_plots=args.show_plots,
        )

    for task_name, setting_name in zip(task_names, setting_names):
        res = results[task_name]
        if len(res) == 0:
            print(f"No results for {task_name}")
            continue
        mean_res[task_name] = res.mean(level=setting_name)
        median_res[task_name] = res.median(level=setting_name)
        min_idx[task_name] = res.min_vld_loss.idxmin()
        median_min_idx[task_name] = res.min_vld_loss.idxmin()
        df = results[task_name].reset_index()
        df["expt_id"] = (
            df[setting_name]
            .apply(lambda x: x.astype(str), axis=1)
            .apply("_".join, axis=1)
        )
        df.sort_values("expt_id", inplace=True)
        plt.figure()
        sns.pointplot(
            x="expt_id",
            y="min_vld_loss",
            estimator=np.median,
            ci="sd",
            data=df,
            linewidth=0,
        )
        plt.xticks(rotation=90)
        plt.title(f"Summary of {task_name} - median over repeats")
        if save_plots:
            plt.savefig(f"{save_plots}/{task_name}__min_loss_summary.pdf", dpi=300)
            plt.savefig(f"{save_plots}/{task_name}__min_loss_summary.png", dpi=300)
        if args.show_plots:
            plt.show()
        plt.close("all")

    #     sns.pointplot(x='expt_id', y='min_vld_loss', estimator=np.mean,
    #                   ci='sd', data=df, linewidth=0)
    #     plt.xticks(rotation=90)
    #     plt.title(f'Summary of {task_name} - mean over repeats')
    #     plt.show()

    best_models = {
        task: f"{output_dir}/{task}/{val[1]}.checkpoint.best"
        for task, val in min_idx.items()
    }
    best_logs = {
        task: f"{output_dir}/{task}/{val[1]}.log" for task, val in min_idx.items()
    }
    print(f"best models: {best_models}")
    for task_name, log_file in best_logs.items():
        plot_log_file(log_file)
        plt.title(f"{task_name} best model training curve")
        if save_plots:
            plt.savefig(f"{save_plots}/{task_name}__best_model_loss.png", dpi=300)
            plt.savefig(f"{save_plots}/{task_name}__best_model_loss.pdf", dpi=300)
        if args.show_plots:
            plt.show()
        plt.close("all")

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
        logging.disable(logging.WARNING)
        log_info = eval_main(eval_args)
        logging.disable(logging.NOTSET)
        task_eval_log[task_name] = log_info

    dict_of_df = {k: pd.DataFrame(v).T for k, v in task_eval_log.items()}
    res_df = pd.concat(dict_of_df).drop(["batch", "epoch", "mode"], axis=1)
    res_df["avg_acc"] = res_df["avg_acc"] / 100  # I think this is better for tables
    print(res_df)

    confusion = {}
    summary_tab = pd.DataFrame(columns=["Task", "Model", "Loss", "Metric"])
    for ii, task_name in enumerate(task_names):
        print(task_name)
        df = res_df.loc[task_name].dropna(
            axis=1
        )  # removes cols with na in them (not a metric for this task)
        if "confusion_mat" in df.columns:
            confusion[task_name] = df["confusion_mat"]
            for split in splits:
                confusion_mat = df.loc[split, "confusion_mat"]
                plt.figure(figsize=(5, 5))
                plot_confusion(
                    confusion_mat,
                    save_plots=f"{save_plots}/{task_name}__{split}_confusion",
                )
                if args.show_plots:
                    plt.show()
                plt.close("all")
            df.drop("confusion_mat", axis=1, inplace=True)
        for colname in ["p_per_deg", "r_per_deg", "f_per_deg", "avg_acc_per_deg"]:
            if colname in df.columns:
                array = df[colname]
                for split in splits:
                    array = df.loc[split, colname]
                    plt.figure(figsize=(5, 5))
                    plot_1d_array_per_deg(
                        array, save_plots=f"{save_plots}/{task_name}__{split}_{colname}"
                    )
                    plt.xticks([0], [colname], rotation=45, fontname="serif", ha="left")
                    if args.show_plots:
                        plt.show()
                    plt.close("all")
                df.drop(colname, axis=1, inplace=True)
        df = df.apply(
            pd.to_numeric
        ).applymap(  # they are strings, convert to int or float
            round_to_n
        )  # round to 3sf
        print(df)
        df.to_latex(f"{output_dir}/{task_name}_table.tex")
        summary_tab.loc[ii] = [
            task_desc[task_name],
            f"{task_name} baseline",
            df.loc["test", "avg_loss"],
            df.loc["test", metrics[task_name]],
        ]

    # Hard coded dumb-baseline results
    rule_based = {
        "task1": ["ErrorDetection", "Rule-based", 0.466, 0.00],
        "task2": ["ErrorClassification", "Rule-based", 2.197, 0.113],
        "task3": ["ErrorLocation", "Rule-based", 0.404, 0.00],
        "task4": ["ErrorCorrection", "Rule-based", 0.690, 0.590],
    }
    rule_based_df = pd.DataFrame.from_dict(
        rule_based, orient="index", columns=["Task", "Model", "Loss", "Metric"]
    )

    task_cat = [
        "ErrorDetection",
        "ErrorClassification",
        "ErrorLocation",
        "ErrorCorrection",
    ]
    summary_tab_ = pd.concat((summary_tab, rule_based_df))
    summary_tab_["Task"] = pd.Categorical(
        summary_tab_["Task"], categories=task_cat, ordered=True
    )
    summary_tab_ = summary_tab_.set_index(["Task", "Model"]).sort_index()
    summary_tab_.round(3).to_latex(f"{output_dir}/summary_table.tex")
    print(summary_tab_.round(3))
    # summary_tab_.applymap(round_to_n).to_latex(f'{output_dir}/summary_table.tex')
    # summary_tab_.applymap(round_to_n)

    return results, min_idx, task_eval_log, res_df, summary_tab_, confusion


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
