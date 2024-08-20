import json
import os
from glob import glob

import pandas as pd


SUITES = {
    "Adroit": {"pen", "hammer", "door", "relocate"},
    "MuJoCo": {"halfcheetah", "walker2d", "hopper"},
    "AntMaze": {"antmaze"},
    "Kitchen": {"kitchen"},
}


def get_suite_name(task_name):
    task_name = task_name.split("-")[0]
    for suite_name, suite_set in SUITES.items():
        if task_name in suite_set:
            return suite_name
        
    assert False, f"Unknown task {task_name}"


def get_dataset_name(task_name):
    return "-".join(task_name.split("-")[1:-1])


def get_simp_task_name(task_name):
    return task_name.split("-")[0]


def show_results():
    # Load metrics
    report = ""
    data = {
        "return": [],
        "task": [],
        "settings": []
    }

    metrics_files = glob("metrics/*.jsonl")
    for filename in metrics_files:
        with open(filename, "rt") as f:
            metrics = list(map(json.loads, f.readlines()))

        for item in metrics:
            data["return"].append(item["metrics"]["return"])
            data["task"].append(item["task"])
            data["settings"].append(item["conf"]["log_group"])

    # Process
    data = pd.DataFrame(data)
    data["suite"] = data["task"].apply(get_suite_name)

    # Aggragate
    data = data.pivot_table(index=["suite", "task"], 
                            columns="settings", 
                            values="return", 
                            aggfunc=["mean", "std", "size"])

    for suite_name, suite_metrics in data.groupby('suite'):
        # average value
        suite_metrics = suite_metrics.reset_index(level=0, drop=True)
        suite_metrics.loc['Average', :] = suite_metrics.mean()
        suite_metrics = suite_metrics.dropna(axis=1, how='all')

        # format
        m_mean = suite_metrics["mean"]
        m_std  = suite_metrics["std"]
        m_size = suite_metrics["size"]

        formatted = pd.DataFrame(index=suite_metrics.index)
        show_std = set(m_std.columns)
        for col in m_mean.columns:
            if col in show_std:
                formatted[col] = m_mean[col].map('{:.1f}'.format) + " ± " + m_std[col].map('{:.1f}'.format) + " (" + m_size[col].map('{:.0f}'.format) + ")"
            else:
                formatted[col] = m_mean[col].map('{:.1f}'.format)

        # sort
        formatted = formatted.reset_index()
        formatted.insert(0, "dataset", formatted["task"].apply(get_dataset_name))
        formatted["task"] = formatted["task"].apply(get_simp_task_name)

        formatted = formatted.sort_values(["dataset", "task"])

        report += f"<h2>{suite_name}</h2>\n{formatted.to_html(index=False)}\n"

        # save csv report
        os.makedirs("reports/", exist_ok=True)
        formatted.to_csv(f"reports/{suite_name}.csv", index=False)

    # Write report
    with open("report.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    show_results()
