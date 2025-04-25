import os

from sparse_frontier.utils.globals import GlobalSettings
from sparse_frontier.utils.data import (
    read_jsonl,
    get_data_path,
    get_pred_path,
    get_results_path,
)


def prepration_needed():
    cfg = GlobalSettings.get("cfg")
    data_path = get_data_path()

    if not os.path.exists(data_path):
        return True

    if cfg.overwrite:
        os.remove(data_path)
        return True

    data = read_jsonl(data_path)
    return (len(data) < cfg.samples)


def prediction_needed():
    cfg = GlobalSettings.get("cfg")
    pred_path = get_pred_path()

    if not os.path.exists(pred_path):
        return True

    if cfg.overwrite:
        os.remove(pred_path)
        return True

    data = read_jsonl(pred_path)
    return (len(data) < cfg.samples)


def evaluation_needed():
    cfg = GlobalSettings.get("cfg")
    results_path = get_results_path()

    if not os.path.exists(results_path):
        return True

    if cfg.overwrite:
        os.remove(results_path)
        return True

    return False
