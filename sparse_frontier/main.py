import hydra
from omegaconf import DictConfig


def setting_up(cfg):
    import os
    import random
    import numpy as np

    from sparse_frontier.utils import GlobalSettings
    from sparse_frontier.utils.data import get_data_path, get_pred_path, get_results_path

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    # In case of torch seed it's set in vLLM Model class

    if cfg.overwrite:
        assert cfg.debug, "Overwrite is only allowed in debug mode"

    GlobalSettings.set('cfg', cfg)

    os.makedirs(os.path.dirname(get_data_path()), exist_ok=True)
    os.makedirs(os.path.dirname(get_pred_path()), exist_ok=True)
    os.makedirs(os.path.dirname(get_results_path()), exist_ok=True)


def run(cfg):
    setting_up(cfg)

    from sparse_frontier.utils.checks import prepration_needed, prediction_needed, evaluation_needed

    if cfg.mode in ["prep", "all"]:
        if prepration_needed():
            from sparse_frontier.preparation import prepare_task
            prepare_task()
    
    if cfg.mode in ["pred", "all"]:
        if prediction_needed():
            from sparse_frontier.prediction import predict_task
            predict_task()
    
    if cfg.mode in ["eval", "all"]:
        if evaluation_needed():
            from sparse_frontier.evaluation import evaluate_task
            evaluate_task()
    
    if cfg.mode not in ["prep", "pred", "eval", "all"]:
        raise ValueError(f'Invalid mode: {cfg.mode}')


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
