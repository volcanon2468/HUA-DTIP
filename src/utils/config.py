import os
from omegaconf import DictConfig, OmegaConf


def load_configs():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_cfg = OmegaConf.load(os.path.join(base, "configs", "data.yaml"))
    model_cfg = OmegaConf.load(os.path.join(base, "configs", "model.yaml"))
    training_cfg = OmegaConf.load(os.path.join(base, "configs", "training.yaml"))
    return OmegaConf.merge(data_cfg, model_cfg, training_cfg)


def get_device(cfg: DictConfig):
    import torch
    device_str = cfg.get("device", "cuda")
    return torch.device(device_str if torch.cuda.is_available() else "cpu")


def get_processed_dir(cfg: DictConfig) -> str:
    return cfg.paths.processed


def get_checkpoint_dir(cfg: DictConfig) -> str:
    return cfg.checkpoints.dir


def get_results_dir(cfg: DictConfig) -> str:
    d = cfg.checkpoints.results_dir
    os.makedirs(d, exist_ok=True)
    return d
