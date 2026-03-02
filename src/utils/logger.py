import os
import wandb
import torch
from omegaconf import DictConfig, OmegaConf


_run = None


def init_run(cfg: DictConfig, name: str):
    global _run
    _run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    return _run


def log_metrics(metrics: dict, step: int = None):
    if _run is not None:
        wandb.log(metrics, step=step)


def log_model(model: torch.nn.Module, name: str, cfg: DictConfig):
    path = os.path.join(cfg.checkpoints.dir, f"{name}.pt")
    os.makedirs(cfg.checkpoints.dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    if _run is not None:
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        _run.log_artifact(artifact)
    return path


def save_checkpoint(state: dict, name: str, cfg: DictConfig):
    os.makedirs(cfg.checkpoints.dir, exist_ok=True)
    path = os.path.join(cfg.checkpoints.dir, f"{name}.pt")
    torch.save(state, path)
    return path


def load_checkpoint(name: str, cfg: DictConfig) -> dict:
    path = os.path.join(cfg.checkpoints.dir, f"{name}.pt")
    return torch.load(path, map_location="cpu")


def finish_run():
    if _run is not None:
        wandb.finish()
