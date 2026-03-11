import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List


class FedPerClient:
    def __init__(self, model: nn.Module, personal_layer_names: List[str],
                 client_id: int, lr: float = 1e-3, local_epochs: int = 5):
        self.model = model
        self.personal_layer_names = personal_layer_names
        self.client_id = client_id
        self.lr = lr
        self.local_epochs = local_epochs
        self.personal_state = {}
        self._save_personal_layers()

    def _save_personal_layers(self):
        state = self.model.state_dict()
        self.personal_state = {
            k: v.clone() for k, v in state.items()
            if any(pl in k for pl in self.personal_layer_names)
        }

    def _load_personal_layers(self):
        state = self.model.state_dict()
        for k, v in self.personal_state.items():
            if k in state:
                state[k] = v.clone()
        self.model.load_state_dict(state)

    def receive_global(self, global_state: OrderedDict):
        filtered = {}
        for k, v in global_state.items():
            if not any(pl in k for pl in self.personal_layer_names):
                filtered[k] = v
        current = self.model.state_dict()
        current.update(filtered)
        self.model.load_state_dict(current)
        self._load_personal_layers()

    def train_local(self, dataloader, loss_fn, device, proximal_fn=None):
        self.model.to(device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.local_epochs):
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = torch.nan_to_num(batch["features"].to(device), nan=0.0)
                    y = torch.nan_to_num(batch.get("hrv", batch.get("label", torch.zeros(x.shape[0]))).to(device), nan=0.0)
                elif isinstance(batch, (list, tuple)):
                    x, y = torch.nan_to_num(batch[0].to(device), nan=0.0), torch.nan_to_num(batch[1].to(device), nan=0.0)
                else:
                    x = torch.nan_to_num(batch.to(device), nan=0.0)
                    y = torch.zeros(x.shape[0], device=device)

                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = loss_fn(output, y)

                if proximal_fn is not None:
                    loss = loss + proximal_fn(self.model)

                opt.zero_grad(); loss.backward(); opt.step()

        self._save_personal_layers()
        return self.get_shared_state()

    def get_shared_state(self) -> OrderedDict:
        state = self.model.state_dict()
        shared = OrderedDict()
        for k, v in state.items():
            if not any(pl in k for pl in self.personal_layer_names):
                shared[k] = v.clone()
        return shared

    def get_full_state(self) -> OrderedDict:
        return copy.deepcopy(self.model.state_dict())

    def personalize(self, dataloader, loss_fn, device, n_epochs: int = 10):
        self.model.to(device).train()
        personal_params = []
        for name, p in self.model.named_parameters():
            if any(pl in name for pl in self.personal_layer_names):
                personal_params.append(p)
            else:
                p.requires_grad = False

        opt = torch.optim.Adam(personal_params, lr=self.lr * 0.5)

        for epoch in range(n_epochs):
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = torch.nan_to_num(batch["features"].to(device), nan=0.0)
                    y = torch.nan_to_num(batch.get("hrv", torch.zeros(x.shape[0])).to(device), nan=0.0)
                else:
                    x, y = torch.nan_to_num(batch[0].to(device), nan=0.0), torch.nan_to_num(batch[1].to(device), nan=0.0)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = loss_fn(output, y)
                opt.zero_grad(); loss.backward(); opt.step()

        for _, p in self.model.named_parameters():
            p.requires_grad = True
        self._save_personal_layers()
