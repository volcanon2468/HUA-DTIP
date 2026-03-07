import copy
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List, Dict


class FedProxServer:
    def __init__(self, global_model: nn.Module, mu: float = 0.01):
        self.global_model = global_model
        self.mu = mu
        self.global_state = copy.deepcopy(global_model.state_dict())
        self.round_num = 0

    def distribute(self) -> OrderedDict:
        return copy.deepcopy(self.global_state)

    def aggregate(self, client_states: List[OrderedDict], client_weights: List[float] = None):
        n = len(client_states)
        if client_weights is None:
            client_weights = [1.0 / n] * n

        w_sum = sum(client_weights)
        client_weights = [w / w_sum for w in client_weights]

        new_state = OrderedDict()
        for key in self.global_state.keys():
            new_state[key] = sum(
                client_weights[i] * client_states[i][key].float()
                for i in range(n)
            )

        self.global_state = new_state
        self.global_model.load_state_dict(new_state)
        self.round_num += 1
        return new_state

    def get_proximal_term(self, local_model: nn.Module) -> torch.Tensor:
        prox_loss = torch.tensor(0.0, device=next(local_model.parameters()).device)
        for (name, local_p), (_, global_p) in zip(
            local_model.named_parameters(), self.global_model.named_parameters()
        ):
            if local_p.requires_grad:
                prox_loss += ((local_p - global_p.to(local_p.device).detach()) ** 2).sum()
        return (self.mu / 2.0) * prox_loss

    def get_stats(self) -> dict:
        total_params = sum(p.numel() for p in self.global_model.parameters())
        return {"round": self.round_num, "total_params": total_params, "mu": self.mu}
