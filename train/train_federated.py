import os
import copy
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion
from src.federated.fedprox_server import FedProxServer
from src.federated.fedper_client import FedPerClient
from src.federated.clustering import SubjectClusterer
from src.utils.seed import set_seed
from src.utils.logger import init_run, log_metrics, log_model, finish_run

class SubjectDataset(Dataset):

    def __init__(self, processed_dir: str, subject_id: int):
        pattern = os.path.join(processed_dir, f'subject_{subject_id}', 'windows', 'window_*.pt')
        self.paths = sorted(glob.glob(pattern))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx], map_location='cpu')

class FederatedModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(nn.Linear(48, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 64))
        self.personal_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 6))

    def forward(self, x: torch.Tensor):
        h = self.shared_encoder(x)
        out = self.personal_head(h)
        return out

def get_available_subjects(processed_dir: str) -> list:
    sids = []
    for d in sorted(os.listdir(processed_dir)):
        if d.startswith('subject_'):
            sid = int(d.replace('subject_', ''))
            windows_dir = os.path.join(processed_dir, d, 'windows')
            if os.path.isdir(windows_dir) and len(os.listdir(windows_dir)) > 0:
                sids.append(sid)
    return sids

def simulate_federated(cfg: DictConfig, device: torch.device):
    data_cfg = OmegaConf.load('configs/data.yaml')
    processed_dir = data_cfg.paths.processed
    available_sids = get_available_subjects(processed_dir)
    n_clients = min(len(available_sids), cfg.training.federated.n_clients)
    selected_sids = available_sids[:n_clients]
    if not selected_sids:
        print('No subjects found. Using synthetic simulation.')
        selected_sids = list(range(1, 11))
        n_clients = len(selected_sids)
    print(f'Simulating federated learning with {n_clients} clients')
    global_model = FederatedModel().to(device)
    server = FedProxServer(global_model, mu=cfg.training.federated.fedprox_mu)
    personal_layers = ['personal_head']
    clients = []
    for sid in selected_sids:
        client_model = copy.deepcopy(global_model)
        client = FedPerClient(client_model, personal_layers, client_id=sid, lr=cfg.training.federated.client_lr, local_epochs=cfg.training.federated.local_epochs)
        clients.append(client)
    clusterer = SubjectClusterer(n_clusters=min(5, n_clients))
    profiles = {}
    for sid in selected_sids:
        ds = SubjectDataset(processed_dir, sid)
        if len(ds) > 0:
            feats = [torch.load(p, map_location='cpu')['features'].numpy() for p in ds.paths[:50]]
            profiles[sid] = clusterer.build_subject_profile(feats)
        else:
            profiles[sid] = np.random.randn(48).astype(np.float32)
    if len(profiles) >= 5:
        clusterer.fit(profiles)
        print(f'Cluster stats: {clusterer.get_stats()}')
    loss_fn = nn.MSELoss()
    n_rounds = cfg.training.federated.n_rounds
    for rnd in range(n_rounds):
        global_state = server.distribute()
        client_states = []
        client_weights = []
        clients_this_round = np.random.choice(len(clients), size=max(1, int(0.3 * len(clients))), replace=False)
        for ci in clients_this_round:
            client = clients[ci]
            sid = selected_sids[ci]
            client.receive_global(global_state)
            ds = SubjectDataset(processed_dir, sid)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=_collate_fn)
            shared_state = client.train_local(loader, loss_fn, device, proximal_fn=server.get_proximal_term)
            client_states.append(shared_state)
            client_weights.append(max(1, len(ds)))
        if client_states:
            server.aggregate(client_states, client_weights)
        if rnd % 5 == 0:
            log_metrics({'fed/round': rnd, 'fed/clients_sampled': len(clients_this_round)}, step=rnd)
            print(f'  Round {rnd:3d}/{n_rounds}  clients={len(clients_this_round)}')
    log_model(global_model, 'federated_global', cfg)
    return (server, clients, clusterer)

def _collate_fn(batch):
    features = torch.stack([b['features'] for b in batch])
    hrv = torch.stack([b['hrv'] for b in batch])
    target = torch.cat([features[:, 20:21], hrv], dim=-1)
    return {'features': features, 'hrv': target}

@hydra.main(config_path='../configs', config_name='training', version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    init_run(cfg, name='federated-training')
    simulate_federated(cfg, device)
    print('Federated training simulation complete.')
    finish_run()
if __name__ == '__main__':
    main()