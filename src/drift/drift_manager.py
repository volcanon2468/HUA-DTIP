import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.drift.adwin import ADWINDetector
from src.drift.mmd import MMDDetector
from src.drift.autoencoder_novelty import AutoencoderNoveltyDetector

class DriftDiagnoser:
    FEATURE_GROUPS = {'imu_gait': list(range(0, 8)), 'imu_stats': list(range(8, 20)), 'cardio_hr': list(range(20, 30)), 'cardio_hrv': list(range(30, 40)), 'quality': list(range(40, 48))}

    def diagnose(self, adwin_result: dict, mmd_result: dict, ae_result: dict) -> dict:
        causes = []
        severity = 0.0
        if adwin_result.get('drift_detected'):
            drifted = adwin_result.get('drifted_features', [])
            affected_groups = set()
            for feat_idx in drifted:
                for group_name, indices in self.FEATURE_GROUPS.items():
                    if feat_idx in indices:
                        affected_groups.add(group_name)
            causes.append({'type': 'concept_drift', 'features': drifted, 'groups': list(affected_groups)})
            severity += 0.3 * len(drifted) / 48.0
        if mmd_result.get('drift_detected'):
            causes.append({'type': 'distribution_shift', 'mmd': mmd_result.get('mmd_value', 0.0)})
            severity += 0.4
        if ae_result.get('drift_detected'):
            causes.append({'type': 'novelty', 'error': ae_result.get('reconstruction_error', 0.0), 'ratio': ae_result.get('novelty_ratio', 0.0)})
            severity += 0.3 * min(1.0, ae_result.get('novelty_ratio', 0.0))
        if severity > 0.5:
            action = 'full_retrain'
        elif severity > 0.2:
            action = 'ewc_update'
        elif severity > 0.05:
            action = 'adapt_reference'
        else:
            action = 'no_action'
        return {'causes': causes, 'severity': float(min(1.0, severity)), 'recommended_action': action, 'n_detectors_triggered': sum([adwin_result.get('drift_detected', False), mmd_result.get('drift_detected', False), ae_result.get('drift_detected', False)])}

class EWCRegularizer:

    def __init__(self, model: nn.Module, lambda_ewc: float=1000.0):
        self.lambda_ewc = lambda_ewc
        self._params = {}
        self._fisher = {}
        self._is_fitted = False

    def compute_fisher(self, model: nn.Module, data_loader, device, n_samples: int=200):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        count = 0
        for batch in data_loader:
            if count >= n_samples:
                break
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            model.zero_grad()
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            loss = output.pow(2).mean()
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += x.shape[0]
        for n in fisher:
            fisher[n] /= max(count, 1)
        self._fisher = fisher
        self._params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self._is_fitted = True

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._is_fitted:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self._fisher and p.requires_grad:
                loss += (self._fisher[n].to(p.device) * (p - self._params[n].to(p.device)).pow(2)).sum()
        return self.lambda_ewc * loss

class DriftManager:

    def __init__(self, n_features: int=48, device: str='cpu'):
        self.adwin = ADWINDetector(n_features=n_features)
        self.mmd = MMDDetector()
        self.ae = AutoencoderNoveltyDetector(input_dim=n_features, device=device)
        self.diagnoser = DriftDiagnoser()
        self.ewc = None
        self._history = []

    def fit_reference(self, ref_data: np.ndarray):
        self.mmd.fit(ref_data)
        self.ae.fit(ref_data)

    def setup_ewc(self, model: nn.Module, data_loader, device, lambda_ewc: float=1000.0):
        self.ewc = EWCRegularizer(model, lambda_ewc)
        self.ewc.compute_fisher(model, data_loader, device)

    def update(self, feature_vec: np.ndarray, timestamp: float=None) -> dict:
        adwin_res = self.adwin.update(feature_vec, timestamp)
        mmd_res = self.mmd.update(feature_vec, timestamp)
        ae_res = self.ae.update(feature_vec, timestamp)
        diagnosis = self.diagnoser.diagnose(adwin_res, mmd_res, ae_res)
        if diagnosis['recommended_action'] == 'adapt_reference':
            self.mmd.adapt()
        elif diagnosis['recommended_action'] in ('ewc_update', 'full_retrain'):
            self.ae.retrain(np.array([feature_vec]), epochs=5)
        self._history.append({'timestamp': timestamp, 'adwin': adwin_res, 'mmd': mmd_res, 'ae': ae_res, 'diagnosis': diagnosis})
        return diagnosis

    def get_ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        if self.ewc is not None:
            return self.ewc.penalty(model)
        return torch.tensor(0.0)

    def get_history(self) -> list:
        return self._history