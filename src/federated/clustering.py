import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


class SubjectClusterer:
    def __init__(self, n_clusters: int = 5, feature_dim: int = 48):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = {}
        self.cluster_centroids = None
        self._is_fitted = False

    def fit(self, subject_profiles: dict):
        sids = sorted(subject_profiles.keys())
        X = np.array([subject_profiles[sid] for sid in sids])
        self.kmeans.fit(X)
        self.cluster_centroids = self.kmeans.cluster_centers_
        for i, sid in enumerate(sids):
            self.cluster_labels[sid] = int(self.kmeans.labels_[i])
        self._is_fitted = True

    def assign(self, profile: np.ndarray) -> int:
        if not self._is_fitted:
            return 0
        return int(self.kmeans.predict(profile.reshape(1, -1))[0])

    def get_cluster_members(self, cluster_id: int) -> list:
        return [sid for sid, cid in self.cluster_labels.items() if cid == cluster_id]

    def get_cluster_groups(self) -> dict:
        groups = defaultdict(list)
        for sid, cid in self.cluster_labels.items():
            groups[cid].append(sid)
        return dict(groups)

    def build_subject_profile(self, feature_windows: list) -> np.ndarray:
        if not feature_windows:
            return np.zeros(self.feature_dim, dtype=np.float32)
        stacked = np.stack(feature_windows)
        mean = stacked.mean(axis=0)
        std  = stacked.std(axis=0)
        return np.concatenate([mean[:24], std[:24]])

    def get_stats(self) -> dict:
        groups = self.get_cluster_groups()
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": {k: len(v) for k, v in groups.items()},
            "is_fitted": self._is_fitted,
        }
