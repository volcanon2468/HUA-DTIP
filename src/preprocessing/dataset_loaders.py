import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
MHEALTH_COLUMNS = ['chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'ecg_1', 'ecg_2', 'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z', 'wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'wrist_gyro_x', 'wrist_gyro_y', 'wrist_gyro_z', 'wrist_mag_x', 'wrist_mag_y', 'wrist_mag_z', 'activity_label']
IMU_COLS = ['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'wrist_gyro_x', 'wrist_gyro_y', 'wrist_gyro_z', 'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z']
ECG_COLS = ['ecg_1', 'ecg_2']

class MHEALTHDataset(Dataset):

    def __init__(self, data_dir: str, subject_ids: list=None, split: str='all'):
        self.data_dir = data_dir
        self.subject_ids = subject_ids or list(range(1, 11))
        self.split = split
        self.records = []
        self._load()

    def _load(self):
        for sid in self.subject_ids:
            path = os.path.join(self.data_dir, 'MHEALTHDATASET', f'mHealth_subject{sid}.log')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep='\\s+', header=None, names=MHEALTH_COLUMNS)
            df = df[df['activity_label'] != 0]
            df['subject_id'] = sid
            self.records.append(df)

    def get_subject_df(self, subject_id: int) -> pd.DataFrame:
        for df in self.records:
            if df['subject_id'].iloc[0] == subject_id:
                return df
        raise KeyError(f'Subject {subject_id} not loaded.')

    def get_all_dfs(self) -> list:
        return self.records

    def __len__(self):
        return sum((len(df) for df in self.records))

    def __getitem__(self, idx):
        for df in self.records:
            if idx < len(df):
                row = df.iloc[idx]
                imu = row[IMU_COLS].values.astype(np.float32)
                ecg = row[ECG_COLS].values.astype(np.float32)
                label = int(row['activity_label'])
                return (imu, ecg, label)
            idx -= len(df)
        raise IndexError('Index out of range.')
PAMAP2_IMU_COLS = ['hand_acc_16_x', 'hand_acc_16_y', 'hand_acc_16_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 'chest_acc_16_x', 'chest_acc_16_y', 'chest_acc_16_z']
PAMAP2_HR_COL = 'heart_rate'

def _build_pamap2_columns():
    base = ['timestamp', 'activity_label', 'heart_rate']
    for loc in ['hand', 'chest', 'ankle']:
        base += [f'{loc}_temp', f'{loc}_acc_16_x', f'{loc}_acc_16_y', f'{loc}_acc_16_z', f'{loc}_acc_6_x', f'{loc}_acc_6_y', f'{loc}_acc_6_z', f'{loc}_gyro_x', f'{loc}_gyro_y', f'{loc}_gyro_z', f'{loc}_mag_x', f'{loc}_mag_y', f'{loc}_mag_z', f'{loc}_orient_w', f'{loc}_orient_x', f'{loc}_orient_y', f'{loc}_orient_z']
    return base
PAMAP2_COLUMNS = _build_pamap2_columns()

class PAMAP2Dataset(Dataset):

    def __init__(self, data_dir: str, subject_ids: list=None):
        self.data_dir = data_dir
        self.subject_ids = subject_ids or list(range(101, 110))
        self.records = []
        self._load()

    def _load(self):
        for sid in self.subject_ids:
            path = os.path.join(self.data_dir, 'PAMAP2_Dataset', 'Protocol', f'subject{sid}.dat')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep='\\s+', header=None, names=PAMAP2_COLUMNS)
            df = df[df['activity_label'] != 0]
            df[PAMAP2_HR_COL] = df[PAMAP2_HR_COL].ffill()
            df[PAMAP2_IMU_COLS] = df[PAMAP2_IMU_COLS].interpolate(method='linear', limit_direction='both')
            df['subject_id'] = sid
            self.records.append(df)

    def get_subject_df(self, subject_id: int) -> pd.DataFrame:
        for df in self.records:
            if df['subject_id'].iloc[0] == subject_id:
                return df
        raise KeyError(f'Subject {subject_id} not loaded.')

    def get_all_dfs(self) -> list:
        return self.records

    def __len__(self):
        return sum((len(df) for df in self.records))

    def __getitem__(self, idx):
        for df in self.records:
            if idx < len(df):
                row = df.iloc[idx]
                imu = row[PAMAP2_IMU_COLS].values.astype(np.float32)
                hr = np.array([row[PAMAP2_HR_COL]], dtype=np.float32)
                label = int(row['activity_label'])
                return (imu, hr, label)
            idx -= len(df)
        raise IndexError('Index out of range.')

class FourWeekPPGDataset(Dataset):

    def __init__(self, data_dir: str, subject_ids: list=None):
        self.data_dir = data_dir
        self.subject_ids = subject_ids or list(range(1, 50))
        self.records = {}
        self._load()

    def _load(self):
        for sid in self.subject_ids:
            path = os.path.join(self.data_dir, f'subject_{sid}.csv')
            if not os.path.exists(path):
                path = os.path.join(self.data_dir, 'sensor_data.csv')
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path, parse_dates=['timestamp'])
                df = df[df['subject_id'] == sid] if 'subject_id' in df.columns else df
            else:
                df = pd.read_csv(path, parse_dates=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['subject_id'] = sid
            df['date'] = df['timestamp'].dt.date
            days = {}
            for d, grp in df.groupby('date'):
                days[str(d)] = grp.reset_index(drop=True)
            self.records[sid] = days

    def get_subject_days(self, subject_id: int) -> dict:
        return self.records.get(subject_id, {})

    def __len__(self):
        return sum((len(days) for days in self.records.values()))

    def __getitem__(self, idx):
        for sid, days in self.records.items():
            for day_key, df in days.items():
                if idx == 0:
                    return (sid, day_key, df)
                idx -= 1
        raise IndexError('Index out of range.')

class StrokeRehabDataset(Dataset):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.records = []
        self._load()

    def _load(self):
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(self.data_dir, fname))
            parts = fname.replace('.csv', '').split('_')
            patient_id = int(parts[1]) if len(parts) > 1 else -1
            visit = int(parts[-1].replace('visit', '')) if 'visit' in parts[-1] else 1
            df['patient_id'] = patient_id
            df['visit'] = visit
            self.records.append(df)

    def get_all_dfs(self) -> list:
        return self.records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

class CAPTURE24Dataset(Dataset):

    def __init__(self, data_dir: str, subject_ids: list=None):
        self.data_dir = data_dir
        self.subject_ids = subject_ids or list(range(1, 152))
        self.records = []
        self._load()

    def _load(self):
        for sid in self.subject_ids:
            path = os.path.join(self.data_dir, f'subject_{sid}.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['subject_id'] = sid
            self.records.append(df)

    def get_all_dfs(self) -> list:
        return self.records

    def __len__(self):
        return sum((len(df) for df in self.records))

    def __getitem__(self, idx):
        for df in self.records:
            if idx < len(df):
                return df.iloc[idx]
            idx -= len(df)
        raise IndexError('Index out of range.')

class MExDataset(Dataset):

    def __init__(self, data_dir: str, subject_ids: list=None):
        self.data_dir = data_dir
        self.subject_ids = subject_ids or list(range(1, 31))
        self.records = []
        self._load()

    def _load(self):
        for sid in self.subject_ids:
            path = os.path.join(self.data_dir, f'subject_{sid}.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['subject_id'] = sid
            self.records.append(df)

    def get_all_dfs(self) -> list:
        return self.records

    def __len__(self):
        return sum((len(df) for df in self.records))

    def __getitem__(self, idx):
        for df in self.records:
            if idx < len(df):
                return df.iloc[idx]
            idx -= len(df)
        raise IndexError('Index out of range.')