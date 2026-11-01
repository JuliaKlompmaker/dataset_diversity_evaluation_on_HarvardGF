import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

HARVARDGF_RAW = RAW_DATA_DIR / "dataset"   # folder holding Training/, Validation/, Test/
NORM_MAX = 1033.0   # global max over all 3,300 rnflt maps = 1032.25584
def normalise(x, norm):
    x = x.astype(np.float32)
    if norm == "raw":        return x                                       # like the official EyeFair loader
    if norm == "fixed":      return (x + 2) / (NORM_MAX + 2)                # one global scale, nothing clipped
    if norm == "clip350":    return (np.clip(x, -2, 350) + 2) / 352.0
    if norm == "disc_zero":  return np.clip(np.where(x < 0, 0, x), 0, 350) / 350.0
    raise ValueError(norm)

class HarvardGFDataset(Dataset):
    def __init__(self, split, dataset_name, group=None, norm="fixed", as_tensor=False):
        self.labels_csv = pd.read_csv(
            PROCESSED_DATA_DIR / "harvardgf" / f"{split.lower()}_labels.csv", index_col=0)
        if group:
            col, val = group
            self.labels_csv = self.labels_csv[self.labels_csv[col] == val]
        self.labels_csv = self.labels_csv.reset_index(drop=True)
        self.labels_csv["img_paths"] = [str(HARVARDGF_RAW / s / f) for s, f in
                                        zip(self.labels_csv["split_dir"], self.labels_csv["img_id"])]
        self.norm, self.as_tensor, self.dataset_name = norm, as_tensor, dataset_name

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        row = self.labels_csv.iloc[idx]
        rnflt = np.load(row["img_paths"], allow_pickle=True)["rnflt"]
        image = np.stack([normalise(rnflt, self.norm)] * 3, axis=0)   # (3, 200, 200)
        metadata = np.array([row["age"] / 100.0, row["male"], row["hispanic"],
                             row["language"], row["maritalstatus"], row["race_code"]], dtype=float)
        label = int(row["label"])
        if self.as_tensor:
            return torch.Tensor(image), "", torch.tensor(label), row["img_id"], metadata, self.dataset_name
        return image, "", label, row["img_id"], metadata, self.dataset_name

def get_harvardgf_datasets_to_evaluate(norm="fixed"):
    configs = {"All": None, "Female": ["male", 0], "Male": ["male", 1],
               "Asian": ["race", "Asian"], "Black": ["race", "Black or African American"],
               "White": ["race", "White or Caucasian"]}
    return [HarvardGFDataset("training", n, group=g, norm=norm, as_tensor=True) for n, g in configs.items()]

def get_harvardgf_test(norm="fixed"):
    return HarvardGFDataset("test", "test", group=None, norm=norm, as_tensor=True)
