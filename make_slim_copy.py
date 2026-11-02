from pathlib import Path
import numpy as np
from tqdm import tqdm

src = Path("data/raw/dataset")
dst = Path("data/slim/dataset")
keys = ["rnflt", "md", "glaucoma", "age", "male", "race", "hispanic", "language", "maritalstatus"]

for f in tqdm(sorted(src.glob("*/*.npz"))):
    out = dst / f.parent.name / f.name
    out.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(f, allow_pickle=True)
    np.savez(out, **{k: z[k] for k in keys})
