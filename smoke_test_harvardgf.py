from src.data.harvardgf_pytorch import get_harvardgf_datasets_to_evaluate, get_harvardgf_test
from src.data.utils import stratified_downsampling_dataset, bootstrap_resampling

scen = get_harvardgf_datasets_to_evaluate()
print({d.dataset_name: len(d) for d in scen})        # All 2100, each race 700, Female+Male = 2100

test = get_harvardgf_test()
print("test", len(test))                               # 900

asian = [d for d in scen if d.dataset_name == "Asian"][0]
img, text, label, img_id, meta, name = asian[0]
print(img.shape, float(img.min()), float(img.max()), int(label), img_id, name)
print("metadata", meta)

ds = stratified_downsampling_dataset(asian)
print("downsampled", len(ds), ds.labels_csv["label"].value_counts().to_dict(), ds[0][0].shape)

db = bootstrap_resampling(asian)
print("bootstrap", len(db), db.labels_csv["img_id"].nunique())
