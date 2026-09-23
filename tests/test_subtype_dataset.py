from src.data.dataset import SUBTYPE_LABEL_MAP, BreakHisSubtypeDataset


def test_subtype_dataset_excludes_benign_samples(breakhis_root):
    ds = BreakHisSubtypeDataset(breakhis_root)
    assert len(ds) > 0
    assert all(s["subtype"] != "adenosis" for s in ds.samples)


def test_subtype_dataset_labels_match_subtype_map(breakhis_root_all_subtypes):
    ds = BreakHisSubtypeDataset(breakhis_root_all_subtypes)
    assert len(ds) == 8 * 3  # 8 patients, 3 images each

    for sample in ds.samples:
        assert sample["label"] == SUBTYPE_LABEL_MAP[sample["subtype"]]


def test_subtype_dataset_getitem_returns_subtype_label(breakhis_root_all_subtypes):
    ds = BreakHisSubtypeDataset(breakhis_root_all_subtypes)
    image, label = ds[0]
    assert label in SUBTYPE_LABEL_MAP.values()
