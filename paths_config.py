from pathlib import Path

class config_org:
    BASE_PATH = Path("./input/hubmap-organ/hubmap-organ-segmentation")
    TRAIN_CSV_PATH = BASE_PATH / "train.csv"
    TRAIN_IMAGES_PATH = BASE_PATH / "train_images/"
    TRAIN_ANNOTATIONS_PATH = BASE_PATH / "train_annotations/" # Not needed
    TEST_CSV_PATH = BASE_PATH / "test.csv"
    TEST_IMAGES_PATH = BASE_PATH / "test_images/"

class config_kid:
    BASE_PATH= Path("./input/hubmap-kidney")
    DSET_PATH= BASE_PATH / "hubmap-kidney-segmentation"
    TRAIN_CSV_PATH = DSET_PATH / "train.csv"
    TRAIN_PATH = DSET_PATH / "train/"
    TEST_PATH = DSET_PATH / "test/"