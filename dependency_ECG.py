from pathlib import Path

source_dir = str(Path('data') / 'MM-ECG')
img_info_path = str(Path(source_dir) / 'final_report_with_mapping_with_ecg_new.csv')
train_index_path = str(Path(source_dir) / 'splits' / 'train_indexes.csv')
test_index_path = str(Path(source_dir) / 'splits' / 'test_indexes.csv')


IMAGE_NPY_DIR = None
ECG_NPY_DIR = None

SPLIT_SEED = 42
TEST_FRACTION = 0.2
SPLIT_CANDIDATES = 2048
ECG_SHAPE = (12, 1000)
META_DIM = 20


CLASS_NAMES = ['Atrial fibrillation', 'Sinus bradycardia', 'Sinus tachycardia']
LABEL_TO_INDEX = {'A': 0, 'B': 1, 'C': 2}
num_label = len(CLASS_NAMES)


AMD_label = CLASS_NAMES
label_list = [[name] for name in CLASS_NAMES]
class_list = [num_label]
