import requests
import os
from pathlib import Path
import pickle
from shutil import unpack_archive

bras_low_quality_raw_path = Path('/home/qhw/python/RNN-Time-series-Anomaly-Detection/dataset/bras/raw/rpt_min_network_bras_new_0811-0911-61-187-89-79_rnn_raw.csv')
labeled_data = []
with open(str(bras_low_quality_raw_path),'r') as f:
    for i, line in enumerate(f):
        tokens = [float(token) for token in line.strip().split(',')]
        labeled_data.append(tokens)
bras_low_quality_train_path = bras_low_quality_raw_path.parent.parent.joinpath('labeled','train',bras_low_quality_raw_path.name).with_suffix('.pkl')
bras_low_quality_train_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(bras_low_quality_train_path),'wb') as pkl:
    pickle.dump(labeled_data[:3500], pkl)

bras_low_quality_test_path = bras_low_quality_raw_path.parent.parent.joinpath('labeled','test',bras_low_quality_raw_path.name).with_suffix('.pkl')
bras_low_quality_test_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(bras_low_quality_test_path),'wb') as pkl:
    pickle.dump(labeled_data[3500:], pkl)
