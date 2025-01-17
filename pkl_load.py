#/media/ailab/새 볼륨/hojun_ws/data/CAMERA/train/00000

import pickle
from marshal import load


def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 사용 예시
file_path = '/media/ailab/새 볼륨/hojun_ws/data/CAMERA/train/00000/0000_label.pkl'
loaded_data = load_pkl_file(file_path)
print(loaded_data)