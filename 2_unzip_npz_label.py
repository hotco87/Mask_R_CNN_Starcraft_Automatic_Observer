import os
import numpy as np
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

def load_all_data(path, type = "training"):

    replay_name = path.split('/')[-2]

    data_path = path
    label_path = path.replace(replay_name + "/", '')

    data = np.load(data_path + replay_name + ".rep.channels_compressed.npz" )["data"]
    label = label_path + replay_name + '/' + "vpds_label_masked.npy"
    if os.path.exists(label):
        # os.mkdir("./data/" + replay_name)
        # np.save("./data2/" + replay_name +"/" + replay_name +".npy", data)
        labels = np.load(label)
    else:
        print (replay_name, ": No label file")
        results = {
        "data": [],
        "label": []}
        return results

    temp = []
    for i in range(0,labels.shape[0],8):
        temp.append(labels[i])

    labels = np.array(temp)
    len_min = min(data.shape[0], labels.shape[0])
    # len = labels.shape[0]


    for i in range(0, len_min):
        data_single = np.array([[data[i]],[labels[i]]])
        if type == "training":
            os.makedirs("./trainig_data/" + replay_name, exist_ok=True)
            np.save("./trainig_data/" + replay_name + '/' +str(i) + ".npy", data_single)
        if type == "testing":
            os.makedirs("./testing_data/" + replay_name + "/" + replay_name, exist_ok=True)
            np.save("./testing_data/" + replay_name + '/' + replay_name + '/' + str(i) + ".npy", data_single)

    results = {
        "data": data[:len_min,:],
        "label": labels[:len_min,1:]
    }

    return results

path = "./data_compressed3/"
pth = os.listdir(path)
training = ['36', '212', '438', '522', '1660']
testing = ['6254', '4037', '1725']

os.makedirs("./testing_data/", exist_ok=True)
os.makedirs("./trainig_data/", exist_ok=True)

for i in pth:
    if os.path.isdir(path + i):
        if i in training: # 한 개만 할 때  # 저장할 때 training data와 test data를 구분해야 함.
            load_all_data(path + i +'/', type = "training")
            print(f"Loaded {i}")
        if i in testing: # 한 개만 할 때  # 저장할 때 training data와 test data를 구분해야 함.
            load_all_data(path + i +'/', type = "testing")
            print(f"Loaded {i}")
