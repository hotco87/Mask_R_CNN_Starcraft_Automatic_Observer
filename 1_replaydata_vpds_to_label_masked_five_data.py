import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import patches
import os
from tqdm import tqdm

from scipy.spatial import distance
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure, binary_erosion



SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

HUD_WIDTH = 0
HUD_HEIGHT = 96 # 640 x 384
# HUD_HEIGHT = 104 # 640 x 376

ORIGIN_SHAPE = (128, 128)
KERNEL_SHAPE = (20, 12)
TILE_SIZE = 32

def load_all_data(path, labels, type = "training"):

    replay_name = path.split('/')[-2]
    print(replay_name)

    data_path = path
    label_path = path.replace(replay_name + "/", '')

    data = np.load(data_path + replay_name + ".rep.channels_compressed.npz" )["data"]
    # labels = label_path + replay_name + '/' + "vpds_label_masked.npy"

    temp = [[],[],[],[],[]]
    for j in range(len(labels)):
        for i in range(0,labels[0].shape[0],8):
            temp[j].append(labels[j][i])

    labels = np.array(temp)
    len_min = min(data.shape[0], labels[0].shape[0])
    # len = labels.shape[0]


    for i in range(0, len_min):
        data_single = np.array([[data[i]],[labels[0][i]],[labels[1][i]],[labels[2][i]],[labels[3][i]],[labels[4][i]]])
        if type == "training":
            os.makedirs("./trainig_data2/" + replay_name, exist_ok=True)
            np.save("./trainig_data2/" + replay_name + '/' +str(i) + ".npy", data_single)
        if type == "testing":
            os.makedirs("./testing_data2/" + replay_name + "/" + replay_name, exist_ok=True)
            np.save("./testing_data2/" + replay_name + '/' + replay_name + '/' + str(i) + ".npy", data_single)

    results = {
        "data": data[:len_min,:],
        "label": labels[:len_min,1:]
    }

    return results


def get_players_vpds(data_dir):
    vpd_paths = glob(data_dir + "*.vpd")

    dataframes = [pd.read_csv(_, index_col=None) for _ in vpd_paths]
    num_dataset = len(dataframes)

    result = []
    for dataframe in dataframes:
        dataframe = dataframe.set_index('frame')
        dataframe = dataframe.reindex(range(dataframe.tail(1).index[0]))
        dataframe = dataframe.fillna(method='ffill')
        dataframe = dataframe.reset_index()
        dataframe = dataframe.astype(int)
        dataframe = dataframe.set_index('frame')
        result.append(dataframe)

    # print(result)
    dataframes = pd.concat(result, axis=1)
    dataframes = dataframes.fillna(method='ffill')
    dataframes = dataframes.astype(int)

    dataframes_columns = []
    for i in range(num_dataset):
        dataframes_columns.append('vpx_{}'.format(i + 1))
        dataframes_columns.append('vpy_{}'.format(i + 1))
    dataframes.columns = dataframes_columns

    return dataframes, num_dataset


def mapping_vpd_into_channel(vpds_tile, kernel_shape=KERNEL_SHAPE, origin_shape=ORIGIN_SHAPE, num_dataset=5):
    frames = np.zeros((num_dataset, np.max(vpds_tile.index.max()) + 1,) + (origin_shape))
    kernel = np.ones(kernel_shape)

    for idx, row in vpds_tile.iterrows():
        for i in range(num_dataset):
            frames[i][idx][row['vpx_{}'.format(i + 1)]:row['vpx_{}'.format(i + 1)] + kernel.shape[0],
            row['vpy_{}'.format(i + 1)]:row['vpy_{}'.format(i + 1)] + kernel_shape[1]] += kernel
    return frames


path = "./data_compressed4/"
pth = os.listdir(path)
num_dataset = 5
training = ['36', '212', '438', '522', '1660']
testing = ['6254']  # , '4037', '1725']
os.makedirs("./testing_data2/", exist_ok=True)
os.makedirs("./trainig_data2/", exist_ok=True)

for i in pth:
    # if i in ["36","212","438","522","1660","6254"]:
    # if i in ["6254"]:
        if os.path.isdir(path + i):
            print(path+i)


            if i in training:  # 한 개만 할 때  # 저장할 때 training data와 test data를 구분해야 함.
                vpds, num_dataset = get_players_vpds(path + i + '/')
                vpds_tile = (vpds / TILE_SIZE).astype(int)
                channel_vpds = mapping_vpd_into_channel(vpds_tile, num_dataset=num_dataset)
                load_all_data(path + i + '/', channel_vpds, type="training")
                print(f"Loaded {i}")
            if i in testing:  # 한 개만 할 때  # 저장할 때 training data와 test data를 구분해야 함.
                vpds, num_dataset = get_players_vpds(path + i + '/')
                vpds_tile = (vpds / TILE_SIZE).astype(int)
                channel_vpds = mapping_vpd_into_channel(vpds_tile, num_dataset=num_dataset)
                load_all_data(path + i + '/', channel_vpds,  type="testing")
                print(f"Loaded {i}")
            else:
                pass

            #for j in range(num_dataset):
            #    np.save(path + i +'/' + "vpds_label_masked_v"+ str(j+1)+ ".npy", channel_vpds[j])


