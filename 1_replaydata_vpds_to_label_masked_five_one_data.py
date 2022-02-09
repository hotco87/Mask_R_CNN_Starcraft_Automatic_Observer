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


def kernel_sum(frame, origin_shape=ORIGIN_SHAPE, kernel_shape=KERNEL_SHAPE):
    width_tile = origin_shape[0] - kernel_shape[0]
    height_tile = origin_shape[1] - kernel_shape[1]

    result = np.zeros((width_tile, height_tile))

    for x in range(width_tile):
        for y in range(height_tile):
            result[x][y] = frame[x:x + kernel_shape[0], y:y + kernel_shape[1]].sum()
            # print(x, x+kernel_width, y, y+kernel_height, frame[x:x+kernel_width,y:y+kernel_height].sum())
            # print(frame[x:x+kernel_width,y:y+kernel_height].shape)

    max_vpx = np.argmax(result)
    max_vpx_tile = (max_vpx // height_tile) * TILE_SIZE
    max_vpy_tile = (max_vpx % height_tile) * TILE_SIZE

    return result, (max_vpx_tile, max_vpy_tile)

def vpds_tile_into_channel_kernel_sum(channel_vpds):
    res_kernel_sum = []
    max_vp_coord = []

    for t, frame in tqdm(enumerate(channel_vpds), total=channel_vpds.shape[0]):
        res, coord = kernel_sum(frame, kernel_shape=KERNEL_SHAPE)
        res_kernel_sum.append(res)
        max_vp_coord.append(coord)

    return np.asarray(res_kernel_sum), np.asarray(max_vp_coord)

def load_all_data(path, labels, type = "training"):

    replay_name = path.split('/')[-2]

    data_path = path
    label_path = path.replace(replay_name + "/", '')

    data = np.load(data_path + replay_name + ".rep.channels_compressed.npz" )["data"]
    # label = label_path + replay_name + '/' + "vpds_label_masked.npy"
    # if os.path.exists(labels):
    #     # os.mkdir("./data/" + replay_name)
    #     # np.save("./data2/" + replay_name +"/" + replay_name +".npy", data)
    #     labels = np.load(label)
    # else:
    #     print (replay_name, ": No label file")
    #     results = {
    #     "data": [],
    #     "label": []}
    #     return results

    temp = []
    for i in range(0,labels.shape[0],8):
        temp.append(labels[i])

    labels = np.array(temp)
    len_min = min(data.shape[0], labels.shape[0])
    # len = labels.shape[0]


    for i in range(0, len_min):
        data_single = np.array([[data[i]],[labels[i]]])
        if type == "training":
            os.makedirs("./trainig_data_five_one/" + replay_name, exist_ok=True)
            np.save("./trainig_data_five_one/" + replay_name + '/' +str(i) + ".npy", data_single)
        if type == "testing":
            os.makedirs("./testing_data_five_one/" + replay_name + "/" + replay_name, exist_ok=True)
            np.save("./testing_data_five_one/" + replay_name + '/' + replay_name + '/' + str(i) + ".npy", data_single)

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


# def mapping_vpd_into_channel(vpds_tile, kernel_shape=KERNEL_SHAPE, origin_shape=ORIGIN_SHAPE, num_dataset=5):
#     frames = np.zeros((num_dataset, np.max(vpds_tile.index.max()) + 1,) + (origin_shape))
#     kernel = np.ones(kernel_shape)
#
#     for idx, row in vpds_tile.iterrows():
#         for i in range(num_dataset):
#             frames[i][idx][row['vpx_{}'.format(i + 1)]:row['vpx_{}'.format(i + 1)] + kernel.shape[0],
#             row['vpy_{}'.format(i + 1)]:row['vpy_{}'.format(i + 1)] + kernel_shape[1]] += kernel
#     return frames

def mapping_vpd_into_channel(vpds_tile, kernel_shape=KERNEL_SHAPE, origin_shape=ORIGIN_SHAPE, num_dataset=5):
    frames = np.zeros((np.max(vpds_tile.index.max()) + 1,) + (origin_shape))
    kernel = np.ones(kernel_shape)

    for idx, row in vpds_tile.iterrows():
        for i in range(num_dataset):
            frames[idx][row['vpx_{}'.format(i + 1)]:row['vpx_{}'.format(i + 1)] + kernel.shape[0],
            row['vpy_{}'.format(i + 1)]:row['vpy_{}'.format(i + 1)] + kernel_shape[1]] += kernel
    return frames

path = "./data_compressed4/"
pth = os.listdir(path)
num_dataset = 5
# training = ['36', '212', '438', '522', '1660']
training = []
testing = ['6254']  # , '4037', '1725']
os.makedirs("./testing_data_five_one/", exist_ok=True)
os.makedirs("./trainig_data_five_one/", exist_ok=True)

for i in pth:
    # if i in ["36","212","438","522","1660","6254"]:
    # if i in ["6254"]:
        if os.path.isdir(path + i):
            print(path+i)
            if i in training:  # 한 개만 할 때  # 저장할 때 training data와 test data를 구분해야 함.
                vpds, num_dataset = get_players_vpds(path + i + '/')
                vpds_tile = (vpds / TILE_SIZE).astype(int)
                channel_vpds = mapping_vpd_into_channel(vpds_tile, num_dataset=num_dataset)
                channel_kernel_sums, max_vpd_coords = vpds_tile_into_channel_kernel_sum(channel_vpds)
                vpds_label = pd.DataFrame((max_vpd_coords / 32).astype(int), columns=['vpx_1', 'vpy_1'])
                vpds_label_masked = mapping_vpd_into_channel(vpds_label, num_dataset=1)
                load_all_data(path + i + '/', vpds_label_masked, type="training")
                print(f"Loaded {i}")
            if i in testing:  # 한 개만 할 때  # 저장할 때 training data와 test data를 구분해야 함.
                vpds, num_dataset = get_players_vpds(path + i + '/')
                vpds_tile = (vpds / TILE_SIZE).astype(int)
                channel_vpds = mapping_vpd_into_channel(vpds_tile, num_dataset=num_dataset)
                channel_kernel_sums, max_vpd_coords = vpds_tile_into_channel_kernel_sum(channel_vpds)
                vpds_label = pd.DataFrame((max_vpd_coords / 32).astype(int), columns=['vpx_1', 'vpy_1'])
                vpds_label_masked = mapping_vpd_into_channel(vpds_label, num_dataset=1)
                load_all_data(path + i + '/', vpds_label_masked,  type="testing")
                print(f"Loaded {i}")
            else:
                pass


