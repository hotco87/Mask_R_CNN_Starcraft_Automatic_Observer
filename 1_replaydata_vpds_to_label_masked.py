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

    print(result)
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
    frames = np.zeros((np.max(vpds_tile.index.max()) + 1,) + (origin_shape))
    kernel = np.ones(kernel_shape)

    for idx, row in vpds_tile.iterrows():
        for i in range(num_dataset):
            frames[idx][row['vpx_{}'.format(i + 1)]:row['vpx_{}'.format(i + 1)] + kernel.shape[0],
            row['vpy_{}'.format(i + 1)]:row['vpy_{}'.format(i + 1)] + kernel_shape[1]] += kernel
    return frames


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


path = "./data_compressed3/"
pth = os.listdir(path)
for i in pth:
    #if i == "4037": # 한개만 할 때
        if os.path.isdir(path + i):
            print(path+i)
            vpds, num_dataset = get_players_vpds(path + i +'/')
            vpds_tile = (vpds / TILE_SIZE).astype(int)
            channel_vpds = mapping_vpd_into_channel(vpds_tile, num_dataset=1)
            np.save(path + i +'/' + "vpds_label_masked.npy", channel_vpds)