import numpy as np
import pandas as pd
from shapely.geometry import box
import os
os.makedirs("labels",exist_ok=True)


def load(label_path,a,TEST_NAME):
    viewport_data = pd.read_csv(label_path + TEST_NAME + ".rep.vpd", index_col=None)
    terminal_frame = int(viewport_data['frame'][-1:].item()/3*a)
    viewport_data = viewport_data.set_index('frame')
    viewport_data = viewport_data.reindex(range(terminal_frame))
    viewport_data = viewport_data.fillna(method='ffill')
    viewport_data = viewport_data.reset_index()
    viewport_data['vpx'] = viewport_data['vpx'].astype(int)
    viewport_data['vpy'] = viewport_data['vpy'].astype(int)
    return viewport_data, terminal_frame

def eval(labels_arr, min_length):
    x_len = 20
    y_len = 12
    width = 128
    height = 128
    max_x = 3456
    max_y = 3720
    total_intersection = []
    is_intersect = []
    for i in range (0,min_length):
        total_tiles = np.zeros((width, height))
        for idx, labels in enumerate(labels_arr[1:]):
            if idx == 0 :
                labels = labels[:min_length]
                x = int(np.round(labels['vpx'][i]/max_x*(width-x_len)))
                y = int(np.round(labels['vpy'][i]/max_y*(height-y_len)))
                total_tiles[x:x+x_len, y:y+y_len] =total_tiles[x:x+x_len, y:y+y_len]+1
            else :
                labels = labels[1200:min_length+1200]
                x = int(np.round(labels['vpx'][i+1200]/max_x*(width-x_len)))
                y = int(np.round(labels['vpy'][i+1200]/max_y*(height-y_len)))
                total_tiles[x:x+x_len, y:y+y_len] =total_tiles[x:x+x_len, y:y+y_len]+1
            #total_tiles[labels['vpx'][i]:labels['vpx'][i]+x_len][labels['vpy'][i]:labels['vpy'][i]-y_len] =+1
        pred_x = int(np.round(labels_arr[0]['vpx'][i]/max_x*(width-x_len)))
        pred_y = int(np.round(labels_arr[0]['vpy'][i]/max_y*(height-y_len)))
        pred_tiles = total_tiles[pred_x:pred_x+x_len, pred_y:pred_y+y_len]
        pred_tiles =np.where(pred_tiles==0, pred_tiles, 1)
        intersection = np.mean(pred_tiles)
        total_intersection.append(intersection)
        if intersection!=0: is_intersect.append(1)
        else: is_intersect.append(0)
    print("eval done")
    return total_intersection, is_intersect


TEST_NAME = "212"
TEST_NAME = "438"
TEST_NAME = "522"
TEST_NAME = "1660"
TEST_NAME = "6254"
TEST_NAME = "36"
TEST_NAME = "6254"

print("replay_name: ",TEST_NAME)

for j in range(3,4,1):
    label_name = ["saved_xy/2_6_five_masked_new", "pdh", "jht",  "yws",  "bcm", "cyh"] #aiide, rcnn, bc
    # label_name = ["cyh", "pdh", "jht", "yws", "bcm" ]  # aiide, rcnn, bc
    # label_name = ["saved_xy/old", "jht"] #aiide, rcnn, bc
    # label_name = ["saved_xy/hotaek", "pdh", "jht",  "yws",  "bcm", "cyh"] #aiide, rcnn, bc
    # label_name = ["rcnn",  "cyh"] #aiide, rcnn, bc
    length = 0
    min_length = np.inf
    labels_arr = []

    for i in range (0,len(label_name)):
        if i == 0 :
            label_path = label_name[i] + '/'
            label, length = load(label_path, j, TEST_NAME)
        else:
            label_path = "./labels/" + label_name[i] + '/'
            label, length = load(label_path, j, TEST_NAME)
            length = length - 1200
        print("min_length:",i,  length)
        labels_arr.append(label)
        min_length = min(length, min_length)

    print("min_length:", min_length)
    total_intersection, is_intersect = eval(labels_arr, min_length)
    # print("intersect: ", total_intersection, "is_intersect: ", is_intersect)
    print(j,"/3  total_intersection percent: ", np.round(np.mean(total_intersection),3), "total_is_intersect_percent: ", np.round(np.mean(is_intersect),3))
    # print(j, "/3  total_intersection percent: ", np.round(np.mean(total_intersection), 3), ', ',np.round(np.mean(is_intersect), 3))
