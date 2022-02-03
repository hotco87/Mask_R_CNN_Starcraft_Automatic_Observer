import numpy as np
import pandas as pd
from shapely.geometry import box
import os
os.makedirs("labels",exist_ok=True)

TEST_NAME = "6254"
def load(label_path):
    viewport_data = pd.read_csv(label_path + TEST_NAME + ".rep.vpd", index_col=None)
    terminal_frame = int(viewport_data['frame'][-1:].item()/3)
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
        for labels in labels_arr[1:]:
            labels = labels[:min_length]
            x = int(np.round(labels['vpx'][i]/max_x*(width-x_len)))
            y = int(np.round(labels['vpy'][i]/max_y*(height-y_len)))
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

label_name = ["rcnn", "pdh", "jht",  "yws",  "bcm", "cyh"] #aiide, rcnn, bc
# label_name = ["rcnn",  "cyh"] #aiide, rcnn, bc
length = 0
min_length = np.inf
labels_arr = []

for i in range (0,6):
    label_path = "./labels/" + label_name[i] +'/'
    label, length = load(label_path)
    labels_arr.append(label)
    min_length = min(length, min_length)
total_intersection, is_intersect = eval(labels_arr,min_length)
#print("intersect: ", total_intersection, "is_intersect: ", is_intersect)
print("total_intersection percent: ", np.mean(total_intersection), "total_is_intersect_percent: ", np.mean(is_intersect))