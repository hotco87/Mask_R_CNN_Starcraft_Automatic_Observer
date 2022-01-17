import numpy as np
import pandas as pd

filename = "test"
xy = np.load("./saved_xy/" + filename + ".npy", allow_pickle=True)

xy = np.where(xy>0, xy, 0)
xy[:,0] = np.where(xy[:,0]<3456,xy[:,0],3456)
xy[:,1] = np.where(xy[:,1]<3720,xy[:,1],3720)

xy[:,0] = xy[:,0]*3456
xy[:,1] = xy[:,1]*3720

xy = xy.astype(int)

temp = np.zeros((xy.shape[0],1))
for i in range (0,xy.shape[0]):
    temp[i] = i*8

temp = temp.astype(int)
temp2 = np.zeros((temp.max(),1))
for i in range(0, temp.max()):
    temp2[i] = int(i)

print(temp)
dataset_temp = pd.DataFrame({"frame": temp[:,0],"vpx": xy[:,0],"vpy": xy[:,1]})
dataset_temp2 = pd.DataFrame({"frame": temp2[:,0]})
dataset = pd.merge(left = dataset_temp2, right = dataset_temp, how= "left", on = "frame")
dataset = dataset.fillna(method="ffill")

dataset.to_csv("./saved_xy/" + filename + ".vpd", header= True, index=False)
print("loaded")
