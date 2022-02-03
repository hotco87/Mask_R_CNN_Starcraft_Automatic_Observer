# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import argparse

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import time

def load_all_label(path):
    replay_name = path.split('/')[-2]
    # print("replay_name:", replay_name)
    label_path = path.replace(replay_name + "/", '')
    # print("label_path:", label_path)
    label_len = len(os.listdir(label_path + replay_name +"/"))
    # print("label_len:", label_len)
    labels = np.array([])
    for i in range(0, label_len):
        labels = np.append(labels, np.load(label_path + replay_name +"/"+ str(i) + ".npy", allow_pickle=True)[1])

    results = labels

    return results

class PennFudanDataset(object):
    def __init__(self, path, transforms, window_size):
        self.root = path
        self.transforms = transforms
        pth = os.listdir(path)
        self.label_sequences = []

        self.dir_paths = []
        for i in pth:
            if os.path.isdir(path + i):
                self.label_sequences += [load_all_label(path + i +'/')]
                self.dir_paths.append(path + i + '/')
                print(f"Loaded {i}")

        self.window_size = window_size

        self.seq_indexs = []
        start = 0
        for i, seq in enumerate(self.label_sequences):
            end = start + len(seq) - (self.window_size - 1) - 150*(i+1)
            self.seq_indexs.append((i, start, end))
            start = end


    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        # load images and masks
        for i, start, end in self.seq_indexs:
            if idx >= start and idx < end:
                real_idx = idx - start + 150 # *(i+1)

                data = np.load(self.dir_paths[i] + '/' + str(real_idx) + ".npy", allow_pickle=True)[0][0]    # (11, 128, 128)
                masks1 = np.load(self.dir_paths[i] + '/' + str(real_idx) + ".npy", allow_pickle=True)[1][0]   #  (128, 128)
                masks2 = np.load(self.dir_paths[i] + '/' + str(real_idx) + ".npy", allow_pickle=True)[2][0]  # (128, 128)
                masks3 = np.load(self.dir_paths[i] + '/' + str(real_idx) + ".npy", allow_pickle=True)[3][0]  # (128, 128)
                masks4 = np.load(self.dir_paths[i] + '/' + str(real_idx) + ".npy", allow_pickle=True)[4][0]  # (128, 128)
                masks5 = np.load(self.dir_paths[i] + '/' + str(real_idx) + ".npy", allow_pickle=True)[5][0]  # (128, 128)
                masks = np.stack((masks1, masks2,masks3,masks4,masks5))

                break

        input_data = self.preprocessing(data)

        num_objs = 5
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return input_data, target


    def preprocessing(self, data):
        #0 ground 1 air 2 building 3 spell 4 ground 5 air 6 building 7 spell 8 resource 9 vision 10 terrain
        temp = np.zeros([self.window_size, 9, data.shape[2],data.shape[2]])

        temp[:,0] = data[0]
        temp[:,1] = data[1]
        temp[:,2] = data[2]
        temp[:,3] = data[4]

        temp[:,4] = data[6]
        temp[:,5] = data[7]
        temp[:,6] = data[8]
        temp[:,7] = data[10]

        temp[:,8] = data[13]


        data = temp
        data = data.reshape(self.window_size*data.shape[1],data.shape[2],-1)
        # #data = data.reshape(self.window_size*data.shape[0],data.shape[1],-1)
        # label = np.array([label[0]/3456, label[1]/3720])
        return torch.FloatTensor(data)

    # def __len__(self):
    #     return len(self.imgs)

import torch.nn as nn
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.backbone.body.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):

    torch.cuda.empty_cache()
    data_path = args.load_dir
    log_save_path = os.path.join(
        args.log_save_dir,
        f"model_{args.id_string}_lr{args.learning_rate}_w_size{args.window_size}_{str(int(time.time()))[4:]}/"
    )

    #model_type = args.model_type
    batch_size = args.batch_size
    window_size = args.window_size
    learning_rate = args.learning_rate
    cuda = args.cuda
    cuda_idx = args.cuda_idx
    max_epoch = args.max_epoch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset(data_path, get_transform(train=False), window_size=1)
    dataset_test = PennFudanDataset(data_path, get_transform(train=False),window_size=1)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 30
    data_path_test = "./saved_models/"
    os.makedirs(data_path_test, exist_ok=True)
    # torch.load("./saved_models/model_0.pth")
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), os.path.join(data_path_test, f"model_{epoch}.pth"))

    print("That's it!")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--log_save-dir", type=str, default=f"./saved_models/")
    parser.add_argument("--load-model", type=bool, default=False)
    parser.add_argument("--load-dir", type=str, default=f"./trainig_data2/")
    parser.add_argument("--batch-size", type=int, default=64) #256
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--id-string", type=str, default="")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--eval", type=bool, default=False)

    args = parser.parse_args()

    main(args)

