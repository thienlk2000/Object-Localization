import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import split_train_val, normalize_box, unnormalize, IOU, tune_box, plot_training_process
import torch.nn.functional as F
from torchvision import models
from dataset import Localize_Set


parser = argparse.ArgumentParser()
parser.add_argument("root_dir", help='directory contains data with each class in correspond directory')
parser.add_argument("model_type", help='Choose model to train (ResNet18,EfficientNet_B0)')
parser.add_argument("-e", "--epoch", help='Number of epochs', type=int, default=20)
parser.add_argument('-b', '--batch_size', help='batch data image to feed to model each iteration',type=int, default=64)
parser.add_argument('-lr', '--learning_rate', help='Initial Learning Rate', type=float, default=1e-3)


args = parser.parse_args()

data_dir = args.root_dir
epoch = args.epoch
batch_size = args.batch_size
lr = args.learning_rate
model_type = args.model_type

img_dir = 'images'

data = {}

with open(os.path.join(data_dir, 'images.txt')) as f:
    lines = f.readlines()
for line in lines:
    i, path = line.split(' ')
    path = path.strip('\n')
    class_name, img_file = path.split('/')
    if class_name not in data:
        data[class_name] = [int(i)]
    else:
        data[class_name].append(int(i))

total = 0
for k, v in data.items():
    print(k, len(v))
    total += len(v)
print('total images:',total)

train_set, val_set = split_train_val(data, 0.8)

trainning_set = Localize_Set(data_dir, train_set)
validation_set = Localize_Set(data_dir, val_set)

train_loader = DataLoader(trainning_set, batch_size=64, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=64)

loss_fn = nn.MSELoss()

if model_type == 'resnet':
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 4)
elif model_type == 'efficient':
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
epoch = 20

best_acc = 0
num_correct = 0
num_sample = 0
loss_train = 0
loss_train_hist = []
acc_train_hist = []
loss_val_hist = []
acc_val_hist = []
for e in range(epoch):
    model.train()
    for i,(imgs, boxes, sizes) in enumerate(train_loader):
        imgs = imgs.to(device)
        boxes = boxes.to(device)
        # print(boxes[:2])
        # print(boxes[:5])
        sizes = sizes.to(device)
        preds = F.sigmoid(model(imgs))
        loss = loss_fn(preds, boxes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds_box = unnormalize(preds, sizes)
            preds_box = tune_box(preds_box, sizes)
            # print(preds_box[0])
            # print(boxes[0])
            
            un_boxes = unnormalize(boxes, sizes)
            # print(preds_box[0])
            # print(un_boxes[0])
            iou = IOU(un_boxes, preds_box)
            # print(boxes[:2])
            num_correct += (iou > 0.75).sum().item()
            num_sample += boxes.shape[0]
            loss_train += loss.item()
            print(f"Epoch:{e} Batch:{i} Loss:{loss.item()}")
            # break
    # break        
    acc_train_hist.append(float(num_correct)/num_sample)
    loss_train_hist.append(loss_train/len(train_loader))
    print(f"Epoch:{e} Acc:{float(num_correct)/num_sample} loss:{loss_train/len(train_loader)}")
    scheduler.step()
    num_correct = 0
    num_sample = 0
    loss_train = 0
    model.eval()
    with torch.no_grad():
        for imgs, boxes, sizes in val_loader:
            imgs = imgs.to(device)
            boxes = boxes.to(device)
            
            sizes = sizes.to(device)
            preds = F.sigmoid(model(imgs))
            loss = loss_fn(preds, boxes)

            preds_box = unnormalize(preds, sizes)
            preds_box = tune_box(preds_box, sizes)
            un_boxes = unnormalize(boxes, sizes)
            iou = IOU(un_boxes, preds_box)
            num_correct += (iou > 0.75).sum().item()
            num_sample += iou.shape[0]
            loss_train += loss.item()
        acc_val_hist.append(float(num_correct)/num_sample)
        loss_val_hist.append(loss_train/len(val_loader))
        if acc_val_hist[-1] > best_acc:
            best_acc = acc_train_hist[-1]
            torch.save(efficient.state_dict(), model_type+'.pth')
        print(f"Epoch:{e} Acc:{float(num_correct)/num_sample} loss:{loss_train/len(val_loader)}")
        num_correct = 0
        num_sample = 0
        loss_train = 0

plot_training_process(loss_train_hist, acc_train_hist, loss_val_hist, acc_val_hist)