import numpy as np
import torch
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def split_train_val(file_name, ratio_train, ratio_val=None):
    train_set = []
    val_set = []
    for k, v in file_name.items():
        train_sample = int(len(v)*ratio_train)
        val_sample = int(len(v)*ratio_val) if ratio_val is not None else len(v) - train_sample
        shuffle_index = np.random.permutation(len(v))
        train_index = shuffle_index[:train_sample]
        val_index = shuffle_index[train_sample:train_sample+val_sample]
        train_set += [v[i] for i in train_index]
        val_set += [v[i] for i in val_index]
    return train_set, val_set

def normalize_box(box, size):
    box_copy = box.clone() 
    box_copy = box_copy.view(-1, 4)
    size = size.view(-1, 2)
    box_copy[:, 0] /= size[:,0]
    box_copy[:, 1] /= size[:,1]
    box_copy[:, 2] /= size[:,0]
    box_copy[:, 3] /= size[:,1]
    return box_copy

def unnormalize(box, size):
    box_copy = box.clone() 
    box_copy = box_copy.view(-1, 4)
    size = size.view(-1, 2)
    box_copy[:, 0] *= size[:,0]
    box_copy[:, 1] *= size[:,1]
    box_copy[:, 2] *= size[:,0]
    box_copy[:, 3] *= size[:,1]
    return box_copy
    
def plot_img_box(img, size, box, pred=None):
    # print(box)
    # print(size)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img*std + mean
    img = T.ToPILImage()(img)
    img = T.Resize((int(size[1]), int(size[0])))(img)
    img = np.asarray(img)
    plt.imshow(img)
    box = unnormalize(box, size)
    box = box.squeeze()
    print(box)
    plt.gca().add_patch(
            plt.Rectangle( (box[0], box[1]), box[2], box[3],
                           fill=False, edgecolor='green', linewidth=2, alpha=0.5)

        )
    
def IOU(box, pred):
    box = xywh_to_x1y1x2y2(box)
    pred = xywh_to_x1y1x2y2(pred)
    intersect = torch.zeros_like(box)
    intersect[:, 0] = torch.maximum(box[:, 0], pred[:, 0])
    intersect[:, 1] = torch.maximum(box[:, 1], pred[:, 1])
    intersect[:, 2] = torch.minimum(box[:, 2], pred[:, 2])
    intersect[:, 3] = torch.minimum(box[:, 3], pred[:, 3])
    area_intersect = (intersect[:, 2] - intersect[:,0]).clip(0)*(intersect[:, 3] - intersect[:, 1]).clip(0)
    area_1 = (box[:,2]-box[:,0])*(box[:,3]-box[:,1])
    area_2 = (pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])
    return area_intersect / (area_1 + area_2 - area_intersect)
    
    
def xywh_to_x1y1x2y2(box):
    box = box.clone()
    box[:, 2] += box[:, 0]
    box[:, 3] += box[:, 1]
    return box

def x1y1x2y2_to_xywh(box):
    box = box.clone()
    box[:, 2] -= box[:, 0]
    box[:, 3] -= box[:, 1]
    return box

def tune_box(box, img_size):
    box = box.clone()
    box = xywh_to_x1y1x2y2(box)
    box[:, 0] = torch.maximum(torch.minimum(box[:, 0], img_size[:, 0]), torch.tensor([0]).to(device))
    box[:, 1] = torch.maximum(torch.minimum(box[:, 1], img_size[:, 1]), torch.tensor([0]).to(device))
    box[:, 2] = torch.maximum(torch.minimum(box[:, 2], img_size[:, 0]), box[:, 0])
    box[:, 3] = torch.maximum(torch.minimum(box[:, 3], img_size[:, 1]), box[:, 1])
    box = x1y1x2y2_to_xywh(box)
    return box

def plot_training_process(train_loss, train_acc, val_loss, val_acc):
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(train_loss, 'b', label='Train')
  plt.plot(val_loss, 'r', label='Val')
  plt.legend()
  plt.xlabel('Iter')
  plt.ylabel('Loss')
  plt.figure()
  plt.subplot(2,1,2)
  plt.plot(train_acc, 'b', label='Train')
  plt.plot(val_acc, 'r', label='Val')
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.show()