import argparse
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as T
from PIL import Image
from torchvision import models
import torch.nn as nn
from utils import unnormalize, tune_box


parser = argparse.ArgumentParser()
parser.add_argument('data', help='data folder contain image to detect')
# parser.add_argument('label', help='label class name of data')
parser.add_argument('model_type', help='model consist alexnet, efficientnet, resnet, vgg')
parser.add_argument('model_weight', help='weight of pretrained model')

args = parser.parse_args()
model_type = args.model_type
data_dir = args.data
weights = args.model_weight
# label = args.label

if model_type == 'resnet':
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 4)
elif model_type == 'efficient':
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(1280, 4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(weights)
model.load_state_dict(torch.load(weights,map_location=torch.device(device)))
model = model.to(device)
model.eval()
file_img = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]

test_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

with torch.no_grad():
    num_data = len(file_img)
    plt.figure()
    for i in range(1,num_data+1):
        img = Image.open(file_img[i-1])
        size = torch.tensor(img.size)
        img_transformed = test_transform(img)
        img_transformed = img_transformed.unsqueeze(0)
        plt.subplot(num_data//5+1 if ((num_data % 5) != 0) else num_data//5,5,i)
        plt.imshow(T.ToTensor()(img).permute(1,2,0))
        plt.axis('off')
        pred_box = torch.sigmoid(model(img_transformed))
        # print(pred_box)
        # print(size)
        un_pred_box = unnormalize(pred_box, size)
        # print(un_pred_box)
        size = size.view(-1,2)
        un_pred_box = tune_box(un_pred_box, size)
        un_pred_box = un_pred_box.squeeze()
        plt.gca().add_patch(
        plt.Rectangle( (un_pred_box[0], un_pred_box[1]), un_pred_box[2], un_pred_box[3],
                           fill=False, edgecolor='red', linewidth=1, alpha=0.8)

        )
        
        # print(score)

    plt.show()