from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import os
import torch
from utils import normalize_box

class Localize_Set(Dataset):
    def __init__(self, data_dir, id_set,transform=None):
        self.data_dir = data_dir
        self.id_set = id_set
        self.id_to_path = {}
        self.id_to_box = {}
        with open(os.path.join(data_dir, 'images.txt').replace('\\', '/')) as f:
            lines = f.readlines()
        for line in lines:
            i, path = line.split(' ')
            path = path.strip('\n')
            self.id_to_path[int(i)] = path
        with open(os.path.join(data_dir, 'bounding_boxes.txt').replace('\\', '/')) as f:
            lines = f.readlines()
#         self.boxes = torch.randn(len(lines), 5)
        self.path = [self.id_to_path[i] for i in self.id_set]
        for line in lines:
            i, box = line.split(' ', 1)
            box = box.strip()
#             print(box)
            self.id_to_box[int(i)] = torch.tensor(list(map(float, box.split())))
        self.boxes = [self.id_to_box[i] for i in self.id_set]
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform   
    
            
            
    def __len__(self):
        return len(self.id_set)
    
    def __getitem__(self,i):
        path = self.path[i]
        full_path = os.path.join(self.data_dir, 'images', path)
#         print(full_path)
        img = Image.open(full_path)
        img = img.convert('RGB')
        size = torch.tensor(img.size)
        box = self.boxes[i]
#         print(box)
        box_transformed = normalize_box(box, size)
        box_transformed = box_transformed.squeeze()
        img = self.transform(img)
        return img, box_transformed, size
        
        