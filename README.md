# Object Localization
This repo train a deeplearning model to perform object localization to produce a bounding that specifies a object in the image
## 1.Dataset
We use animal dataset.Dataset consists 200 classes with about 60 images each class.

Dataset directory contain folders corresponding to each class of dataset. We split the dataset into training set(ratio=0.8) and validation set(ratio=0.2) 

## 2.Select model
We fine-tune pre-trained model from ImageNet to perform this task. Here you can use some common backbone model:
- Resnet18
- EfficientNet-B0
We modify the last fully-connected layer to preduce 4 output (x-y-w-h of bouding box)

## 3.Train
Train model on your dataset by specifying image folder, model type, epoch, batch size and learning rate. We train model with Adam optimizer and StepLR scheduler
```bash
python train.py src/data efficientnet --epoch 40 --batch_size 64 --learning_rate 1e-3
```

## 4.Detect
Detect image folder using a model that you have trained by specifying image folder, model type and weight 
```bash
python detect.py new-data efficientnet efficient.pth 
```

## 5.Result 
I train model restnet and efficientnet on this animal dataset. Resnet has accuracy 87% while EfficientNet has 90% on validation set
### Resnet
![file](https://github.com/thienlk2000/Object-Localization/blob/main/imgs/loss_ef.JPG)
![file](https://github.com/thienlk2000/Object-Localization/blob/main/imgs/acc_res.JPG)
![file](https://github.com/thienlk2000/Object-Localization/blob/main/imgs/resnet.png)



### EfficientNet
![file](https://github.com/thienlk2000/Object-Localization/blob/main/imgs/loss_train.JPG)
![file](https://github.com/thienlk2000/Object-Localization/blob/main/imgs/acc_ef.JPG)
![file](https://github.com/thienlk2000/Object-Localization/blob/main/imgs/efficient.png)

The EfficientNet perform better when detect cat images
