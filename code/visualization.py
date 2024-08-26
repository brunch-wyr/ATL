import cv2
import numpy as np
# Python
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.data.sampler import SubsetRandomSampler
from  utils.smapler import SubsetSequentialSampler 
import argparse
import os
import torchvision.transforms as T

# from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# Custom
from model.unet import U_Net ,U_Net2 ,U_Net_lloss
from model.lossnet import LossNet 
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *
from PIL import Image 

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default=DATASET,
                    help="InES")

parser.add_argument("-m", "--method_type", type=str, default="ATL_Seg",
                    help="['ATL_Seg','Random', 'Entropy', 'CoreSet', 'lloss' ]")
parser.add_argument("-c", "--cycles", type=int, default=CYCLES,
                    help="Number of active learning cycles")

parser.add_argument("--subset", type=int, default=SUBSET, # CIFAR10 
                    help="The size of subset.")

parser.add_argument("-w", "--num_workers", type=str, default=0,
                    help="The number of workers.")
parser.add_argument("-a", "--add_num", type=int, default=ADD,
                    help="Number of add samples every cycle.")

args = parser.parse_args()

def train():
    data_train1, data_train2, data_unlabeled, data_test, data_valid, adden, no_train = load_dataset(args)
        # device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')

    print('The entire datasize is {}'.format(len(data_train1))) 
    NUM_TRAIN = no_train
    indices = list(range(NUM_TRAIN))
    labeled_set = indices[:ADD]
    # labeled_set =  self.label_set
    unlabeled_set = [x for x in indices if x not in labeled_set]
    random.shuffle(labeled_set)
    random.shuffle(unlabeled_set)    

    train_loader1 = DataLoader(data_train1, batch_size=BATCH,
                                sampler=SubsetSequentialSampler(labeled_set),
                                pin_memory=True)
    
    train_loader2 = DataLoader(data_train2, batch_size=BATCH,
                        sampler=SubsetSequentialSampler(labeled_set),
                        pin_memory=True)

    test_loader = DataLoader(data_test, batch_size=BATCH,
                                pin_memory=True)
    
    dataloaders = {'train1': train_loader1, 'train2': train_loader2,'test': test_loader}
    device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')
    unet_ = U_Net(in_channels=3, out_channel=2, init_features=32).to(device)
    unet2_ = U_Net2(in_channels=3, out_channel=2, init_features=32).to(device)
    models = {'backbone': unet_ ,'backbone2': unet2_ }   
    criterion = {}
    criterion['CE'] = nn.CrossEntropyLoss()

    for cycle in range(CYCLES):
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]

        optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                    momentum = MOMENTUM, weight_decay=WDECAY)
        
        optim_backbone2 = optim.SGD(models['backbone2'].parameters(), lr=LR,
                                    momentum = MOMENTUM, weight_decay=WDECAY)

        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
        sched_backbone2 = lr_scheduler.MultiStepLR(optim_backbone2, milestones=MILESTONES)
        optimizers = {'backbone': optim_backbone,'backbone2':optim_backbone2}
        schedulers = {'backbone': sched_backbone,'backbone2':sched_backbone2}
   
        pred_y = models['backbone'].load_state_dict(torch.load('param/ATL_Seg/Trial1_Cycle1model_checkpoint.pth'))


    return pred_y








# 加载测试图像
input_image = 'data/InES/test/pic/HD_130.png'
device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')
# 将图像转换为PyTorch张量，并进行归一化处理
test_transform = T.Compose([
                T.ToTensor(),
                T.Resize([256,256]),
                T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])#图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
                ])    

input_tensor = test_transform(Image.open(input_image)).to(device)
# input_tensor = test_transform(input_image )
# input_tensor = input_tensor.permute(2, 0, 1).to(device)  # 调整张量维度顺序为(C, H, W)
input_tensor = input_tensor.unsqueeze(0)  # 添加一个维度作为batch

# 使用你的模型进行推理，这里假设你的模型命名为unet，并且已经加载了预训练权重

unet_ = U_Net(in_channels=3, out_channel=2, init_features=32).to(device)
unet2_ = U_Net2(in_channels=3, out_channel=2, init_features=32).to(device)
models = {'backbone': unet_ ,'backbone2': unet2_ }   
models['backbone'].load_state_dict(torch.load('param/ATL_Seg/Trial1_Cycle3model_checkpoint.pth'))
# checkpoint_path = 'param/ATL_Seg/Trial1_Cycle6_backbone.ckpt'
# checkpoint = torch.load(checkpoint_path)
# models['backbone'].load_state_dict(checkpoint['state_dict'])
# models['backbone'].load_from_checkpoint('param/ATL_Seg/Trial1_Cycle6_backbone.ckpt')
with torch.no_grad():
    output_tensor = models['backbone'](input_tensor) #(1,256,256,2)

# 取出分割结果
segmented_image = output_tensor[0, :, :, 1].squeeze().cpu().numpy() #(256,256)

# 设置分割区域的颜色
segment_color = (0, 255, 0)  # 这里选择绿色

# 创建一张空白图像，与输入图像大小相同
output_image = np.zeros((256, 256, 3), dtype=np.uint8)

# 在空白图像上根据分割结果绘制分割区域
output_image[np.where(segmented_image == 1)] = segment_color  # 假设类别为1

# 显示输出图像
# cv2.imshow('Segmentation Result', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存输出图像

cv2.imwrite('input_image.jpg', cv2.imread(input_image))
cv2.imwrite('output_image.jpg', output_image)
print("Image saved successfully!")

