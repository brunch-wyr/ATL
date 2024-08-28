# import torch
# import torchvision.transforms as T
# from torch.utils.data import Dataset
# from torchvision.datasets import CIFAR100, CIFAR10 , MNIST

# from config import *
# # from utils.make_imbalance import *
# import glob
# from PIL import Image


# # ####
# """读取数据"""
# all_imgs_path = glob.glob("changhai/*/*.jpg")  
# species = ['normal','polyp','ulceration']
# species_to_id = dict((c, i) for i, c in enumerate(species))
# id_to_species = dict((v, k) for k, v in species_to_id.items())
# all_labels = []

# for img in all_imgs_path:
#     for i, c in enumerate(species):
#         if c in img:
#             all_labels.append(i)

# index = np.random.permutation(len(all_imgs_path))
# all_imgs_path = np.array(all_imgs_path)[index]
# all_labels = np.array(all_labels)[index]

# #80% as train
# r = int(len(all_imgs_path))
# s = int(len(all_imgs_path)*0.8)
# print(r,s)

# train_imgs_path = all_imgs_path[:s]
# train_labels = all_labels[:s]
# test_imgs_path = all_imgs_path[s:]
# test_labels = all_labels[s:]

# ###


# def read_imgs(imgs_path):
#     transform = T.Compose([
#         # T.ToTensor(),
#         T.ToTensor(),
#         T.Resize([256,256]),
#         T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛

#                            ])
#     all_images = []
#     for image_file in imgs_path:
#         img = Image.open(image_file)
#         img_tensor = transform(img)
#         all_images.append(img_tensor)
#     all_images = np.array(all_images)
#     # all_images = torch.stack(all_images)
#     return all_images

# train_data = read_imgs(train_imgs_path)
# test_data = read_imgs(test_imgs_path) 
# ##
# ############################### 
# class Mydatasetpro(Dataset):
#     def __init__(self, data, targets, transform):
#         self.data = data
#         self.targets = targets
#         self.transforms = transform
#     def __getitem__(self, index):                #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor 
#         data = self.data[index] #3
#         target = self.targets[index]              
#         # data = self.transforms(data) #trans 500  transform的问题
#         return data, target
#     def __len__(self):
#         return len(self.data)
# ###############################

# class MyDataset(Dataset):
#     def __init__(self, dataset_name, train_flag=None, transf=None, args=None):
#         self.dataset_name = dataset_name
#         self.args = args

#         # if args is not None:
#             # random.seed(args.seed)
#             # np.random.seed(args.seed)
#             # torch.manual_seed(args.seed)
#             # torch.backends.cudnn.deterministic = True

#         if self.dataset_name == "cifar10":
#             self.dataset = CIFAR10('../cifar10', train=train_flag,
#                                    download=True, transform=transf) #tensor
#         elif self.dataset_name == "cifar100":
#             self.dataset = CIFAR100('../cifar100', train=train_flag,
#                                     download=True, transform=transf) #tensor
#         elif self.dataset_name == "mnist":
#             self.dataset = MNIST('../mnist', train=train_flag,
#                                     download=True, transform=transf) #array
#         # elif self.dataset_name == "changhai":
#         #     self.dataset = Mydatasetpro(train_data, train_labels, transf)   

#         if args is not None:
#             targets = np.array(self.dataset.targets)
#             classes, class_counts = np.unique(targets, return_counts=True)
#             nb_classes = len(classes)
#             self.moving_prob = np.zeros((len(self.dataset), nb_classes), dtype=np.float32)

#             if args.init_dist == 'uniform':
#                 init_num_per_cls = int(args.initial_size / nb_classes)
#             elif args.init_dist == 'random':
#                 init_num_per_cls = 1

#             initial_idx = []
#             for j in range(nb_classes):
#                 jth_cls_idx = [i for i, label in enumerate(targets) if label == j]
#                 random.shuffle(jth_cls_idx)
#                 initial_idx += jth_cls_idx[:init_num_per_cls]

#             dataset_idx = [i for i in range(len(targets))]
#             unlabel_idx = [item for item in dataset_idx if item not in initial_idx]

#             temp_data = self.dataset.data
#             temp_targets = self.dataset.targets

#             self.dataset.data = temp_data[initial_idx] # tensor 而不是array
#             self.dataset.targets = list(np.array(temp_targets)[initial_idx])
#             self.dataset.unlabeled_data = temp_data[unlabel_idx]
#             self.dataset.unlabeled_targets = list(np.array(temp_targets)[unlabel_idx])

#             self.img_num_per_cls = get_img_num_per_cls_unif(self.dataset, cls_num=nb_classes,
#                                                             imb_factor=args.imb_factor)
#             self.dataset = gen_imbalanced_data_unif(self.dataset, self.img_num_per_cls)

#     def __getitem__(self, index):
#         if self.args is not None:
#             data, target = self.dataset[index] # array 3,32,32
#             moving_prob = self.moving_prob[index]
#             return data, target, index, moving_prob
#         else:
#             data, target = self.dataset[index]
#             return data, target, index

#     def __len__(self):
#         return len(self.dataset)


# def sync_dataset(labeled_set, unlabeled_set):
#     unlabeled_set.dataset.data = labeled_set.dataset.data
#     unlabeled_set.dataset.targets = labeled_set.dataset.targets
#     return unlabeled_set


# # Data
# def load_dataset(args):
#     dataset = args.dataset

#     if dataset == 'InES':
#         train_transform = T.Compose([
#         T.ToTensor(),
#         T.Resize([256,256]),
#         T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
#     ])

#         test_transform = T.Compose([
#         T.ToTensor(),
#         T.Resize([256,256]),
#         T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
#         ])    

#         data_train = MyDataset(dataset, train_flag=None, transf=train_transform, args=args) 
#         data_unlabeled = MyDataset(dataset, train_flag=None, transf=test_transform) #array
#         data_unlabeled = sync_dataset(data_train, data_unlabeled)
#         data_test = Mydatasetpro(test_data, test_labels, test_transform) #TestSet TensorData
#         # data_test.pred_main = np.zeros((len(data_test), EPOCH, 3), dtype=np.float32)####
#         # data_test.pred_sub = np.zeros((len(data_test), EPOCH, 3), dtype=np.float32)####
#         NO_CLASSES = 3
#         NUM_TRAIN = len(data_train)
#         no_train = NUM_TRAIN


#     adden = args.add_num
#     return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train

# #################################################################################################

import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image 
from config import * 
import torchvision.transforms as T


class SegDataset(Dataset):
    
    def __init__( self , method, images_dir, transform_sample ):   #(method) 输入的是图片所在的路径文件夹 data/InES/test
        
        # 始终要想这个2个事情：图片+标签
        image_root_path = images_dir + '/pic'# 原图所在的位置
        mask_root_path = images_dir + '/mask'  # Mask所在的位置
        
        # 将图片与Mask读入后，分别存在image_slices与mask_slices中
        self.image_slices = []
        self.mask_slices = []
        
        transform_target = T.Compose([
        T.ToTensor(),
        T.Resize([256,256]),
        # T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
        ])    

        
        for im_name in os.listdir(image_root_path):
            
            mask_name = im_name.split('.')[0] + '.png'        # 根据图片名称，定位匹配下标签名称，重点是方便下面对应 

            image_path = image_root_path + "/"+ im_name   # 图片完整路径
            mask_path = mask_root_path + "/" + mask_name   # 对应的标签完整路径

            im = np.asarray(transform_sample(Image.open(image_path)))   #
            mask = np.asarray(transform_target(Image.open(mask_path))) #(32,32) >> (2,32,32)
            
            # 创建形状为 (256, 256) 的单通道二值图像
            binary_image = (mask == 0).astype(np.float32)
            # 将二值图像转换为形状为 (256, 256, 2) 的双通道张量
            #背景是（1,0） 前景是（0,1）
            output_tensor = np.zeros((256, 256, 2), dtype=np.float32)
            output_tensor[:, :, 0] = binary_image
            output_tensor[:, :, 1] = 1 - binary_image
            # mask_output_tensor = torch.from_numpy(output_tensor)

            self.image_slices.append(im )#/ 255.) #灰度归一化
            self.mask_slices.append(output_tensor)

        # if method in ['TiDAL']:
        #     targets = np.array(self.dataset.targets)
        #     classes, class_counts = np.unique(targets, return_counts=True)
        #     nb_classes = len(classes)
        #     self.moving_prob = np.zeros((len(self.dataset), nb_classes), dtype=np.float32)
        
        
        # self.moving_prob的形状：即为网络输出形状————[batchsize, W,H,C]即[8,256,256,2]       
        # 自此在初始中，就把图片及对应的标签就准备好了

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        
        #  始终要想这个2个事情：图片+标签

        image = self.image_slices[idx] 
        mask = self.mask_slices[idx] 

        # image = image.transpose(2, 0, 1) #(32,32,3)>(3,32,32)
        # mask = mask[np.newaxis, :, :]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        
        return image, mask , idx 



# Data
def load_dataset(args):
    dataset = args.dataset
    method = args.method_type
    "train"
    train_path = 'data/InES/train'
    # train_mask_path = 'data/InES/train/mask'  
    "test"
    test_path = 'data/InES/test'
    # test_mask_path = 'data/InES/test/mask' 
    "valid"
    valid_path = 'data/InES/valid'
    # valid_mask_path = 'data/InES/valid/mask' 
    
    if dataset == 'InES':
        train_transform1 = T.Compose([
        T.ToTensor(),
        T.Resize([256,256]),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
        ])
        
        train_transform2 = T.Compose([
        # T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5, hue=0.3),
        T.ToTensor(),
        T.Resize([256,256]),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
        ])

        test_transform = T.Compose([
        T.ToTensor(),
        T.Resize([256,256]),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
        ])    
        
        valid_transform = test_transform
        
        transform_target = T.Compose([
        T.ToTensor(),
        T.Resize([256,256]),
        # T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
        ])    

    if method == "TiDAL":
        data_train1 = SegDataset_for_TiDAL(dataset, train_flag=True, transf=train_transform1, args=args) 
        data_train2 = SegDataset_for_TiDAL(dataset, train_flag=True, transf=train_transform2, args=args) 
        data_unlabeled = SegDataset_for_TiDAL(dataset, train_flag=True, transf=train_transform1, args=args) 
        data_test = SegDataset_for_TiDAL(dataset, train_flag=True, transf=test_transform, args=args)
        data_valid = SegDataset_for_TiDAL(dataset, train_flag=True, transf=valid_transform, args=args)

    else:

        data_train1 = SegDataset(method, train_path,  train_transform1) 
        data_train2 = SegDataset(method, train_path,  train_transform2) 
        data_unlabeled = SegDataset(method, train_path,  train_transform1) 
        data_test = SegDataset(method, test_path,  test_transform)
        data_valid = SegDataset(method, valid_path,  valid_transform)
        # data_test.pred_main = np.zeros((len(data_test), EPOCH, 3), dtype=np.float32)####
        # data_test.pred_sub = np.zeros((len(data_test), EPOCH, 3), dtype=np.float32)####
        # NO_CLASSES = 3

    NUM_TRAIN = len(data_train1)
    no_train = NUM_TRAIN


    adden = args.add_num
    return  data_train1, data_train2, data_unlabeled, data_test, data_valid, adden, no_train


class SegDataset_for_TiDAL(Dataset):
    def __init__(self, dataset_name, train_flag=None, transf=None, args=None):
        self.dataset_name = dataset_name
        self.args = args
        method = args.method_type
        train_path = 'data/InES/train'
        train_transform1 =  T.Compose([
        T.ToTensor(),
        T.Resize([256,256]),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
    ])
        self.dataset = SegDataset(method,  train_path,  train_transform1)  
        
        if args is not None:
            # targets = torch.tensor(self.dataset.mask_slices) #np.array(self.dataset.mask_slices) #748,256,256,2

# Assuming self.dataset.mask_slices is a list of NumPy arrays
            mask_slices_array = np.array(self.dataset.mask_slices)
            targets = torch.tensor(mask_slices_array)

            # num, weight , height ,class_counts = np.unique(targets, return_counts=True)
            targets_size = targets.size()
            num, weight , height ,num_class =targets_size[0],targets_size[1],targets_size[2],targets_size[3]
            # nb_classes = len(num)
            self.moving_prob = np.zeros(( num, weight , height ,num_class), dtype=np.float32)
            # self.moving_prob = np.zeros(( num, num_class), dtype=np.float32)
            #[batchsize, W,H,C]即[8,256,256,2]
            # ######################
            # if args.init_dist == 'uniform':
            #     init_num_per_cls = int(args.initial_size / num_class)
            # elif args.init_dist == 'random':
            #     init_num_per_cls = 1

            # initial_idx = []
            # for j in range(num_class):
            #     jth_cls_idx = [i for i, label in enumerate(targets) if label == j]
            #     np.random.shuffle(jth_cls_idx)
            #     initial_idx += jth_cls_idx[:init_num_per_cls]

            # dataset_idx = [i for i in range(len(targets))]
            # unlabel_idx = [item for item in dataset_idx if item not in initial_idx]

            # temp_data = self.dataset.image_slices
            # temp_targets = self.dataset.mask_slices

            # self.dataset.image_slices = temp_data[initial_idx] # tensor 而不是array
            # self.dataset.mask_slices = list(np.array(temp_targets)[initial_idx])
            # # self.dataset.unlabeled_data = temp_data[unlabel_idx]
            # # self.dataset.unlabeled_targets = list(np.array(temp_targets)[unlabel_idx])

            # # self.img_num_per_cls = get_img_num_per_cls_unif(self.dataset, cls_num= num_class,
            # #                                                 imb_factor=args.imb_factor)
            # # self.dataset = gen_imbalanced_data_unif(self.dataset, self.img_num_per_cls)
            # #########################

    def __getitem__(self, index):
        if self.args is not None:
            data  = self.dataset.image_slices[index]
            target = self.dataset.mask_slices[index] # array 3,256,256
            moving_prob = self.moving_prob[index]
            return data, target, index, moving_prob
        else:
            data, target = self.dataset[index]
            return data, target, index

    def __len__(self):
        return len(self.dataset)





###########
# #测试
# dataset_train = SegDataset(images_dir = "data/InES/test", image_size = 32)
# loader_train = DataLoader( dataset_train, batch_size=32, shuffle=True, num_workers=4 )
    