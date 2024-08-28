import torch
import torch.nn as nn
import torch.nn.functional as F



def pixel_accuracy(predicted, target):
    """
    计算像素准确率
    Args:
        predicted (torch.Tensor): 预测值张量，形状为(N, H, W, C)
        target (torch.Tensor): 标签值张量，形状为(N, H, W, C)，通常是独热编码形式

    Returns:
        float: 像素准确率
    """
    # 获取预测值和标签值中每个像素的类别
    _, predicted_labels = torch.max(predicted, dim=3)
    _, target_labels = torch.max(target, dim=3)

    # 计算正确分类的像素数
    correct_pixels = torch.sum(predicted_labels == target_labels).item()

    # 计算总像素数
    total_pixels = target_labels.numel()

    # 计算像素准确率
    accuracy = correct_pixels / total_pixels

    return accuracy


def intersection_over_union(predicted, target, num_classes):
    """
    计算 Intersection over Union (IoU)
    Args:
        predicted (torch.Tensor): 预测值张量，形状为(N, H, W, C)(8,256,256,2)
        target (torch.Tensor): 标签值张量，形状为(N, H, W, C)(8,256,256,2)
        num_classes (int): 类别数，不包括背景类

    Returns:
        float: 平均 IoU
    """
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for cls in range(1, num_classes):  # 从1开始，排除背景类
        pred_cls = predicted[..., cls]  # 取出特定类别的预测概率, 取出第二个通道
        target_cls = (target[..., cls] == 1).float()  # 将标签值转换为二元张量


        intersection[cls] = torch.sum(pred_cls * target_cls)
        union[cls] = torch.sum(pred_cls) + torch.sum(target_cls) - intersection[cls]

    iou = torch.mean(intersection[1:] / union[1:])  # 计算非背景类别的平均 IoU
    return iou.item()


# def mean_pixel_accuarcy(input, target):
#     """
#     input: torch.FloatTensor:(N, C, H, W)
#     target: torch.LongTensor:(N, H, W)
#     return: Tensor
#     """
#     N, num_classes, H, W = input.size()
#     input = F.softmax(input, dim=1)
#     arg_max = torch.argmax(input, dim=1)
#     confuse_matrix = confusion_matrix(arg_max, target, num_classes)
#     result = 0
#     for i in range(num_classes):
#         result += (confuse_matrix[i, i] / torch.sum(confuse_matrix[i, :]))

#     return result / num_classes



# def mean_iou(input, target):
#     """
#     input: torch.FloatTensor:(N, C, H, W)  #8,2,256,256   应该是c=2
#     target: torch.LongTensor:(N, H, W)   #8,256,256
#     return: Tensor
#     """
#     # target = target.mean(dim=1)  
#     # assert len(input.size()) == 4
#     # assert len(target.size()) == 3
#     N, H, W , num_classes= input.size()

#     # mask = labels > 0.5  
#     # labels[mask] = 1  
#     # mask1 = labels <= 0.5 
#     # labels[mask1] = 0 
#     # labels= labels.long() 
    
#     input = F.softmax(input, dim=1)
#     arg_max = torch.argmax(input, dim=1)
#     result = 0
#     confuse_matrix = confusion_matrix(arg_max, target, num_classes) ####
#     for i in range(num_classes):
#         nii = confuse_matrix[i, i]
#         # consider the case where the denominator is zero.
#         if nii == 0:
#             continue
#         else:
#             ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
#             result += (nii / (ti + tj - nii))

#     return result / num_classes




# def frequency_weighted_iou(input, target):
#     """
#     input: torch.FloatTensor:(N, C, H, W)
#     target: torch.LongTensor:(N, H, W)
#     return: Tensor
#     """
#     assert len(input.size()) == 4
#     assert len(target.size()) == 3
#     N, num_classes, H, W = input.size()
#     input = F.softmax(input, dim=1)
#     arg_max = torch.argmax(input, dim=1)
#     # get confusion matrix
#     result = 0
#     confuse_matrix = confusion_matrix(arg_max, target, num_classes)
#     for i in range(num_classes):
#         nii = confuse_matrix[i, i]
#         # consider the case where the denominator is zero.
#         if nii == 0:
#             continue
#         else:
#             ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
#             result += (ti * nii / (ti + tj - nii))

#     return result / torch.sum(confuse_matrix)