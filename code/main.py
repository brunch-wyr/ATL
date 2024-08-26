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
# from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# Custom
from model.unet import U_Net ,U_Net2 ,U_Net_lloss
from model.lossnet import LossNet , LossNet_for_TiDAL
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default=DATASET,
                    help="InES")

parser.add_argument("-m", "--method_type", type=str, default=METHOD,
                    help="['ATL_Seg','Random', 'Entropy', 'CoreSet', 'lloss', 'TiDAL']")
parser.add_argument("-c", "--cycles", type=int, default=CYCLES,
                    help="Number of active learning cycles")
# parser.add_argument("-t", "--total", type=bool, default=False,
#                     help="Training on the entire dataset")
# parser.add_argument("--seed", type=int, default=0,
#                     help="Training seed.")
parser.add_argument("--subset", type=int, default=SUBSET, # CIFAR10 
                    help="The size of subset.")
parser.add_argument("-q", "--query", type=str, default='Entropy',
                    help="The size of subset. [Entropy, AUM]")
parser.add_argument("-w", "--num_workers", type=str, default=0,
                    help="The number of workers.")
parser.add_argument("-a", "--add_num", type=int, default=ADD,
                    help="Number of add samples every cycle.")

args = parser.parse_args()
 
args.initial_size = args.add_num

# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'Entropy', 'CoreSet', 'lloss', 'ATL_Seg','TiDAL'] #, 'BALD', 'TiDAL']
    datasets = ['InES']#'mnist','cifar10', 'cifar100',]
    assert method in methods, 'No method %s! Try options %s' % (method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s' % (args.dataset, datasets)
    '''
    method_type: 'Random', 'Entropy', 'CoreSet', 'lloss', 'ATL_Seg','TiDAL'
    '''
    os.makedirs('./results', exist_ok=True)
    txt_name = f'./results/results_{args.dataset}_{str(args.method_type)}'+"_lr"+str(LR)+"_epoch"+str(EPOCH)+'.txt'
    results = open(txt_name, 'w')

    print(txt_name)
    print("Dataset: %s" % args.dataset)
    print("Method type:%s" % method)
    # if args.total:
    #     TRIALS = 3
    #     CYCLES = 6
    # else:
    CYCLES = args.cycles

    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train1, data_train2, data_unlabeled, data_test, data_valid, adden, no_train = load_dataset(args)
        print('The entire datasize is {}'.format(len(data_train1)))
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))

        # if args.total:
        #     labeled_set = indices
        # else:
        labeled_set = indices[:args.add_num]
        unlabeled_set = [x for x in indices if x not in labeled_set]

        random.shuffle(labeled_set)
        random.shuffle(unlabeled_set)

        train_loader1 = DataLoader(data_train1, batch_size=BATCH,
                                  sampler=SubsetSequentialSampler(labeled_set),
                                  pin_memory=True, num_workers=args.num_workers)
        
        train_loader2 = DataLoader(data_train2, batch_size=BATCH,
                            sampler=SubsetSequentialSampler(labeled_set),
                            pin_memory=True, num_workers=args.num_workers)

        test_loader = DataLoader(data_test, batch_size=BATCH,
                                 pin_memory=True, num_workers=args.num_workers)
        
        if MODE == "test":
            test_loader = DataLoader(data_test, batch_size=BATCH,
                                        pin_memory=True, num_workers=args.num_workers)
            dataloaders = {'train1': train_loader1, 'train2': train_loader2,'test': test_loader}

        if MODE == "valid":
            valid_loader = DataLoader(data_valid, batch_size=BATCH,
                                    pin_memory=True, num_workers=args.num_workers)
            dataloaders = {'train1': train_loader1, 'train2': train_loader2,'valid': valid_loader}
        device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')
        # Model - create new instance for every trial so that it resets
        # with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        unet_ = U_Net(in_channels=3, out_channel=2, init_features=32).to(device)
        unet2_ = U_Net2(in_channels=3, out_channel=2, init_features=32).to(device)#两个输出头
        unet_lloss = U_Net_lloss(in_channels=3, out_channel=2, init_features=32).to(device)
        
        if method == 'lloss':
            # out_dim = NO_CLASSES if method == 'TiDAL' else 1 
            out_dim = 1 
            pred_module = LossNet().to(device) #
        elif method == 'TiDAL':
            out_dim = 1 
            pred_module = LossNet_for_TiDAL().to(device) #

        if method in ['TiDAL','lloss']: #
            models = {'backbone': unet_lloss, 'module': pred_module ,'backbone2': unet2_}
        else:
            models = {'backbone': unet_ ,'backbone2': unet2_ }

        # Loss, criterion and scheduler (re)initialization
        criterion = {}
        criterion['CE'] = nn.CrossEntropyLoss()
        criterion['KL_Div'] = nn.KLDivLoss(reduction='batchmean')

        
        for key, val in models.items():
            models[key] = models[key].to(device)

        for cycle in range(CYCLES):
            # Randomly sample 10000 unlabeled data points
            # if not args.total:
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:args.subset]

            torch.backends.cudnn.benchmark = True
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum = MOMENTUM, weight_decay=WDECAY)
            
            optim_backbone2 = optim.SGD(models['backbone2'].parameters(), lr=LR,
                                       momentum = MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_backbone2 = lr_scheduler.MultiStepLR(optim_backbone2, milestones=MILESTONES)

            if method in [ 'TiDAL','lloss']: #
                optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                         momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module }
                schedulers = {'backbone': sched_backbone, 'module': sched_module}
            else:
                optimizers = {'backbone': optim_backbone,'backbone2':optim_backbone2}
                schedulers = {'backbone': sched_backbone,'backbone2':sched_backbone2}

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCH2, EPOCHL)
            #save param
            torch.save(models['backbone'].state_dict(), 'param/'+str(method)+'/Trial' + str(trial + 1) + '_Cycle' + str(cycle + 1) + 'model_checkpoint.pth')
            torch.save(models['backbone2'].state_dict(), 'param/'+str(method)+'/Trial' + str(trial + 1) + '_Cycle' + str(cycle + 1) + 'model2_checkpoint.pth')

            # torch.save(models['backbone'], 'param/'+str(method)+'/Trial' + str(trial + 1) + '_Cycle' + str(cycle + 1) + '_backbone.ckpt')
            # torch.save(models['backbone2'], 'param/'+str(method)+'/Trial' + str(trial + 1) + '_Cycle' + str(cycle + 1) + '_backbone2.ckpt')

            acc, iou = test(models, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: \n Test m_acc: {} ||m_iou: {} '.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc,iou  ))
            np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc, iou]).tofile(results, sep=" ")
            results.write("\n")
            

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args) #(350,256,2)
 
            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:args.add_num].numpy()) #(50,256,2)

            labeled_set += list(torch.tensor(subset)[arg][-args.add_num:].numpy())#取后n个将入标注集
            listd = list(torch.tensor(subset)[arg][:-args.add_num].numpy())#取（sub-n）个样本，用于加入无标注样本集
            unlabeled_set = listd + unlabeled_set[args.subset:]#加入?
            print(len(labeled_set), min(labeled_set), max(labeled_set))

            # Create a new dataloader for the updated labeled dataset

            random.shuffle(labeled_set)
            # random.shuffle(unlabeled_set)
            dataloaders['train1'] = DataLoader(data_train1, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(labeled_set),
                                              pin_memory=True, num_workers=args.num_workers)
            dataloaders['train2'] = DataLoader(data_train2, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(labeled_set),
                                              pin_memory=True, num_workers=args.num_workers)
        # np.savetxt("results/confusion"+ str(trial)+'_'+ DATASET +"_lr"+str(LR)+".csv",  delimiter=",",fmt='%.2f')

    results.close()
