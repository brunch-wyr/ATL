import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from config import *
from utils.metric import pixel_accuracy, intersection_over_union
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score



# def kl_div(source, target, reduction='batchmean'):
#     loss = F.kl_div(F.log_softmax(source, 1), target, reduction=reduction)
#     return loss


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def test(models, method, dataloaders, mode=MODE):
    # accurary = MulticlassAccuracy(task="multiclass", num_classes=CLASS).to(CUDA_VISIBLE_DEVICES)
    # precision = MulticlassPrecision(task="multiclass", num_classes=CLASS).to(CUDA_VISIBLE_DEVICES) #精确率
    # recall = MulticlassRecall(task="multiclass", num_classes=CLASS).to(CUDA_VISIBLE_DEVICES) #召回率
    # F1Score = MulticlassF1Score(task="multiclass", num_classes=CLASS).to(CUDA_VISIBLE_DEVICES) #F1分数
    # confusion = MulticlassConfusionMatrix(task="multiclass", num_classes=CLASS).to(CUDA_VISIBLE_DEVICES) 

    assert mode in ['val', 'test']
    models['backbone'].eval()
    models['backbone2'].eval()

    if method in ['TiDAL','lloss']:
        models['module'].eval()

    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        if method == 'TiDAL':
            for (data,targets,a,b) in dataloaders[mode]:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = data.to(CUDA_VISIBLE_DEVICES)#(8,3,256,256)
                    labels = targets.to(CUDA_VISIBLE_DEVICES)#(8,1,256,256)  #tidal (8,256,256,2)
                # if method in ['TiDAL', 'lloss']:     
                y_pred, feature = models['backbone'](inputs) #8,1,256,256   >> 8,2,256,256
                # else:
                    # y_pred = models['backbone'](inputs) #8,1,256,256   >> 8,2,256,256
                acc = pixel_accuracy(y_pred,labels)
                miou = intersection_over_union(y_pred, labels, num_classes=2) ############1""
        
        else:
            for (inputs, labels, _) in dataloaders[mode]:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.to(CUDA_VISIBLE_DEVICES)#(8,3,256,256)
                    labels = labels.to(CUDA_VISIBLE_DEVICES)#(8,1,256,256)  #tidal (8,256,256,2)
                if method in ['lloss']:     
                    y_pred, feature = models['backbone'](inputs) #8,1,256,256   >> 8,2,256,256
                else:
                    y_pred,_ = models['backbone'](inputs) #8,1,256,256   >> 8,2,256,256
                acc = pixel_accuracy(y_pred,labels)
                miou = intersection_over_union(y_pred, labels, num_classes=2)#############1"

    return acc , miou #total_acc*100, total_pre*100 , total_rec*100, total_f1*100 , total_conf


def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['backbone2'].train()
    if method in ['lloss', 'TiDAL']: #
        models['module'].train()

#########################
    if method == 'ATL_Seg':
        for (data1,data2) in zip(dataloaders['train1'],dataloaders['train2']):
        
            device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')

            #处理data
            inputs1 = data1[0].to(device)
            inputs2 = data2[0].to(device)   

            #处理label
            labels = data1[1].to(device)
            optimizers['backbone2'].zero_grad()
            # score(8,256,256,2)/  target_loss1(0.773) / target_loss2(0.773) 
            pred1,_  = models['backbone2'](inputs1)  # (8,256,256,2)
            target_loss1 = criterion['CE'](pred1, labels)   

            _,pred2 = models['backbone2'](inputs2)#
            target_loss2 = criterion['CE'](pred2, labels)    
##############
            target_loss3 = torch.norm(pred1-pred2, p=DIS, dim=(2, 3))# (8,2)
            target_loss3 = torch.mean(torch.mean(target_loss3, dim=1))
            # target_loss3 = torch.mean(torch.norm(scores1-scores2, p=DIS, dim=1),dim=0)  #一范式 ,用的Logits和cost数据级不匹配
            loss = target_loss1 + target_loss2 - LAMBDA*target_loss3 # 总损失（256,256）
            

            loss.backward()
            optimizers['backbone2'].step()
        print("stage1",loss.item())
        return loss

#########################
    else:
        for data in dataloaders['train1']:
            device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')
            # with torch.cuda.device(CUDA_VISIBLE_DEVICES):  
            inputs = data[0].to(device)  ##???存疑
            labels = data[1].to(device)

            index = data[2].detach().numpy().tolist()
            
            optimizers['backbone'].zero_grad()
            if method in ['TiDAL','lloss']: #
                optimizers['module'].zero_grad()
            if method in ['TiDAL','lloss']:
                scores ,features = models['backbone']( inputs) #####emb, features
            else:
                scores, _  = models['backbone']( inputs) #####emb, features
            target_loss   = criterion['CE'](scores, labels) # (8,256,256,2)    (8,256,256,2)
            probs = torch.softmax(scores, dim=3) #（8, 256, 256, 2）

            if method == 'TiDAL':
                moving_prob = data[3].to(CUDA_VISIBLE_DEVICES)
                moving_prob = (moving_prob * epoch + probs * 1) / (epoch + 1)
                dataloaders['train1'].dataset.moving_prob[index, :] = moving_prob.cpu().detach().numpy()

                cumulative_logit = models['module'](features) #8,256,256,2?   8,2
                m_module_loss = criterion['KL_Div'](F.log_softmax(cumulative_logit, 1), moving_prob.detach()) #    ,8,256,256,2
                m_backbone_loss = torch.sum(target_loss) #/ target_loss.size(0)
                loss = m_backbone_loss + WEIGHT * m_module_loss

            if method == 'lloss':
                if epoch > epoch_loss:
                    features[0] = features[0].detach()   #(8,256,32,32)   
                    features[1] = features[1].detach()#(8,128,64,64)
                    features[2] = features[2].detach()#(8,64,128,128)
                    features[3] = features[3].detach() #(8,32,256,256)   

                pred_loss = models['module'](features) #
                pred_loss = pred_loss.view(pred_loss.size(0))
                cri = nn.CrossEntropyLoss(reduction="none")
                target_loss   = cri(scores, labels) 
                target_loss_reduc = torch.sum(target_loss,dim = (1,2)) 
                m_module_loss = LossPredLoss(pred_loss, target_loss_reduc, margin=MARGIN)

                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                loss = m_backbone_loss + WEIGHT * m_module_loss



            else:
                m_backbone_loss = target_loss #torch.sum(target_loss) / target_loss.size(0)  ####
                loss = m_backbone_loss

            loss.backward()
            optimizers['backbone'].step()
            if method in [ 'lloss']:
                optimizers['module'].step()
        return loss

def train_epoch2(models,  criterion, optimizers, dataloaders, epoch ):
    models['backbone'].train()
    models['backbone2'].train()
    for data1 in dataloaders['train1']:

        device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')

        #处理data
        inputs1 = data1[0].to(device)

        #处理label
        labels = data1[1].to(device)
        # labels = labels.squeeze(1) 
        # mask = labels > 0.5  
        # labels[mask] = 1  
        # mask1 = labels <= 0.5 
        # labels[mask1] = 0 
        # labels= labels.long()

        optimizers['backbone'].zero_grad()

        scores1, _ = models['backbone'](inputs1) #####emb, features
        target_loss1 = criterion['CE'](scores1, labels) # (8,2,256,256)    (8,256,256)
        loss = target_loss1  # 总损失
        loss.backward()
        optimizers['backbone'].step()
        # print("stage2",loss.item())
    print("stage2",loss.item())
    return loss



def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, num_epochs2, epoch_loss):
    print('>> Train a Model.')
    
    best_acc, best_epoch, n_stop = 0., 0, 0

    if method == "ATL_Seg":

        #stage 1 
        for epoch in range(num_epochs):
            train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)
            schedulers['backbone2'].step() #两个输出头
            
        #stage1 model parameters give to stage2 model
        param = models['backbone2'].state_dict()
        models['backbone'].load_state_dict(param, strict=False)
        
        #stage 2
        for epoch in range(num_epochs2):
            best_loss = torch.tensor([0.5]).to(CUDA_VISIBLE_DEVICES)
            train_epoch2(models,  criterion, optimizers, dataloaders, epoch )
            schedulers['backbone'].step()

        print('>> Finished.')


    else:    
        for epoch in range(num_epochs):

            best_loss = torch.tensor([0.5]).to(CUDA_VISIBLE_DEVICES)
            loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)
            
            schedulers['backbone'].step()
            if method in ['TiDAL','lloss']:
                schedulers['module'].step()

            # if True and epoch % 20 == 7:
            #     acc, iou  = test(models, method, dataloaders, mode='test')
            #     if best_acc < acc:
            #         best_acc = acc
            #         print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
            # # acc , iou = test(models, method, dataloaders, mode='test')
            # if acc is not None:
            #         if acc > best_acc:
            #             best_acc = acc
            #             # if True and epoch % 20 == 7:
            #             #     print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
            #             best_epoch = epoch
            #             n_stop = 0
            #         else:
            #             n_stop += 1
                    
            #         if n_stop > EARLYSTOP:
            #             print('Early stopping at epoch %d ' % epoch)
            #             break
            # print(' Best Acc: {:.3f}'.format(best_acc))
        
        print('>> Finished.')
