import torch
import torch.nn as nn
from config import *
import numpy as np

device = torch.device( CUDA if torch.cuda.is_available() else 'cpu')

class Block(nn.Module):
   

    def __init__(self, in_channels, features):
        super(Block, self).__init__()

        self.features = features

        self.conv1 = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )
        self.bn1 =  nn.BatchNorm2d(num_features=self.features)
        self.conv2 = nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )
        self.bn2 =  nn.BatchNorm2d(num_features=self.features)
        

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1 (x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)

        return x


# 这个是主模块
class U_Net(nn.Module):

    def __init__(self,  in_channels=3, out_channel=2, init_features=32):
        super(U_Net, self).__init__()
        
        
        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)  # 代表的是第一层，初始输入通道数是3，输出是
        self.conv_encoder_2 = Block(features, features * 2) # 代笔第二层，
        self.conv_encoder_3 = Block(features * 2, features * 4)
        self.conv_encoder_4 = Block(features * 4, features * 8)

        self.bottleneck = Block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.conv_decoder_4 = Block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.conv_decoder_3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.conv_decoder_2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channel, kernel_size=1
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 256 * 256, 512 * 2 * 2)
        

    def forward(self, x):
        # 第一横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+ 最大池化
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)  
        
        # 第二横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)
   
        # 第三横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)

        # 第四横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)

        # 最后一层：只有模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’
        bottleneck = self.bottleneck(conv_encoder_4_2)
   
        # 第一层转置卷积
        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)  # 特征图拼接
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2) #(8,256,32,32)                  # 再进行和下采样一样的‘模块化’卷积操作
        
        # 第二层转置卷积
        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)#(8,128,64,64)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)#(8,64,128,128)
        
        # 最后一层转置卷积：
        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)                             # 转置卷积
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)     # 特征图拼接
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)      #(8,32,256,256)                      # 再进行和下采样一样的‘模块化’卷积操作 
        
        #两个输出头
        out1 = self.conv(conv_decoder_1_3) # 
        out2 = out1.permute(0, 2, 3, 1) #(b_s,w,h,2)

        outf = self.flatten(conv_decoder_1_3) ##(#(8,32,256,256) >>8,32*256*256) 
        outf = self.fc(outf) # 8,32*256*256 >> 8, 512 * 2 * 2
        # outf = outf.view(-1, 512, 2, 2)


        return  torch.softmax(out2,dim =-1), outf #用sigmoid



class U_Net2(nn.Module):#两个输出头

    def __init__(self, in_channels=3, out_channel=2, init_features=32):
        super(U_Net2, self).__init__()

        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)  # 代表的是第一层，初始输入通道数是3，输出是
        self.conv_encoder_2 = Block(features, features * 2) # 代笔第二层，
        self.conv_encoder_3 = Block(features * 2, features * 4)
        self.conv_encoder_4 = Block(features * 4, features * 8)

        self.bottleneck = Block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.conv_decoder_4 = Block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.conv_decoder_3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.conv_decoder_2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channel, kernel_size=1
        )

    def forward(self, x):
        # 第一横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+ 最大池化
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)  
        
        # 第二横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)
   
        # 第三横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)

        # 第四横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)

        # 最后一层：只有模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’
        bottleneck = self.bottleneck(conv_encoder_4_2)
   
        # 第一层转置卷积
        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)  # 特征图拼接
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2)                   # 再进行和下采样一样的‘模块化’卷积操作
        
        # 第二层转置卷积
        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)
        
        # 最后一层转置卷积：
        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)                             # 转置卷积
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)     # 特征图拼接
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)                            # 再进行和下采样一样的‘模块化’卷积操作 
        
   
        #两个输出头
        out1 = self.conv(conv_decoder_1_3) #(8,2,256,256)
        out2 = self.conv(conv_decoder_1_3) #(8,2,256,256)
                
        out1 = out1.permute(0, 2,3, 1) #(8,256,256,2)
        out2 = out2.permute(0, 2,3, 1)

        return torch.softmax(out1, dim=-1), torch.softmax(out2, dim=-1)       # 最终输出：维度为2，尺寸为388*388的特征图


class U_Net_lloss(nn.Module):

    def __init__(self,  in_channels=3, out_channel=2, init_features=32):
        super(U_Net_lloss, self).__init__()
        
        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)  # 代表的是第一层，初始输入通道数是3，输出是
        self.conv_encoder_2 = Block(features, features * 2) # 代笔第二层，
        self.conv_encoder_3 = Block(features * 2, features * 4)
        self.conv_encoder_4 = Block(features * 4, features * 8)

        self.bottleneck = Block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.conv_decoder_4 = Block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.conv_decoder_3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.conv_decoder_2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channel, kernel_size=1
        )

    def forward(self, x):
        # 第一横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+ 最大池化
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)  
        
        # 第二横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)
   
        # 第三横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)

        # 第四横向层：模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’+最大池化
        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)

        # 最后一层：只有模块化的'3x3卷积、BN、 ReLU、3x3卷积、BN、 ReLU’
        bottleneck = self.bottleneck(conv_encoder_4_2)
   
        # 第一层转置卷积
        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)  # 特征图拼接
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2) #(8,256,32,32)                  # 再进行和下采样一样的‘模块化’卷积操作
        
        # 第二层转置卷积
        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)#(8,128,64,64)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)#(8,64,128,128)
        
        # 最后一层转置卷积：
        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)                             # 转置卷积
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)     # 特征图拼接
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)      #(8,32,256,256)                      # 再进行和下采样一样的‘模块化’卷积操作 
        
        #两个输出头
        out1 = self.conv(conv_decoder_1_3) # 
        out1 = out1.permute(0, 2,3, 1)
     

        return torch.softmax(out1,dim =-1) , [conv_decoder_4_3, conv_decoder_3_3, conv_decoder_2_3 ,conv_decoder_1_3]
