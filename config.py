''' Configuration File.
'''
# # cifar
# DATASET = "cifar10"
# CUDA_VISIBLE_DEVICES = 1
# NUM_TRAIN = 50000 # N \Fashion MNIST 60000, cifar 10/100 50000
# BATCH     = 128 # B
# MARGIN = 1.0 # xi
# WEIGHT = 1.0 # lambda
# TRIALS = 5
# CYCLES = 6
# SUBSET = 2500
# EPOCH = 200
# EPOCH_GCN = 200
# LR = 0.01 #
# LR_GCN = 1e-3
# MILESTONES =[200] #[160, 260]
# EPOCHL = 120  #20 #120 # After 120 epochs, stop
# EPOCHV = 100  # VAAL number of epochs
# MOMENTUM = 0.9
# WDECAY =0.001  #2e-3# 5e-4
# CHANNEL = 3

# #mnist
# CUDA_VISIBLE_DEVICES = 1
# DATASET= "mnist"
# NUM_TRAIN=60000 
# BATCH=128
# CYCLES = 6  
# SUBSET = 600
# LR = 0.001 #0.001
# MOMENTUM=0.9
# MARGIN = 1.0 # xi ll4al的参数
# WEIGHT = 1.0 # lambda 损失权重 1.0
# TRIALS = 5
# EPOCH = 200
# EPOCH_GCN = 200
# LR_GCN = 1e-3
# MILESTONES =[200] #[160, 260]
# EPOCHL = 120  #20 #120 # After 120 epochs, stop
# EPOCHV = 100  # VAAL number of epochs
# WDECAY =0.001  
# CHANNEL = 1


#CHANGHAI
CUDA_VISIBLE_DEVICES = 0
CUDA = 'cuda:0'

DATASET= "InES"
NUM_TRAIN=784
BATCH=8
CYCLES = 6  
SUBSET = 350
LR = 0.001 #0.001
MOMENTUM = 0.9
MARGIN = 1.0 # xi ll4al的参数
WEIGHT = 1.0 # lambda 损失权重 1.0
TRIALS = 3
EPOCH = 100 #100
EPOCH2 = 100 #100
# EARLYSTOP = 15
# EPOCH_GCN = 100
# LR_GCN = 1e-3
MILESTONES =[100] #[160, 260]
EPOCHL = 120  #20 #120 # After 120 epochs, stop
# EPOCHV = 100  # VAAL number of epochs
WDECAY = 0.001  
CHANNEL = 3
# CLASS = 3
ADD = 50
IMAGE_SIZE = 32# 
MODE = "test" #"val" 
DIS = 1
LAMBDA = 0.0001
METHOD = "CoreSet" #['ATL_Seg','Random', 'Entropy', 'CoreSet', 'lloss', 'TiDAL']