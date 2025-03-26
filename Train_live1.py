
from torchsummary import summary
import torch
import os
import numpy as np
from scipy import stats
import yaml
from argparse import ArgumentParser
import random
import torch.nn as nn
import random

import scipy.io as sc
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
# import h5py
import torch.utils.data as Data
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tqdm import tqdm

import logging
from testmmd import MMDLoss

from model_test import *
from model_test1 import *
from model_live1 import *
from Load_dataset_live1_noref import SIQADataset

def get_log(file_name):
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever
    formatter = logging.Formatter('[%(asctime)s]\n%(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)




def regulization(model, Lamda):
        """定义了一个计算正则化的方法"""
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lamda * torch.sum(torch.abs(w))
        return err
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_indexNum(config, index, status):
    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index = [] 
    test_index = [] 

    ref_ids = []
    for line0 in open("./data_live1/ids.txt", "r"):
        line0 = float(line0[:-1])
        ref_ids.append(line0)
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        train_index.append(i) if (i in trainindex) else \
            test_index.append(i) if (i in testindex) else \
                print("Error in splitting data")

    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index


    return len(index)


if __name__ == '__main__':
    # Training settings
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dataset", type=str, default="live1")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--lamda", type=float, default=0.00001)
    
    args = parser.parse_args()
    
    # seed = random.randint(10000000, 99999999) 
    # seed = 32707809
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # print("#==> Seed:", seed)

    seed = 1639
    seed_everything(seed)
    print('"#==> Seed:',seed)
    ##  定义日志文件！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    logger = get_log('log_live1.txt')
    with open("./config_vtl1.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        print('#==> Using GPU device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('#==> Using CPU....')

    index = []
    if args.dataset == "live1":
        print("#==> Dataset: live1")
        index = list(range(0, 365))
        random.shuffle(index)
    print('#==> Random indexes', index)

    allsrcc=[]
    allplcc=[]
    allrmse=[]
    allmad=[]
    
    for n in range(10):
        if n>=0:
            save_path = "./model_{}_5z_W1_{}.pth" .format("live1",n)
    
            ensure_dir('results2')
            save_model = "./results2/model_{}_5z_W1_{}.pth" .format("live1",n)
            save_model1 = "./results2/model_{}_5z_RMSE_{}.pth" .format("live1",n)
            model_dir = "./results2/"

            dataset = args.dataset

            testnum = get_indexNum(config, index, "test")

            train_dataset = SIQADataset(dataset, config, index, "train")
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)


            test_dataset = SIQADataset(dataset, config, index, "test")
            test_loader = torch.utils.data.DataLoader(test_dataset)


            ###model
            model = newNet4().to(device)


            Q_index = 0
            ##
            # criterion = nn.MSELoss(size_average=True).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)# , weight_decay=0.01
            criterion = nn.L1Loss() #nn.L1Loss(size_average=None, reduce=True, reduction= 'mean')
            criterion_cls =nn.CrossEntropyLoss().to(device)
            MMD_loss = MMDLoss().to(device)
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
            torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
            ###
            best_PLCC = -1
            best_SROCC = -1
            best_RMSE = 100000

            # training phase
            for epoch in range(args.epochs):
                sum_loss = 0.0
                Total = 0
                SUMLOSS = 0.0
                mos_scores = []
                predictions = []
                correct = 0
                total = 0

                model.train()
                LOSS_all = 0
                LOSS = 0

                for i, (patchesL, patchesR,(label, dis_L, dis_R)) in enumerate(tqdm(train_loader)):
                    patchesL = patchesL.to(device)
                    patchesR = patchesR.to(device)
                    label = label.to(device)
                    dis_L = dis_L.to(device)
                    dis_R = dis_R.to(device)

                    optimizer.zero_grad()
                    Total += 1
                    
                    outputs = model(patchesL,patchesR)
                    
                    # print(dis_L)
                    loss = criterion(outputs, label)
                    
                    loss.backward()
                    optimizer.step()

                   

                    LOSS = LOSS + loss.item()
                    

                train_loss = LOSS / (i + 1)
                print('#==> Stereo Quality score training loss',train_loss)
                 #每个epoch算一下训练集指标
                for x in list(outputs.cpu().detach().numpy()):
                    mos_scores.append(float(x))
                for x in list(label.cpu().detach().numpy()):
                    predictions.append(float(x))
                PLCC = pearsonr(mos_scores, predictions)[0]
                SROCC = spearmanr(mos_scores, predictions)[0]
                logger.info('==>第{}次epoch：{}训练集：PLCC: {}, SRCC: {}, LOSS: {},准确率：{}'.format(n, epoch+1, PLCC, SROCC, train_loss,(100 * torch.true_divide(correct, total))))
                model.eval()

                # test phase
                y_pred = np.zeros(testnum)
                y_pred_stereo = np.zeros(testnum)
                y_test = np.zeros(testnum)
                L, L_stereo = 0, 0    # L is for the global quality predictions and L_stereo is for the stereo quality predictions

                with torch.no_grad():
                    sum_test_loss=0
                    sum_PLCC = 0.0
                    sum_SRCC = 0.0
                    mos_scores = []
                    predictions = []
                    correct = 0
                    total = 0
                    for i, (patchesL,patchesR, (label, dis_L, dis_R)) in enumerate(test_loader):
                        patchesL = patchesL.to(device)
                        patchesR = patchesR.to(device)
                        label = label.to(device)
                        dis_L = dis_L.to(device)
                        dis_R = dis_R.to(device)

                        y_test[i] = label.item()
                        # print(label.shape)
                        

                        outputs = model(patchesL,patchesR)
                        # print(cls_l.shape)
                        score = outputs.mean()
                        # print(score)
                        y_pred[i] = score

                        loss = criterion(score, label[0])
                        L = L + loss.item()

                        
                
                        

                    test_loss = L / (i + 1)
                    SROCC = stats.spearmanr(y_pred, y_test)[0]
                    PLCC = stats.pearsonr(y_pred, y_test)[0]
                    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
                    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                    acc = torch.true_divide(correct, total)
                    #test_loss_stereo = L_stereo / (i + 1)
                    #SROCC_stereo = stats.spearmanr(y_pred_stereo, y_test)[0]
                    #PLCC_stereo = stats.pearsonr(y_pred_stereo, y_test)[0]
                    #KROCC_stereo = stats.stats.kendalltau(y_pred_stereo, y_test)[0]
                    #RMSE_stereo = np.sqrt(((y_pred_stereo - y_test) ** 2).mean())

                    print("#==>第{}次 Epoch {} Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f} acc={:.3f}".format(n,epoch,
                                                                                                                                test_loss,
                                                                                                                                SROCC,
                                                                                                                                PLCC,
                                                                                                                                KROCC,
                                                                                                                                RMSE,acc))

                    logger.info(' PLCC: {}, SRCC: {},RMSE: {}, KROCC: {},准确率：{}'.format(PLCC, SROCC,RMSE,KROCC,(100 * torch.true_divide(correct, total))))
                    if best_SROCC < SROCC :
                        best_SROCC = SROCC
                        print("#==> Update Epoch {} best valid SROCC".format(epoch))
                        # logger_best.info('第{}次，loss: {} PLCC: {}, SRCC: {},RMSE: {}, MAD: {},准确率：{}'.format(n,loss.numpy(),PLCC, SRCC,RMSE,MAD,(100 * torch.true_divide(correct, total))))
                        # torch.save(model.state_dict(), './model/datatest_vtl1_{}_{}_{:.4f}.pth'.format(seed,epoch+1, best_SROCC))
                        torch.save(model.state_dict(), save_model)
                    logger.info('平均测试集best SRCC为：{}' .format(best_SROCC))

                    if RMSE < best_RMSE :
                        print("#==> Update Epoch {} best valid RMSE".format(epoch))
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
                        torch.save(model.state_dict(), save_model1)
                        #best_PLCC = PLCC
                        best_RMSE = RMSE
                    logger.info('平均测试集best_RMSE为：{}' .format(best_RMSE))

            ########################################################################## final test ############################################
            model.load_state_dict(torch.load(save_model))
            model.eval()
            with torch.no_grad():
                sum_test_loss=0
                sum_PLCC = 0.0
                sum_SRCC = 0.0
                mos_scores = []
                predictions = []
                correct = 0
                total = 0
                y_pred = np.zeros(testnum)
                y_test = np.zeros(testnum)

                L = 0

                for i, (patchesL,patchesR, (label, dis_L, dis_R)) in enumerate(test_loader):

                    patchesL = patchesL.to(device)
                    patchesR = patchesR.to(device)
                    label = label.to(device)
                    dis_L = dis_L.to(device)
                    dis_R = dis_R.to(device)

                    y_test[i] = label.item()
                    
                    
                    #outputs = model(patchesL,patchesR)[Q_index]
                    outputs = model(patchesL,patchesR)
                    # outputs,cls_l,y_l,cls_r,y_r = model(patchesL,patchesR)
                    # print(outputs)
                    score = outputs.mean()
                    # print(score)
                    y_pred[i] = score

                    loss = criterion(score, label[0])
                    L = L + loss.item()

                    
            
    

                    
                #################################################### SROCC/PLCC/KROCC/RMSE score ####################################################
                SROCC = stats.spearmanr(y_pred, y_test)[0]
                PLCC = stats.pearsonr(y_pred, y_test)[0]
                KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                # acc = torch.true_divide(correct, total)
                # print("acc:",acc)
                # allsrcc=allsrcc+SROCC
                # allplcc=allplcc+PLCC
                # allrmse=allrmse+RMSE
                allsrcc.append(SROCC)
                allplcc.append(PLCC)
                allrmse.append(RMSE)
                if os.path.exists('total_result_10.txt'):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not
                with open('total_result_10.txt', 'a+') as f:
                    f.seek(1)
                    f.write("%s\n" % "live1 : Final test Results:seed={} loss={} SROCC={} PLCC={} KROCC={} RMSE={}".format(seed,test_loss,
                                                                                                                            SROCC,
                                                                                                                            PLCC,
                                                                                                                            KROCC,
                                                                                                                            RMSE))
                    print("live1: Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                            SROCC,
                                                                                                                            PLCC,
                                                                                                                            KROCC,
                                                                                                                            RMSE))
                    f.close() 
        random.shuffle(index)
    srcc=np.array(allsrcc)
    plcc=np.array(allplcc)
    rmse=np.array(allrmse)
    if os.path.exists('total_result_10.txt'):
                append_write = 'a' # append if already exists
    else:
                append_write = 'w' # make a new file if not
    with open('total_result_10.txt', 'a+') as f:
                f.seek(1)
                f.write("%s\n" % "live1 :averager Final test Results:seed={} SROCC={} PLCC={}  RMSE={}".format(seed,np.mean(srcc),np.mean(plcc),np.mean(rmse)))
                f.write("%s\n" % "live1 :averager Final test Results:seed={} SROCC={} PLCC={}  RMSE={}".format(seed,np.median(srcc),np.median(plcc),np.median(rmse)))
                print("live1: Final test Results: SROCC={} PLCC={}  RMSE={}".format(np.median(srcc),np.median(plcc),np.median(rmse)))
                f.close()    
