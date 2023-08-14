import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from distutils.version import LooseVersion
import csv
from torchkeras import summary
from misc.utils import *
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch.nn.parallel
import Res50

class CustomDataset(Dataset):
    def __init__(self, csv_dir):
        self.data = []
        self.csv_files=[]
        for csv_file in os.listdir(csv_dir):
            df = pd.read_csv(os.path.join(csv_dir, csv_file), header=None)
            x = df.iloc[:, :2].values.astype(np.float32)
            t = df.iloc[:, 2].values.astype(np.float32)
            # n = int(np.sqrt(len(x)))
            # x = x.reshape(n, n)
            # y = y.reshape(n, n)
            # t = t.reshape(n, n)
            self.data.append((x, t))
            self.csv_files.append(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, t = self.data[index]
        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        csv_file=self.csv_files[index]
        return (x, t, csv_file)

class TemperatureAccuracy2_5(torch.nn.Module):
    def __init__(self):
        super(TemperatureAccuracy2_5, self).__init__()

    def forward(self, y_pred, y_true):
        """
        计算温度相对误差 5%
        :param y_pred: 预测值，形状为 [batch_size]
        :param y_true: 真实值，形状为 [batch_size]
        :return: 温度相对误差 5%
        """
        error = torch.abs(y_pred - y_true) / y_true
        accuracy = 1 - torch.mean(torch.where(error > 0.025, torch.ones_like(error), torch.zeros_like(error)).float())
        return accuracy
    
class TemperatureAccuracy5(torch.nn.Module):
    def __init__(self):
        super(TemperatureAccuracy5, self).__init__()

    def forward(self, y_pred, y_true):
        """
        计算温度相对误差 5%
        :param y_pred: 预测值，形状为 [batch_size]
        :param y_true: 真实值，形状为 [batch_size]
        :return: 温度相对误差 5%
        """
        error = torch.abs(y_pred - y_true) / y_true
        accuracy = 1 - torch.mean(torch.where(error > 0.05, torch.ones_like(error), torch.zeros_like(error)).float())
        return accuracy

class TemperatureAccuracy10(torch.nn.Module):
    def __init__(self):
        super(TemperatureAccuracy10, self).__init__()

    def forward(self, y_pred, y_true):
        """
        计算温度相对误差 5%
        :param y_pred: 预测值，形状为 [batch_size]
        :param y_true: 真实值，形状为 [batch_size]
        :return: 温度相对误差 5%
        """
        error = torch.abs(y_pred - y_true) / y_true
        accuracy = 1 - torch.mean(torch.where(error > 0.1, torch.ones_like(error), torch.zeros_like(error)).float())
        return accuracy

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l    
    
def SavePoints(points,savePath):
    # 写入csv文件保存
    with open(savePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points)
    print("点已保存到:",savePath)

def Save2D3D(points,saveVisPath):
    # 将输入数据划分为x, y和t数组
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # 定义网格点坐标
    xi = np.linspace(min(x), max(x), 500)
    yi = np.linspace(min(y), max(y), 500)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # 三维可视化
    fig = plt.figure()
    fig.set_size_inches(12, 5)
    # ax = Axes3D(fig)
    ax1 = fig.add_subplot(121,projection='3d')
    surf=ax1.plot_surface(xi, yi, zi, cmap = plt.get_cmap('rainbow'))
    ax1.invert_yaxis()
    fig.colorbar(surf, ax=ax1)
    # 二维可视化
    ax2 = fig.add_subplot(122)
    p=ax2.contourf(xi, yi, zi, cmap=plt.get_cmap('rainbow'))
    ax2.invert_yaxis()
    fig.colorbar(p, ax=ax2)

    plt.savefig(saveVisPath)
    # plt.show()
    plt.close()
    
    
def MkDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created at", path)
                
# 定义输入数据
parser = argparse.ArgumentParser()

#parser for training
parser.add_argument('--train', type=bool, default = True, help='train or test?')
parser.add_argument('--multi_gpu', type=bool, default = False, help='use multi gpu?')
parser.add_argument('--set_gpu', type=int, default = None, help='id=0, 1, 2...')
parser.add_argument('--batch_size', type=int, default = 128, help='batch size')

parser.add_argument('--log_dir', type=str, default = '/log', help='dir to save the train log')
parser.add_argument('--tensorboard_dir', type=str, default = '/log/tensorboard', help='dir to save the train log')
parser.add_argument('--model_dir', type=str, default = '/model', help='dir to save the trained model')

#dataset
parser.add_argument('--dataset_dir', type=str, default = '/dataset', help='dataset dir for train and test data')

#parser for evaluating
parser.add_argument('--load_model_path', type=str, default = '/model/model_best.pth', help='load the model to infer')
parser.add_argument('--result_dir', type=str, default = '/result', help='Inference result')

# 加载数据的线程数目
parser.add_argument('--workers', type=int, default = 1, help='number of data loading workers')

# 初始学习率
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--lr_rate2', default=0.5, help='learning rate decay rate')
# 权重衰减率
parser.add_argument('--decay_start', type=bool, default = False, help='decay_start when decay_margin is satisfied')
parser.add_argument('--decay_margin', default=0.85, help='margin to decay lr')
parser.add_argument('--decay_start2', type=bool, default = True, help='decay_start2 when decay_margin is satisfied')
parser.add_argument('--decay_margin2', default=0.92, help='margin2 to decay lr')#二阶段

parser.add_argument('--nepoch', type=int, default = 2000, help='max number of epochs to train') 
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')

opt = parser.parse_args()


def main():
    #测试查看模型用
    # model = Res50().to('cuda')
    # summary(model,input_shape=(2,14,14))
    
    if(opt.set_gpu is not None):
        torch.cuda.set_device(opt.set_gpu)
    # 定义模型和损失函数
    print('the number of GPU is {0}'.format(torch.cuda.device_count()))
    model = Res50.ResNet().to('cuda')
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)  # 将模型包装成并行模型
    

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    accuracy2_5 = TemperatureAccuracy2_5()
    accuracy5 = TemperatureAccuracy5()
    accuracy10 = TemperatureAccuracy10()
    writer = SummaryWriter(opt.tensorboard_dir)
    GetGraph=False
    
    
    if opt.train:
        train_dataset = CustomDataset(os.path.join(opt.dataset_dir, "train"))
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        test_dataset = CustomDataset(os.path.join(opt.dataset_dir, "test"))
        test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)
        print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}'.format(len(train_dataset), len(test_dataset)))
        
        # 进行模型训练
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
        best_test = 0
        
        if opt.start_epoch == 1:
            for log in os.listdir(opt.log_dir):
                os.remove(os.path.join(opt.log_dir, log))
        st_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.nepoch):
            # 保存开始开始迭代的log信息
            logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
            print("")
            logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
            train_count = 0
            running_loss = 0.0
            running_accu = 0.0
            train_loss = 0.0
            train_accu2_5 = 0
            train_accu = 0
            train_accu10 = 0
            
            for i, (inputs, labels, file_name) in enumerate(train_loader, 0):
                # inputs, labels = data
                #数据处理
                data_pre_torch=[]
                inputs=inputs.transpose(1, 2)
                for j in range(inputs.shape[0]):
                    channelx = inputs[j][0].reshape(1,14,14)
                    channely = inputs[j][1].reshape(1,14,14)
                    data_pre_torch.append(torch.cat([channelx, channely]))
                data_pre_torch = torch.stack(data_pre_torch, dim=0)
                
                optimizer.zero_grad()
                outputs = model(data_pre_torch.to('cuda'))#.cpu()
                outputs = torch.flatten(outputs, start_dim=1) #?
                labels = labels.to('cuda')  # 将标签移动到CUDA设备上
                
                if not GetGraph:
                    writer.add_graph(model, data_pre_torch.to('cuda'))
                    print("图保存..........")
                    GetGraph=True
                
                loss = criterion(outputs,labels)
                if torch.cuda.device_count() > 1 and opt.multi_gpu:
                    loss=loss.mean()
                loss.backward()    
                running_loss += loss.item()
                train_loss+= loss.item()
                train_count += 1
                accu=accuracy5(outputs,labels)
                train_accu += accu
                running_accu+=accu
                accu2_5=accuracy2_5(outputs,labels)
                accu10=accuracy10(outputs,labels)
                train_accu2_5 += accu2_5
                train_accu10 += accu10
                # if i % 20 == 19:    # 每20个batch打印一次训练状态
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 20))
                #     running_loss = 0.0
                # log信息存储
                print_batch=1
                if train_count % print_batch == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} loss:{3} accuracy:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_count, running_loss / print_batch, running_accu / print_batch))
                    optimizer.step()
                    running_loss = 0.0
                    running_accu = 0.0
                
            # 模型保存
            # if epoch % 10 == 9: 
            print("当前模型保存...")
            if torch.cuda.device_count() > 1 and opt.multi_gpu:
                torch.save(model.module.state_dict(), '{0}/model_current.pth'.format(opt.model_dir))
            else:
                torch.save(model.state_dict(), '{0}/model_current.pth'.format(opt.model_dir))
            
            #保存完整模型而不仅权重
            # torch.save(model, 'C:/Users/1/code_space/3Dprint/net/model/model.pt')
            print("当前模型保存完成")
            
            train_loss = train_loss / len(train_loader)  
            train_accu = train_accu / len(train_loader)    
            train_accu2_5 = train_accu2_5 / len(train_loader)        
            train_accu10 = train_accu10 / len(train_loader)    
            print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
            
            #上面记录训练过程数值+下面best模型自动保存
            logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
            logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
            test_loss = 0.0
            test_count = 0
            test_accu = 0
            test_accu2_5 = 0
            test_accu10 = 0
            ################################################## 验证模型构建##########################################
            model.eval()
            for j, (inputs, labels, file_name) in enumerate(test_loader, 0):
                # inputs, labels = data
                #数据处理
                data_pre_torch=[]
                inputs=inputs.transpose(1, 2)
                for i in range(inputs.shape[0]):
                    channelx = inputs[i][0].reshape(1,14,14)
                    channely = inputs[i][1].reshape(1,14,14)
                    data_pre_torch.append(torch.cat([channelx, channely]))
                data_pre_torch = torch.stack(data_pre_torch, dim=0)
                outputs = model(data_pre_torch.to('cuda'))#.cpu()
                outputs = torch.flatten(outputs, start_dim=1) 
                labels = labels.to('cuda')  # 将标签移动到CUDA设备上
                loss = criterion(outputs,labels)
                if torch.cuda.device_count() > 1 and opt.multi_gpu:
                    loss=loss.mean()
                test_loss += loss.item()
                accu=accuracy5(outputs,labels)
                test_accu += accu
                accu2_5=accuracy2_5(outputs,labels)
                accu10=accuracy10(outputs,labels)
                test_accu2_5 += accu2_5
                test_accu10 += accu10
                # 保存评估的log信息
                # logger.info('Test time {0} Test Frame No.{1} loss:{2} accuracy:{3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss,accu))
                logger.debug(('Test time {0} Test Frame No.{1} loss:{2} accuracy:{3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss,accu)))
                test_count += 1
                
            # 计算测试数据的平均loss accuracy
            test_loss = test_loss / test_count
            test_accu = test_accu / test_count
            test_accu2_5 = test_accu2_5 / test_count
            test_accu10 = test_accu10 / test_count
            logger.info(
                'Test time {0} Epoch {1} TEST FINISH Avg loss: {2} accuracy2.5:{3} accuracy5:{4} accuracy10:{5}'.format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_loss,test_accu2_5,test_accu,test_accu10))
            # 如果该次的测试结果，比之前最好的模型还要好，则保存目前的模型为最好的模型
            # # 依据loss指标
            # if test_loss <= best_test:
            #     best_test = test_loss
            #     print("BEST模型保存")
            #     torch.save(model.state_dict(), '{0}/model_{1}_{2}.pth'.format(opt.model_dir, epoch, test_loss))
            #     print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

            #tb record
            writer.add_scalars('loss', {'train_loss': train_loss, 'test_loss':test_loss }, epoch)
            writer.add_scalars('accuracy', {'train_accuracy5': train_accu, 
                                            'train_accuracy2.5': train_accu2_5, 
                                            'train_accuracy10': train_accu10, 
                                            'test_accuracy2.5':test_accu2_5,
                                            'test_accuracy5':test_accu,
                                            'test_accuracy10':test_accu10 
                                            }, epoch)
            
            # writer.add_scalar('loss', test_loss, epoch)
            # writer.add_scalar('accuracy', test_accu, epoch)
            
            #依据accuracy指标
            if best_test <= test_accu:
                best_test = test_accu
                if best_test>0.75:
                    print("BEST模型保存")
                    if torch.cuda.device_count() > 1 and opt.multi_gpu:
                        torch.save(model.module.state_dict(), '{0}/model_{1}_{2}.pth'.format(opt.model_dir, epoch, test_accu))
                    else:
                        torch.save(model.state_dict(), '{0}/model_{1}_{2}.pth'.format(opt.model_dir, epoch, test_accu))
                    print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
                
            # 判断模型测试的结果是否达到，学习率和权重衰减的衰减要求，达到了则进行权重和学习率的衰减
            lr_write=opt.lr
            if best_test > opt.decay_margin and not opt.decay_start:
                opt.decay_start = True
                opt.lr *= opt.lr_rate
                lr_write=opt.lr
                optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
            if best_test > opt.decay_margin2 and not opt.decay_start2:
                opt.decay_start2 = True
                opt.lr *= opt.lr_rate2
                lr_write=opt.lr
                optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
            writer.add_scalars('learning_rate', {'lr': lr_write }, epoch)
            
            #记录记录权重分布 图像数据 嵌入向量的可视化
            # for name, param in model.named_parameters():
            #     writer.add_histogram(name + '_grad', param.grad.cpu(), epoch)
            #     writer.add_histogram(name + '_data', param.cpu(), epoch)
                # print("权重 梯度保存.............")
                
    
            
        # writer.close()  
          
    else:
        
        val_dataset = CustomDataset(os.path.join(opt.dataset_dir, "test"))
        # val_dataset = CustomDataset("E:\\t_map\\extract_data\\s7test\\val")
        val_loader = DataLoader(val_dataset, batch_size=1,shuffle=False)
        print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the validating set: {0}'.format(len(val_dataset)))

        # model = Res50().to('cuda')  #这里需要重新模型结构，My_model
        model.load_state_dict(torch.load(opt.load_model_path))#这里根据模型结构，调用存储的模型参数
        # model=torch.load('C:/Users/1/code_space/3Dprint/net/model/model.pt')#这里已经不需要重构模型结构了，直接load就可以
        print("模型加载完成")
        
        #保存结果的路径
        MkDir(opt.result_dir)
        MkDir(os.path.join(opt.result_dir, "points"))
        MkDir(os.path.join(opt.result_dir, "vis"))
        st_time = time.time()
        list_accu=[]
        list_accu10=[]
        cost_time=[]
        with torch.no_grad():
            logger = setup_logger('val_result', os.path.join(opt.log_dir, 'val_log.txt'))
            logger.info('Val time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Validating started'))
            val_loss = 0.0
            val_count = 0
            val_accu = 0
            val_accu2_5 = 0
            val_accu10 = 0
            # 验证模型构建
            model.eval()
            for k, (readxy, labels, file_name) in enumerate(val_loader, 0):
                # readxy, labels = data
                #数据处理
                data_pre_torch=[]
                inputs=readxy.transpose(1, 2)
                for i in range(inputs.shape[0]):
                    channelx = inputs[i][0].reshape(1,14,14)
                    channely = inputs[i][1].reshape(1,14,14)
                    data_pre_torch.append(torch.cat([channelx, channely]))
                data_pre_torch = torch.stack(data_pre_torch, dim=0)
                #记录时间
                time1=time.time()
                outputs = model(data_pre_torch.cuda())
                time2=time.time()
                cost_time.append(time2-time1)
                
                outputs=outputs.cpu()
                outputs = torch.flatten(outputs, start_dim=1) 
                loss = criterion(outputs,labels)
                val_loss += loss.item()
                accu=accuracy5(outputs,labels)
                val_accu += accu
                accu2_5=accuracy2_5(outputs,labels)
                accu10=accuracy10(outputs,labels)
                val_accu2_5 += accu2_5
                val_accu10 += accu10
                # 保存箱线图数据
                list_accu.append(accu*100.0)
                list_accu10.append(accu10*100.0)
                # 保存评估的log信息
                logger.info(
                    'Val Frame {} loss:{} accuracy2.5:{} accuracy5:{} accuracy10:{}'.format(
                        file_name[0].split('.')[0], loss,accu2_5,accu,accu10))
                val_count += 1
                #保存result信息
                #点的格式xyt，SavePoints命名为原csv文件，保存到result/points:
                t=outputs.numpy().reshape(-1, 1)
                xy=readxy.numpy().reshape(-1, 2)
                result=np.concatenate((xy, t), axis=1)
                savePath=os.path.join(os.path.join(opt.result_dir, "points"), file_name[0])
                SavePoints(result,savePath)
                print()
                #二维和三维图也保存吧，，保存到result/vis
                saveVisName=str(file_name[0].split('.')[0])+'.png'
                saveVisPath=os.path.join(os.path.join(opt.result_dir, "vis"),saveVisName)
                Save2D3D(result,saveVisPath)
                
            # 计算测试数据的平均loss
            val_loss = val_loss / val_count
            val_accu = val_accu / val_count
            val_accu2_5 = val_accu2_5 / val_count
            val_accu10 = val_accu10 / val_count
            logger.info(
                'Val time {}  TEST FINISH Avg loss: {}, accuracy2.5: {}, accuracy5: {}, accuracy10: {}'.format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), val_loss,val_accu2_5,val_accu,val_accu10))
            
            #绘制箱线图并保存
            # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
            # plt.subplot(1, 2, 1)
            # 将accu5和accu10列表传递给boxplot函数进行绘制
            # plt.boxplot([list_accu, list_accu10],showfliers=False)
            plt.boxplot(list_accu,showfliers=False,patch_artist = True, boxprops = {'color':'orangered','facecolor':'pink'})
            plt.ylabel("Accuracy(%)")
            plt.xticks([1], ['ACCU-5'])
            # plt.xlabel("Evaluation criteria")
            # ax1.set_ylim([0.8, 1.0])
            # 创建右边的坐标轴
            # ax2.boxplot(list_accu10,showfliers=False,patch_artist = True, boxprops = {'color':'orangered','facecolor':'pink'})
            # ax2.set_ylim(99.99, 100.0)
            # ax2.set_ylabel("Accuracy(%)")
            # ax2.set_xticks([1], ['ACCU-10'])
            plt.xlabel("Evaluation criteria")
            # plt.tight_layout()  
            
            
            
            # 显示预测耗时
            # 在右下角显示最大值、最小值和均值
            mean_time=np.mean(cost_time)
            text = f"$t_p$: {round(mean_time, 4 - int(np.floor(np.log10(abs(mean_time)))) - 1)}"
            # plt.text(2, 80, text, ha='right', va='top')

            # 保存箱线图为图像文件（例如PNG格式）
            plt.savefig(os.path.join(opt.result_dir, "boxplot_accu5.png"))
            # 显示箱线图
            plt.show()
            print(text)
            print("mean=",np.mean(list_accu),"\nstd=",np.std(list_accu))
        # print('输入数据：\n', test_loader[:, :, n//2].numpy())
        # print('输出数据：\n', output_data)

    
if __name__ == '__main__':
    main()