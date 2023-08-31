import os
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import ndk
from ndk.quant_train_torch.feathernet import feathernet as FeatherNet
import torch.backends.cudnn as cudnn
from ndk.modelpack import modelpack

parser = argparse.ArgumentParser()
parser.add_argument('-Save_flag', type=bool, default=True, help='whether to save the model and tensorboard')
parser.add_argument('-Save_model', type=str, help='model save path.',default='/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0812_0335')
parser.add_argument('-Save_tensorboard', type=str, help='tensorboard save path.',default='/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/log')
parser.add_argument('-batch_size', type=int, help='batch size for training', default=64)
parser.add_argument('-epoch_num', type=int, help='epoch for training', default=160)
parser.add_argument('-img_size', type=int, help='the input size', default=224)
parser.add_argument('-learn_rate', type=float, help='learn decay rate', default=0.0001)
parser.add_argument('-learn_decay', type=float, help='learn decay rate', default=0.95)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('-class_num', type=int, help='class num', default=2)
parser.add_argument('-input_channels', type=int, help='the input channels', default=2)
parser.add_argument('-retrain', type=bool, help='whether to fine-turn the model', default=True)
parser.add_argument('-resume_model', type=str, help='load the model path',default='/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0811_1031/Feathernet_150.pkl')
# set the img data
"""""""""""""trian data"""""""""""""
parser.add_argument('-train_face', type=str, help='data for training',
                    default='/home/data03_disk/YZhang/irDatas/irTrain0809')
"""""""""""""test data"""""""""""""
parser.add_argument('-test_face', type=str, help='data for testing',
                    default='/home/data03_disk/YZhang/irDatas/irTest0809')
args = parser.parse_args()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Focal_loss(nn.Module):
    def __init__(self,gamm=2,eps=1e-7):
        super(Focal_loss,self).__init__()
        self.gamm=gamm
        self.eps=eps
        self.function=nn.CrossEntropyLoss(reduction='none')
    def forward(self, input,target):
        log_p=self.function(input,target)
        p=torch.exp(-log_p)
        loss=(1-p)**self.gamm*log_p
        return loss.mean()

def adjust_leanrate(optim, learn):
    for param_group in optim.param_groups:
        param_group['lr'] = learn

def soft_max(scores):
    scores=np.exp(scores)
    sum=scores[:, 0] + scores[:, 1]
    scores[:,0]/=sum
    scores[:, 1] /= sum
    return scores

def estimate(predictions,labels):
    y_true=labels
    y_pred=predictions
    Mat_result=confusion_matrix(y_true,y_pred,labels=[0,1])
    sample_num=len(labels)
    acc=np.trace(Mat_result)/sample_num
    FAR=Mat_result[0,1]*1.0/(Mat_result[0,1]+Mat_result[1,1]+0.0001)
    FRR=Mat_result[1,0]*1.0/(Mat_result[1,0]+Mat_result[0,0])
    result=np.hstack((FAR,FRR,acc))
    return result

def train(net, train_loader, optimizer, epoch,focal_loss,writer):
    acc_statis = AverageMeter()
    Loss_statis=AverageMeter()
    batch_time=AverageMeter()

    net.train()
    Time=time.time()
    for i, data in enumerate(train_loader):
        input, label = data
        input=input.double()

        input = input.cuda(device=device)
        label = label.cuda(device=device)
        logits = net(input).float()
        logits = torch.squeeze(torch.squeeze(logits, dim=-1), dim=-1)
        preditions = nn.Softmax(dim=1)(logits)
        loss = focal_loss(logits, label)
        Loss_statis.update(loss)

        _, preds_index = preditions.topk(1)
        preds_index=preds_index.view(-1, )
        preds_index=preds_index.detach().cpu()
        label = label.detach().cpu()
        result=estimate(preds_index,label)
        acc_statis.update(result[2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - Time)
        Time = time.time()
        lr = optimizer.param_groups[0]['lr']
        if i % 10 == 0:
            line = 'Epoch: [{0}][{1}/{2}]\t lr: {3:.6f}\t' \
                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss: {loss.val:.5f} ({loss.avg:.5f})\t' \
                   'Prec: {4:.5f}' \
                .format(epoch, i, len(train_loader), lr,result[2],batch_time=batch_time, loss=Loss_statis)
            print(line)
    writer.add_scalar("Training_Loss", Loss_statis.avg, epoch + 1)
    writer.add_scalar("Training_Accuracy", acc_statis.avg, epoch + 1)


def evaluateing(net,test_loader,epoch,model_dir):
   print('--------test the model--------')
   total_predictions=[]
   total_labels=[]
   net.eval()
   for i,data in enumerate(test_loader):
       input, label = data
       input=input.double()
       input=input.to(device)
       logits = net(input).float()
       logits = torch.squeeze(torch.squeeze(logits, dim=-1), dim=-1)
       preditions = nn.Softmax(dim=1)(logits)
       _, preds_index = preditions.topk(1)
       preds_index = preds_index.view(-1,)
       preds_index=preds_index.detach().cpu()
       total_predictions.extend(preds_index)
       total_labels.extend(label)
   result = estimate(total_predictions, total_labels)
   print('Epoch:{}\nFalse Rejection Rate (FRR) is {:.5f}'.format(epoch,result[1]))  # 拒认率
   print('False Acceptance Rate(FAR) is {:.5f}'.format(result[0]))  # 误识率
   print("test_acc:{:.5f}".format(result[2]))
   record = open(model_dir + '/record.txt', 'a')
   record.write('Epoch:{0}\ttest_acc:{1:.5f}\tFRR:{2:.5f}\tFAR:{3:.5f}\n'.format(epoch,result[2],result[1],result[0]))
   record.close()

def main():
    global device
    cudnn.benchmark = True


    if torch.cuda.is_available():
       device = torch.device('cuda:0')

    quant_model_path = '/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0812_0335/quant_spoof_3D'
    quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=quant_model_path,fname_npz=quant_model_path)
    # quant_net = QuantizedNet(quant_layer_list, quant_param_dict)
    quant_net = FeatherNet(quant_layer_list, quant_param_dict)
    quant_net = quant_net.to(device)
    quant_net = quant_net.double()
    print("start load train data")

    img_train = dataset.get_3d_liveness_data(args.train_face, 'train',False)
    img_test = dataset.get_3d_liveness_data(args.test_face, 'test',False)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=False)
    test_loader = DataLoader(img_test, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(params=quant_net.parameters(), lr=args.learn_rate)

    focal_loss = Focal_loss()

    writer = SummaryWriter(args.Save_tensorboard)
    start_epoch = 0
    if args.retrain:
        print('load the model ......')


    for epoch in range(start_epoch, args.epoch_num + 1):
        learn_rate = args.learn_rate * (args.learn_decay ** (epoch - 1))
        adjust_leanrate(optimizer, learn_rate)
        save_model = args.Save_model
        if epoch==0:
            evaluateing(quant_net, test_loader, epoch, save_model)
        train(quant_net, train_loader, optimizer, epoch, focal_loss,writer)
        evaluateing(quant_net, test_loader, epoch, save_model)
        after_train_quant_param_dict = quant_net.get_param_dict()
        fname = 'after_train_quant_model'
        modelpack(8, quant_layer_list, after_train_quant_param_dict,save_model + "quant_machinecode_E({})".format(str(epoch)),model_name=fname,use_machine_code=True)



if __name__ == '__main__':
    main()