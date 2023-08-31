import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
# from pytorch import FeatherNet
import FeatherNet
from skimage import io

from sklearn.metrics import confusion_matrix
import cv2
from torch.utils import data
from torchvision import transforms as trans
def load_all_txts(txt_dir):
    files=os.listdir(txt_dir)
    txts=list(filter(lambda x : x.endswith(('txt')),files))
    txts=[os.path.join(txt_dir,txt) for txt in txts]
    txts.sort()
    total_imgs_inf=[]
    for txt in txts:
        img_paths=load_data(txt)
        total_imgs_inf.extend(img_paths)
    return total_imgs_inf
def load_data(txt):
    imgs_inf = []
    f = open(txt, 'r')
    paths = f.readlines()
    paths.sort()
    for path in paths:
        path = path.strip()
        if path != '':
            imgs_inf.append(path)
    return imgs_inf

def decode_img_data(imgs_inf):
    img_paths=[]
    labels=[]
    for img_inf in imgs_inf:
        img_paths.append(img_inf.split(' ')[0])
        labels.append(int(img_inf.split(' ')[1]))
    data=np.transpose(np.vstack((img_paths,labels)))
    return data

def statistical_TF_sample(image_train):
    total = len(image_train)
    positive = 0
    negative = 0
    Print=0
    Video=0
    false_key = ['spoof', 'false', 'print','video']
    true_key = ['true', 'live', 'real']
    for img_path in image_train:
        img_save_dir=os.path.split(img_path)[0]
        if any(key in img_save_dir for key in false_key):
            negative += 1
            if 'print' in img_save_dir:
                Print+=1
            elif 'video' in img_save_dir:
                Video+=1
            else:
                print(img_path)
        elif any(key in img_save_dir for key in true_key):
            positive += 1
        else:
            print(img_path)
    return total,positive,negative,Print,Video

def make_data(scence_path,face_path):
    scence_data=load_all_txts(scence_path)
    face_data=np.array(load_all_txts(face_path)).reshape(-1,1)
    scence_data=decode_img_data(scence_data)
    if len(scence_data)!=len(face_data):
        print("scence img don't equal face img")
        return None
    data=np.hstack((scence_data[:,0].reshape(-1,1),face_data,scence_data[:,1].reshape(-1,1)))
    return data

def nomalization(img):
    img = np.array(img)
    img = (img - np.mean(img)) / np.std(img)
    return img

class get_data(data.Dataset):
    def __init__(self,scence_path,phase,transform):
        self.data=decode_img_data(load_all_txts(scence_path))
        self.phase=phase
        if self.phase=='train':
           print('train imgs has:{}'.format(len(self.data)))
        else:
           print('test imgs has:{}'.format(len(self.data)))
        self.total,self.positive,self.negative,self.Print,self.Video=statistical_TF_sample(self.data[:,0])
        print('imgs:%d --> [positive:%d\tnegative:%d]' % (self.total, self.positive, self.negative))
        print('In negative print:%d,video:%d' % (self.Print,self.Video))
        self.transform=transform
    def __getitem__(self, index):
        scence_imgpath=self.data[index][0]
        label=self.data[index][1]
        img = Image.open(scence_imgpath).convert('RGB')
        img=nomalization(img)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)
        img=img.float()
        label=int(label)
        label=torch.from_numpy(np.array(label))
        return img,label
    def __len__(self):
        return len(self.data)

def save_false_class_img(label_true,label_false,labels,Pre_class,predictions,img_paths):
    labels=np.array(labels)
    Pre_class=np.array(Pre_class)
    predictions=np.array(predictions)
    for i in range(len(labels)):
        if labels[i]!=0:
            labels[i]=0
        else:
            labels[i] = 1

        if Pre_class[i]!=0:
            Pre_class[i]=0
        else:
            Pre_class[i] = 1
    Equal=np.equal(Pre_class,labels)
    Equal=Equal.astype(np.int32)
    index = np.where(Equal == 0)
    img_paths=np.array(img_paths)
    error_img_path = img_paths[index]
    class_label=labels[index]
    class_label=np.array(class_label).reshape(-1, 1)
    predictions=predictions[index]
    error_img_path = np.array(error_img_path).reshape(-1, 1)
    error_class = np.hstack((error_img_path, class_label, predictions))
    for img_path in error_class:
        img = io.imread(img_path[0].strip())
        imge = img[:, :, :3].copy()
        #img_data = img_path[0].strip().split('/')[5]
        img_name = img_path[0].strip().split('/')[-2] + '_' + img_path[0].strip().split('/')[-1]
        if int(img_path[1]) == 1:
            cv2.putText(imge, format(img_path[1]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(imge, '{} {}'.format(format(float(img_path[2]), '0.2f'), format(float(img_path[3]), '0.2f')),
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            imge = imge[:, :, ::-1]
            save_dir = label_true
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir,img_name),imge,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        if int(img_path[1]) == 0 :
            cv2.putText(imge, format(img_path[1]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(imge, '{} {}'.format(format(float(img_path[2]), '0.2f'), format(float(img_path[3]), '0.2f')),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            imge = imge[:, :, ::-1]
            save_dir = label_false
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir, img_name), imge, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def estimate(predictions, labels):
    y_true = labels
    y_pred = predictions
    Mat_result = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sample_num = len(labels)
    acc = np.trace(Mat_result) / sample_num
    FAR = (Mat_result[1][0] + Mat_result[2][0]) / (sum(iter(Mat_result[1:,:].reshape(-1,)))+ 1e-5)
    FRR = sum(iter(Mat_result[0][1:3])) / sum(iter(Mat_result[0][:]))
    Act_acc = (np.trace(Mat_result) + Mat_result[1][2] + Mat_result[2][1]) / sample_num
    result = np.hstack((FAR, FRR, acc, Act_acc))
    return result

def main(args):
    global device
    save_path='/home/data01_disk/lcw/code/train_save_file/false_class_img'
    dir=args.model_path.split('/')[-2]
    save_path=os.path.join(save_path,dir)
    save_label_true = save_path+'/pc_T'
    save_label_false = save_path+'/pc_F'
    if (not os.path.exists(save_label_true)) or (not os.path.exists(save_label_false)):
        os.makedirs(save_label_true)
        os.makedirs(save_label_false)
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=6, se=True, avgdown=True)
    #net = nn.DataParallel(net, device_ids=[0, 1])
    #net = net.to(device)
    print("start load train data")
    img_test = get_data(args.test_data, 'test', transform=None)
    test_loader = DataLoader(img_test, batch_size=500, num_workers=2, shuffle=False, drop_last=False)

    map_location = lambda storage, loc: storage
    checkpoint = torch.load(args.model_path,map_location=map_location)
    start_epoch = checkpoint['epoch']
    state_dict=checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    total_Pre_class = []
    total_labels = []
    total_predictions=[]
    total_imgpath=[]
    net.eval()
    for i, data in enumerate(test_loader):
        input, label= data
        # input = input.to(device)
        logits, predictions = net(input)
        _, preds_index = predictions.topk(1)
        preds_index = preds_index.view(-1, )
        preds_index = preds_index.detach().cpu()
        predictions = predictions.detach().cpu()
        label= label.numpy()
        predictions = predictions.numpy()
        preds_index = preds_index.numpy()
        total_Pre_class.extend(preds_index)
        total_labels.extend(label)
        total_predictions.extend(predictions)
        #total_imgpath.extend(img_path)
    result = estimate(total_Pre_class, total_labels)

    save_false_class_img(save_label_true, save_label_false,total_labels,total_Pre_class, total_predictions, total_imgpath)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Save_flag', type=bool, default=True, help='whether to save the model and tensorboard')
    parser.add_argument('-Save_model', type=str, help='model save path.',
                        default='/home/data01_disk/lcw/code/train_save_file/model')
    parser.add_argument('-Save_tensorboard', type=str, help='tensorboard save path.',
                        default='/home/data01_disk/lcw/code/train_save_file/log')
    parser.add_argument('-img_size', type=int, help='the input size', default=224)
    parser.add_argument('-class_num', type=int, help='class num', default=3)
    parser.add_argument('-retrain', type=bool, help='whether to fine-turn the model', default=True)
    parser.add_argument('-flag', type=str, help='train or evaluate the model', default='evaluate')  # softmax_loss,loss1
    parser.add_argument('-model_path', type=str, help='load the model path',
                        default='/home/data01_disk/lcw/code/train_save_file/model/0824_0719/Feathernet_145.pkl')
    # set the img data
    '''test data'''
    parser.add_argument('-test_data', type=str, help='data for testing',
                        default='/home/data01_disk/lcw/code/img_data/Scence/path')
    argv = parser.parse_args()
    return argv


if __name__ == '__main__':
    main(parse_arguments())