import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
# from pytorch import FeatherNet
#import FeatherNet
import FeatherNet_m
from skimage import io

from sklearn.metrics import confusion_matrix
import cv2
from torch.utils import data
import dataset
from torchvision import transforms as trans

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        #device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0':'cuda:2'})
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

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

class get_3d_liveness_data(data.Dataset):
    def __init__(self,data_path,flag,isnorm):
        self.data=self.make_data(data_path)
        if flag=='test':
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])
        else:
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])


    def __getitem__(self, index):
        img_path=self.data[index][0]
        label = self.data[index][1]
        img=Image.open(img_path)
        # img=self.trans(img)
        # img = 255 * img
        return img, label

    def make_data(self,data_path):
        data_lines=[]
        negs=0
        posts=0
        for path,dirs,files in os.walk(data_path):
            imgs = list(filter(lambda x: x.endswith(('png','jpg')), files))
            if imgs!=[]:
                img_paths=[os.path.join(path,img)for img in imgs]
                for img_path in img_paths:
                    label=0
                    if 'real' in img_path:
                        label=1
                        posts+=1
                    elif 'fake' in img_path:
                        label=0
                        negs+=1
                    data_lines.append((img_path,label))
        print('total sample is %d, postive sample is %d, negative sample is %d'%((posts+negs),posts,negs))
        return data_lines
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

# def estimate(predictions, labels):
#     y_true = labels
#     y_pred = predictions
#     Mat_result = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
#     sample_num = len(labels)
#     acc = np.trace(Mat_result) / sample_num
#     FAR = (Mat_result[1][0] + Mat_result[2][0]) / (sum(iter(Mat_result[1:,:].reshape(-1,)))+ 1e-5)
#     FRR = sum(iter(Mat_result[0][1:3])) / sum(iter(Mat_result[0][:]))
#     Act_acc = (np.trace(Mat_result) + Mat_result[1][2] + Mat_result[2][1]) / sample_num
#     result = np.hstack((FAR, FRR, acc, Act_acc))
#     return result

def estimate(predictions,labels):
    y_true=labels
    y_pred=predictions
    Mat_result=confusion_matrix(y_true,y_pred,labels=[0,1])
    sample_num=len(labels)
    acc=np.trace(Mat_result)/sample_num
    FAR=Mat_result[0,1]*1.0/(Mat_result[0,1]+Mat_result[1,1]+0.0001)
    FRR=Mat_result[1,0]*1.0/(Mat_result[1,0]+Mat_result[0,0]+0.0001)
    result=np.hstack((FAR,FRR,acc))
    return result

def eval_datasets(args):
    global device
    save_path = '/home/data03_disk/YZhang/irDatas/false_class_img'
    dir = args.model_path.split('/')[-2]
    save_path = os.path.join(save_path, dir)
    save_label_true = save_path + '/pc_T'
    save_label_false = save_path + '/pc_F'
    if (not os.path.exists(save_label_true)) or (not os.path.exists(save_label_false)):
        os.makedirs(save_label_true)
        os.makedirs(save_label_false)
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    #net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=4, is_training= True,
    #                            se=True, avgdown=True)
    net = FeatherNet_m.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=4,se=True, avgdown=True)
    # net = nn.DataParallel(net, device_ids=[0, 1])
    # net = net.to(device)
    print("start load test data")
    # img_test = get_data(args.test_data, 'test', transform=None)
    # test_loader = DataLoader(img_test, batch_size=128, num_workers=2, shuffle=False, drop_last=False)

    img_test = dataset.get_rgb_ir_liveness_data(args.test_data, 'test', True)
    img_test1 = dataset.get_rgb_ir_liveness_data_imgpath(args.test_data, 'test', True)
    #test_loader = DataLoader(img_test, batch_size=32, num_workers=2, shuffle=False, drop_last=False)
    test_loader = DataLoader(img_test, batch_size=1, num_workers=2, shuffle=False, drop_last=False)

    # net = load_model(net, args.model_path, False)
    # net = torch.load(args.model_path)

    map_location = lambda storage, loc: storage
    checkpoint = torch.load(args.model_path, map_location=map_location)
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # remove `module.`
        name = k  # remove `module.`

        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    total_Pre_class = []
    total_labels = []
    total_predictions = []
    total_ir_imgpath = []
    total_rgb_imgpath = []
    net.eval()
    for i, data in enumerate(test_loader):
        input, label = data
        # input = input.to(device)
        
        ir_img_path, rgb_img_path, label1 = img_test1[i]
        
        logits, predictions = net(input)
        _, preds_index = predictions.topk(1)
        preds_index = preds_index.view(-1, )
        preds_index = preds_index.detach().cpu()
        predictions = predictions.detach().cpu()
        label = label.numpy()
        predictions = predictions.numpy()
        preds_index = preds_index.numpy()
        total_Pre_class.extend(preds_index)
        total_labels.extend(label)
        if preds_index[0] != label1:
            print("*************************************************************************")
            print('preds_index is %d' %(preds_index[0]))
            print('label1 is %d' %(label1))
            print(ir_img_path)
            print(rgb_img_path)
        total_predictions.extend(predictions)
        total_ir_imgpath.extend(ir_img_path)
        total_rgb_imgpath.extend(rgb_img_path)
    result = estimate(total_Pre_class, total_labels)
    print('total data is %d' %(len(test_loader)))
    print("错误接受率 错误拒绝率 准确率")
    print(result)
    # save_false_class_img(save_label_true, save_label_false, total_labels, total_Pre_class, total_predictions,
    #                      total_imgpath)
import glob
try:
  from PIL import Image as pil_image
except ImportError:
  pil_image = None

if pil_image is not None:
  _PIL_INTERPOLATION_METHODS = {
      'nearest': pil_image.NEAREST,
      'bilinear': pil_image.BILINEAR,
      'bicubic': pil_image.BICUBIC,
  }

def load_img(path, grayscale=False, target_size=None, interpolation='nearest'):
  if pil_image is None:
    raise ImportError('Could not import PIL.Image. '
                      'The use of `array_to_img` requires PIL.')
  img = pil_image.open(path)
  if grayscale:
    if img.mode != 'L':
      img = img.convert('L')
  else:
    if img.mode != 'RGB':
      img = img.convert('RGB')
  if target_size is not None:
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
      if interpolation not in _PIL_INTERPOLATION_METHODS:
        raise ValueError('Invalid interpolation method {} specified. Supported '
                         'methods are {}'.format(interpolation, ', '.join(
                             _PIL_INTERPOLATION_METHODS.keys())))
      resample = _PIL_INTERPOLATION_METHODS[interpolation]
      img = img.resize(width_height_tuple, resample)
  return img
def get_ir_samples(ir_filename):
    i = 0
    isNorm = True
    index_array_size = 1
    self_channel = 1
    self_target_size = (224, 224)
    self_interpolation = 'bilinear'
    self_grayscale = True
    batch_x = np.zeros(
          tuple([(index_array_size)] + list(self_target_size) + [self_channel] ), dtype=np.float32)


    ir_img = load_img( ir_filename,
                        target_size=self_target_size, interpolation=self_interpolation, grayscale=self_grayscale)

    ir_image_np = np.array(ir_img)
    # if (isNorm):
    #     ir_image_np = standardImg(ir_image_np)


    if self_grayscale:
        ir_image_np = ir_image_np.reshape(self_target_size[1], self_target_size[0], 1)


    x = ir_image_np
    batch_x[i] = x

    return batch_x

def main(args):
    eval_datasets(args)


    # net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels, is_training=False,
    #                             se=True, avgdown=True)
    #
    #
    # if not os.path.exists(args.model_path):
    #     print('model path is error')
    # else:
    #     net = torch.load(args.model_path)
    #
    # raw_path = args.test_data + '/*/*.png'
    # imgs = sorted(glob.glob(raw_path))
    # print(len(imgs))
    # for p_index, p_name in enumerate(imgs):
    #     ir_img_path = p_name
    #     print('index = {}, {}'.format(p_index, ir_img_path))
    #     ir_img = cv2.imread(ir_img_path, 0)

    # torch.set_grad_enabled(False)
    # global device
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = torch.device('cpu')
    # net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels, is_training=False,
    #                             se=True, avgdown=True)
    # net = net.to(device)
    # net = load_model(net, args.model_path, args.cpu)
    # net.eval()
    # net = net.to(device)
    # print('Finished loading model!')
    #
    # print("start load test data")
    #
    # img_test = dataset.get_3d_liveness_data(args.test_data, 'test', True)
    # test_loader = DataLoader(img_test, batch_size=32, num_workers=4, shuffle=False, drop_last=False)
    #
    # total_Pre_class = []
    # total_labels = []
    # total_predictions = []
    # total_imgpath = []
    # # net.eval()
    # for i, data in enumerate(test_loader):
    #     input, label = data
    #     # input = input.to(device)
    #     logits, predictions = net(input)
    #     _, preds_index = predictions.topk(1)
    #     preds_index = preds_index.view(-1, )
    #     preds_index = preds_index.detach().cpu()
    #     predictions = predictions.detach().cpu()
    #     label = label.numpy()
    #     predictions = predictions.numpy()
    #     preds_index = preds_index.numpy()
    #     total_Pre_class.extend(preds_index)
    #     total_labels.extend(label)
    #     total_predictions.extend(predictions)

    # raw_path = args.test_data + '/*/*.png'
    # imgs = sorted(glob.glob(raw_path))
    # print(len(imgs))
    # for p_index, p_name in enumerate(imgs):
    #     ir_img_path = p_name
    #     print('index = {}, {}'.format(p_index, ir_img_path))
    #     # ir_img = cv2.imread(ir_img_path, 0)
    #     # ir_img = get_ir_samples(ir_img_path)
    #     # img = np.float32(ir_img)
    #     # # img = torch.from_numpy(img).unsqueeze(0)
    #     # img = torch.from_numpy(img)
    #
    #     img = cv2.imread(ir_img_path,0)
    #     img = np.expand_dims(img, 2)
    #     img = img.transpose(2, 0, 1)
    #     # img = img.numpy()
    #     img = torch.from_numpy(img).unsqueeze(0)
    #
    #     img = img.to(device)
    #     logits, predictions= net(img)  # forward pass
    #     _, preds_index = predictions.topk(1)
    #     preds_index = preds_index.view(-1, )
    #     preds_index = preds_index.detach().cpu()
    #     predictions = predictions.detach().cpu()

    # print("start load test data")
    # img_test = get_3d_liveness_data(args.test_data, 'test', False)
    # test_loader = DataLoader(img_test, batch_size=32, num_workers=2, shuffle=False, drop_last=False)
    #
    # total_Pre_class = []
    # total_labels = []
    # total_predictions = []
    # total_imgpath = []
    # for i, data in enumerate(test_loader):
    #     input, label = data
    #     # input = input.to(device)
    #     logits, predictions = net(input)
    #     _, preds_index = predictions.topk(1)
    #     preds_index = preds_index.view(-1, )
    #     preds_index = preds_index.detach().cpu()
    #     predictions = predictions.detach().cpu()
    #     label = label.numpy()
    #     predictions = predictions.numpy()
    #     preds_index = preds_index.numpy()
    #     total_Pre_class.extend(preds_index)
    #     total_labels.extend(label)
    #     total_predictions.extend(predictions)
    #     # total_imgpath.extend(img_path)
    # result = estimate(total_Pre_class, total_labels)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Save_flag', type=bool, default=True, help='whether to save the model and tensorboard')
    parser.add_argument('-Save_model', type=str, help='model save path.',
                        default='/home/data03_disk/YZhang/3dliveness_training_record/model')
    parser.add_argument('-Save_tensorboard', type=str, help='tensorboard save path.',
                        default='/home/data03_disk/YZhang/3dliveness_training_record/log')
    parser.add_argument('-img_size', type=int, help='the input size', default=224)
    parser.add_argument('-class_num', type=int, help='class num', default=2)
    parser.add_argument('-input_channels', type=int, help='the input channels', default=1)
    parser.add_argument('--long_side', default=[224, 224],
                        help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

    parser.add_argument('-retrain', type=bool, help='whether to fine-turn the model', default=False)
    parser.add_argument('-flag', type=str, help='train or evaluate the model', default='evaluate')  # softmax_loss,loss1
    parser.add_argument('-model_path', type=str, help='load the model path',
                        default='/home/data03_disk/YZhang/3dliveness_training_record/model/1108_0215/Feathernet_79.pkl')
    # set the img data
    '''test data'''
    parser.add_argument('-test_data', type=str, help='data for testing',
                        default='/home/data03_disk/YZhang/irDatas/bctc_test_data_1104')
    argv = parser.parse_args()
    return argv


if __name__ == '__main__':
    main(parse_arguments())