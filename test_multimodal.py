import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
# from pytorch import FeatherNet
import FeatherNet
from skimage import io
from torchvision import transforms as trans
from sklearn.metrics import confusion_matrix
import cv2
from torch.utils import data

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
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        pretrained_dict=checkpoint['state_dict']
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def image_paths_list(input_path):
    num = 0
    all_images_path = []
    for path, dirs, files in os.walk(input_path):

        img_files = list(filter(lambda x: x.endswith(('.png', '.jpg')), files))
        if img_files != []:
            img_paths = [os.path.join(path, bin_file) for bin_file in img_files]
            img_paths = sorted(img_paths)
            for image_path in img_paths:
                # print(image_path)
                all_images_path.append(image_path)
                num += 1
    print(num)
    all_images_path = sorted(all_images_path)
    return all_images_path

''' (x - mean) / std
    则weight取1 / std, bias取 - mean / std '''

def multimodal_norm(img):
    # weight = np.array([1 / 255 / 0.5, 1 / 255 / 0.5], dtype=np.float32)
    # bias = np.array([-0.5 / 0.5, -0.5 / 0.5], dtype=np.float32)
    img = np.array(img)
    img = img/255
    mean = 0.5
    std = 0.5
    img = (img - mean) / std
    return img


class get_multimodal_data(data.Dataset):
    def __init__(self, data_path):
        self.data_lines = []
        imgs = image_paths_list(data_path)
        for p_index, p_name in enumerate(imgs):
            if '-depth' in p_name and p_index % 2 == 0:
                depth_path = imgs[p_index]
                ir_path = imgs[p_index + 1]
                label = 0
                if 'real' in ir_path:
                    label = 1
                elif 'fake' in ir_path:
                    label = 0
                self.data_lines.append((ir_path, depth_path,label))

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        ir_img_path = self.data_lines[index][0]
        depth_img_path = self.data_lines[index][1]
        label = self.data_lines[index][2]
        ir_img = cv2.imread(ir_img_path, 0)
        depth_img = cv2.imread(depth_img_path, 0)
        ir_img = cv2.resize(ir_img, (224, 224))
        depth_img = cv2.resize(depth_img, (224, 224))
        ir_img_norm = multimodal_norm(ir_img)
        depth_img_norm = multimodal_norm(depth_img)
        ir_image_np = np.expand_dims(ir_img_norm, 2)
        depth_image_np = np.expand_dims(ir_img_norm, 2)
        img = np.concatenate((ir_image_np, depth_image_np), 2)
        # img = cv2.merge([ir_img, depth_img])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float()
        label = int(label)
        label = torch.from_numpy(np.array(label))
        return img,label
class get_multimodal_liveness_data(data.Dataset):
    def __init__(self,data_path,flag,isnorm):
        self.data=self.make_data(data_path)
        if flag=='test':
            if isnorm:
                self.trans = trans.Compose([
                    # trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    # trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])
        else:
            if isnorm:
                self.trans = trans.Compose([
                    # trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    # trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    # trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    # trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])


    def __getitem__(self, index):
        ir_img_path=self.data[index][0]
        depth_img_path=self.data[index][1]
        label = self.data[index][2]
        # print(ir_img_path)
        # print(depth_img_path)
        # ir_img=Image.open(ir_img_path)
        # depth_img = Image.open(depth_img_path)
        # img = ir_img
        ir_img = cv2.imread(ir_img_path, 0)
        depth_img = cv2.imread(depth_img_path, 0)
        ir_img = cv2.resize(ir_img, (224, 224))
        depth_img = cv2.resize(depth_img, (224, 224))
        img = cv2.merge([ir_img, depth_img])
        img = Image.fromarray(img)

        img=self.trans(img)
        # img = 255 * img
        return img, label

    def image_paths_list(self, input_path):
        num = 0
        all_images_path = []
        for path, dirs, files in os.walk(input_path):

            img_files = list(filter(lambda x: x.endswith(('.png', '.jpg')), files))
            if img_files != []:
                img_paths = [os.path.join(path, bin_file) for bin_file in img_files]
                img_paths = sorted(img_paths)
                for image_path in img_paths:
                    # print(image_path)
                    all_images_path.append(image_path)
                    num += 1
        print(num)
        all_images_path = sorted(all_images_path)
        return all_images_path

    def make_data(self,data_path):
        data_lines=[]
        negs=0
        posts=0
        imgs = self.image_paths_list(data_path)
        for p_index, p_name in enumerate(imgs):

            if '-depth' in p_name and p_index % 2 == 0:
                depth_path = imgs[p_index]
                ir_path = imgs[p_index + 1]
                # print('index = {}, ir_path = {}'.format(p_index, ir_path))
                label = 0
                if 'real' in ir_path:
                    label = 1
                    posts += 1
                elif 'fake' in ir_path:
                    label = 0
                    negs += 1
                data_lines.append((ir_path, depth_path,label))
        print('total sample is %d, postive sample is %d, negative sample is %d'%((posts+negs),posts,negs))
        return data_lines
    def __len__(self):
        return len(self.data)
def main(args):

    torch.set_grad_enabled(False)
    global device
    device = torch.device('cpu')

    net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels,
                                is_training=False, se=True, avgdown=True)

    net = net.to(device)
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    print('Finished loading model!')

    print("start load test data")
    # img_test = get_multimodal_data(args.test_data)
    # test_loader = DataLoader(img_test, batch_size=128, num_workers=2, shuffle=False, drop_last=False)
    img_test = get_multimodal_liveness_data(args.test_data, 'test', True)
    test_loader = DataLoader(img_test, batch_size=128, num_workers=2, shuffle=False, drop_last=False)

    total_Pre_class = []
    total_labels = []
    total_predictions=[]
    total_imgpath=[]

    for i, data in enumerate(test_loader):
        input, label= data
        predictions = net(input)
        print(predictions)
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
    print(result)

def get_paths_list(input_path):
    num = 0
    all_images_path = []
    for path, dirs, files in os.walk(input_path):

        img_files = list(filter(lambda x: x.endswith(('.png', '.jpg')), files))
        if img_files != []:
            img_paths = [os.path.join(path, bin_file) for bin_file in img_files]
            img_paths = sorted(img_paths)
            for image_path in img_paths:
                # print(image_path)
                all_images_path.append(image_path)
                num += 1
    print(num)
    all_images_path = sorted(all_images_path)
    return all_images_path

def make_multimodal_data(data_path):
    data_lines = []
    negs = 0
    posts = 0
    imgs = get_paths_list(data_path)
    for p_index, p_name in enumerate(imgs):

        if '-depth' in p_name and p_index % 2 == 0:
            depth_path = imgs[p_index]
            ir_path = imgs[p_index + 1]
            # print('index = {}, ir_path = {}'.format(p_index, ir_path))
            label = 0
            if 'real' in ir_path:
                label = 1
                posts += 1
            elif 'fake' in ir_path:
                label = 0
                negs += 1
            data_lines.append((ir_path, depth_path, label))
    print('total sample is %d, postive sample is %d, negative sample is %d' % ((posts + negs), posts, negs))
    return data_lines
def multimodal_input(data_line):
    ir_img_path = data_line[0]
    depth_img_path = data_line[1]
    label =data_line[2]
    ir_img = cv2.imread(ir_img_path, 0)
    depth_img = cv2.imread(depth_img_path, 0)
    ir_img = cv2.resize(ir_img, (224, 224))
    depth_img = cv2.resize(depth_img, (224, 224))
    img = cv2.merge([ir_img, depth_img])
    img = Image.fromarray(img)
    img_trans = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5], [0.5]),
    ])
    img = img_trans(img)
    # img = 255 * img
    img = img.unsqueeze(0)

    return img, label

def main_test(args):

    torch.set_grad_enabled(False)
    global device
    device = torch.device('cpu')

    net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels,
                                is_training=False, se=True, avgdown=True)

    net = net.to(device)
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    print('Finished loading model!')

    print("start load test data")

    # img_test = get_multimodal_liveness_data(args.test_data, 'test', True)
    # test_loader = DataLoader(img_test, batch_size=128, num_workers=2, shuffle=False, drop_last=False)
    test_data = make_multimodal_data(args.test_data)
    total_Pre_class = []
    total_labels = []
    total_predictions=[]
    total_imgpath=[]

    for index, data_line in enumerate(test_data):
        input, label= multimodal_input(data_line)
        print(data_line)
        predictions = net(input)
        print(predictions)
        _, preds_index = predictions.topk(1)
        preds_index = preds_index.view(-1, )
        preds_index = preds_index.detach().cpu()
        predictions = predictions.detach().cpu()
        # label= label.numpy()
        label = [label]
        predictions = predictions.numpy()
        preds_index = preds_index.numpy()
        print(preds_index)

        total_Pre_class.extend(preds_index)
        total_labels.extend(label)
        total_predictions.extend(predictions)
        #total_imgpath.extend(img_path)
    result = estimate(total_Pre_class, total_labels)
    print(result)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Save_flag', type=bool, default=True, help='whether to save the model and tensorboard')
    parser.add_argument('-Save_model', type=str, help='model save path.',
                        default='/home/data01_disk/lcw/code/train_save_file/model')
    parser.add_argument('-Save_tensorboard', type=str, help='tensorboard save path.',
                        default='/home/data01_disk/lcw/code/train_save_file/log')
    parser.add_argument('-img_size', type=int, help='the input size', default=224)
    parser.add_argument('-class_num', type=int, help='class num', default=2)
    parser.add_argument('-input_channels', type=int, help='the input channels', default=2)
    parser.add_argument('-retrain', type=bool, help='whether to fine-turn the model', default=True)
    parser.add_argument('-flag', type=str, help='train or evaluate the model', default='evaluate')  # softmax_loss,loss1
    parser.add_argument('-model_path', type=str, help='load the model path',
                        default='/home/data03_disk/YZhang/multimodal_training_record/model/1215_0955/Feathernet_91.pkl')
    parser.add_argument('-m', '--trained_model',
                        default='/home/data03_disk/YZhang/multimodal_training_record/model/1215_0955/Feathernet_91.pkl',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    # set the img data
    '''test data'''
    parser.add_argument('-test_data', type=str, help='data for testing',
                        default='/home/data03_disk/YZhang/HSLX1Datas/multiModalDatas/multiModalFaces1206/test')
    argv = parser.parse_args()
    return argv


if __name__ == '__main__':
    # main(parse_arguments())
    main_test(parse_arguments())