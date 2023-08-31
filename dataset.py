import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as trans
import cv2

def load_all_txts(txt_dir):
    files=os.listdir(txt_dir)
    txts=list(filter(lambda x : x.endswith(('txt')),files))
    txts.sort()
    txts=[os.path.join(txt_dir,txt) for txt in txts]
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
        img_paths.append(img_inf.split('\t')[0])
        labels.append(int(img_inf.split('\t')[1]))
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
    def __init__(self,scence_path,face_path,phase,transform):
        self.data=make_data(scence_path,face_path)
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
        face_imgpath = self.data[index][1]
        label=self.data[index][2]
        img_scence = Image.open(scence_imgpath).convert('RGB')
        img_face = Image.open(face_imgpath).convert('RGB')
        if self.phase=='train':
            img_face = self.transform(img_face)
            #img_scence = self.transform(img_scence)
        img_face=nomalization(img_face)
        img_scence=nomalization(img_scence)
        img=np.concatenate((img_scence,img_face),2)
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
        img=self.trans(img)
        img = 255 * img
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
        #-1：原图 0：灰度图 1：彩色图
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


class get_rgb_ir_liveness_data(data.Dataset):
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
        rgb_img_path=self.data[index][1]
        label = self.data[index][2]
        # print(ir_img_path)
        # print(depth_img_path)
        # ir_img=Image.open(ir_img_path)
        # depth_img = Image.open(depth_img_path)
        # img = ir_img
        #-1：原图 0：灰度图 1：彩色图
        ir_img = cv2.imread(ir_img_path, 0)
        rgb_img = cv2.imread(rgb_img_path)
        ir_img = cv2.resize(ir_img, (224, 224))
        rgb_img = cv2.resize(rgb_img, (224, 224))
        img = cv2.merge([ir_img, rgb_img])
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
            #print(p_name)
            if '-color' in p_name and p_index % 2 == 0:
                rgb_path = imgs[p_index]
                ir_path = imgs[p_index + 1]
                # print('index = {}, ir_path = {}'.format(p_index, ir_path))
                label = 0
                if 'real' in ir_path:
                    label = 1
                    posts += 1
                elif 'fake' in ir_path:
                    label = 0
                    negs += 1
                data_lines.append((ir_path, rgb_path,label))
        print('total sample is %d, postive sample is %d, negative sample is %d'%((posts+negs),posts,negs))
        return data_lines
    def __len__(self):
        return len(self.data)
        
class get_rgb_ir_liveness_data_from_txt(data.Dataset):
    def __init__(self,txtfile_path,flag,isnorm,DATA_ROOT):
        self.data=self.make_data(txtfile_path)
        self.DATA_ROOT = DATA_ROOT
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
        ir_img_path=self.data[index][1]
        rgb_img_path=self.data[index][0]
        label = self.data[index][2]
        # print(ir_img_path)
        # print(depth_img_path)
        # ir_img=Image.open(ir_img_path)
        # depth_img = Image.open(depth_img_path)
        # img = ir_img
        #-1：原图 0：灰度图 1：彩色图
        ir_img = cv2.imread(self.DATA_ROOT+ir_img_path, 0)
        rgb_img = cv2.imread(self.DATA_ROOT+rgb_img_path)
        ir_img = cv2.resize(ir_img, (224, 224))
        rgb_img = cv2.resize(rgb_img, (224, 224))
        img = cv2.merge([ir_img, rgb_img])
        img = Image.fromarray(img)

        img=self.trans(img)
        label=np.array(label).astype(int)
        label=torch.from_numpy(label)
        # img = 255 * img
        return img, label

    def make_data(self,txtfile_path):
        data_lines=[]
        f = open(txtfile_path)
        #print(DATA_ROOT + '/casia_surf_train.txt')
        #0428_new_train/real/real-train/1650870611-color.jpg 0428_new_train/real/real-train/1650870611-ir.jpg 1
        lines = f.readlines()

        for line in lines:
            line = line.strip().split(' ')
            #print(line)
            #data_lines.append(line)
            data_lines.append((line[1], line[0],line[2]))
        return data_lines      
    def __len__(self):
        return len(self.data)
        
class get_ir_liveness_data(data.Dataset):
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
        label = self.data[index][1]
        # print(ir_img_path)
        # print(depth_img_path)
        # ir_img=Image.open(ir_img_path)
        # depth_img = Image.open(depth_img_path)
        # img = ir_img
        #-1：原图 0：灰度图 1：彩色图
        ir_img = cv2.imread(ir_img_path, 0)
        ir_img = cv2.resize(ir_img, (224, 224))
        img = Image.fromarray(ir_img)

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

    def make_data(self, data_path):
        data_lines = []
        negs = 0
        posts = 0
        for path, dirs, files in os.walk(data_path):
            imgs = list(filter(lambda x: x.endswith(('png', 'jpg')), files))
            if imgs != []:
                img_paths = [os.path.join(path, img) for img in imgs]
                for img_path in img_paths:
                    # print(img_path)
                    label = 0
                    if 'real' in img_path:
                        label = 1
                        posts += 1
                    elif 'fake' in img_path:
                        label = 0
                        negs += 1
                    data_lines.append((img_path, label))
        print('total sample is %d, postive sample is %d, negative sample is %d' % ((posts + negs), posts, negs))
        return data_lines
        
    def __len__(self):
        return len(self.data)
 
class get_ir_liveness_data_imgpath(data.Dataset):
    def __init__(self,data_path,flag,isnorm):
        self.data=self.make_data(data_path)
        
    def __getitem__(self, index):
        ir_img_path=self.data[index][0]
        label = self.data[index][1]

        return ir_img_path, label

    def make_data(self, data_path):
        data_lines = []
        negs = 0
        posts = 0
        for path, dirs, files in os.walk(data_path):
            imgs = list(filter(lambda x: x.endswith(('png', 'jpg')), files))
            if imgs != []:
                img_paths = [os.path.join(path, img) for img in imgs]
                for img_path in img_paths:
                    # print(img_path)
                    label = 0
                    if 'real' in img_path:
                        label = 1
                        posts += 1
                    elif 'fake' in img_path:
                        label = 0
                        negs += 1
                    data_lines.append((img_path, label))
        #print('total sample is %d, postive sample is %d, negative sample is %d' % ((posts + negs), posts, negs))
        return data_lines
 
        
class get_rgb_ir_liveness_data_imgpath(data.Dataset):
    def __init__(self,data_path,flag,isnorm):
        self.data=self.make_data(data_path)

    def __getitem__(self, index):
        ir_img_path=self.data[index][0]
        rgb_img_path=self.data[index][1]
        label = self.data[index][2]

        return ir_img_path, rgb_img_path, label

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
            #print(p_name)
            if '-color' in p_name and p_index % 2 == 0:
                rgb_path = imgs[p_index]
                ir_path = imgs[p_index + 1]
                # print('index = {}, ir_path = {}'.format(p_index, ir_path))
                label = 0
                if 'real' in ir_path:
                    label = 1
                    posts += 1
                elif 'fake' in ir_path:
                    label = 0
                    negs += 1
                data_lines.append((ir_path, rgb_path,label))
        #print('total sample is %d, postive sample is %d, negative sample is %d'%((posts+negs),posts,negs))
        return data_lines


class get_rgb_ir_liveness_data_imgpath_from_txt(data.Dataset):
    def __init__(self,txtfile_path,flag,isnorm,DATA_ROOT):
        self.data=self.make_data(txtfile_path)
        self.DATA_ROOT = DATA_ROOT

    def __getitem__(self, index):
        ir_img_path=self.DATA_ROOT+self.data[index][1]
        rgb_img_path=self.DATA_ROOT+self.data[index][0]
        label = self.data[index][2]
        
        return ir_img_path, rgb_img_path, label

    def make_data(self,txtfile_path):
        data_lines=[]
        f = open(txtfile_path)
        #print(DATA_ROOT + '/casia_surf_train.txt')
        #0428_new_train/real/real-train/1650870611-color.jpg 0428_new_train/real/real-train/1650870611-ir.jpg 1
        lines = f.readlines()

        for line in lines:
            line = line.strip().split(' ')
            data_lines.append(line)
        return data_lines









