# -*- coding: utf-8 -*-
"""
Copyright(c) 2018 by Ningbo XiTang Information Technologies, Inc and
WuQi Technologies, Inc. ALL RIGHTS RESERVED.

This Information is proprietary to XiTang and WuQi, and MAY NOT be copied by
any method or incorporated into another program without the express written
consent of XiTang and WuQi. This Information or any portion thereof remains
the property of XiTang and WuQi. The Information contained herein is believed
to be accurate and XiTang and WuQi assumes no responsibility or liability for
its use in any way and conveys no license or title under any patent or copyright
and makes no representation or warranty that this Information is free from patent
or copyright infringement.
"""
import os

import json
import numpy as np
import random
import cv2
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
  """Loads an image into PIL format.
  Arguments:
      path: Path to image file
      grayscale: Boolean, whether to load the image as grayscale.
      target_size: Either `None` (default to original size)
          or tuple of ints `(img_height, img_width)`.
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are "nearest", "bilinear", and "bicubic".
          If PIL version 1.1.3 or newer is installed, "lanczos" is also
          supported. If PIL version 3.4.0 or newer is installed, "box" and
          "hamming" are also supported. By default, "nearest" is used.
  Returns:
      A PIL Image instance.
  Raises:
      ImportError: if PIL is not available.
      ValueError: if interpolation method is not supported.
  """
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

def to_one_hot(y, C):
    batch_size = len(y)
    return np.eye(C)[y.reshape(-1)].reshape(batch_size, -1, 1, 1)

def pad(train_data, size=(32, 32)):
    n, c, h, w = train_data.shape
    paded_data = np.zeros((n, c, size[0], size[1]))
    off_h = (size[0]-h)//2
    off_w = (size[1]-w)//2
    paded_data[:, :, off_h:size[0]-off_h, off_w:size[1]-off_w] = train_data
    return paded_data

def normalization(img):
    img = np.array(img)

    img = (img - np.mean(img)) / np.std(img)
    return img

def standardImg(img):
    img = np.array(img)
    img = (img - 127.5) / 255.0
    return img

def random_float(f_min, f_max):
    return f_min + (f_max-f_min) * random.random()

def get_cut_out(img, length=50):
    # print(len(img.shape))
    h, w = img.shape[0], img.shape[1]  # Tensor [1][2],  nparray [0][1]
    if len(img.shape) == 3:
        mask = np.ones((h, w, 3), np.float32)
    else:
        mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    length_new = np.random.randint(1, length)

    y1 = np.clip(y - length_new // 2, 0, h)
    y2 = np.clip(y + length_new // 2, 0, h)
    x1 = np.clip(x - length_new // 2, 0, w)
    x2 = np.clip(x + length_new // 2, 0, w)

    if len(img.shape) == 3:
        mask[y1: y2, x1: x2, :] = 0
    else:
        mask[y1: y2, x1: x2] = 0
    img *= np.array(mask, np.uint8)
    return img

class Data_Generator():
  """
  Arguments:
      n: Integer, total number of samples in the dataset to loop over.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seeding for data shuffling.
  """

  def __init__(self,
               data_dir,
               batch_size,
               shuffle,
               seed,
               n,
               filenames_to_class='training_data_paired.json',
               target_size=(224,224),
               interpolation='bilinear',
               grayscale = True,
               one_hot = True,
               num_class = 2, isTrain = True):

    self.data_dir = data_dir
    with open(os.path.join(data_dir, filenames_to_class), 'r') as f:
      self.filenames_to_class = json.load(f)

    self.paired_num = len(self.filenames_to_class)
    self.n = (self.paired_num//batch_size)*batch_size
    print('self.n = {}'.format(self.n))
    self.batch_size = batch_size
    self.seed = seed
    self.shuffle = shuffle
    self.batch_index = 0
    self.total_batches_seen = 0
    self.index_array = None
    self.index_generator = self._flow_index()
    self.target_size = target_size
    self.interpolation = interpolation
    self.grayscale = grayscale
    if grayscale:
        self.channel = 2
    else:
        self.channel = 6
    self.one_hot = one_hot
    self.num_class = num_class
    self.isTrain = isTrain

    # self.one_paired_samples_generator = self._get_batches_of_paired_samples()

  def _set_index_array(self):
    self.index_array = np.arange(self.n)
    if self.shuffle:
      self.index_array = np.random.permutation(self.n)

  def __getitem__(self, idx):
    if idx >= len(self):
      raise ValueError('Asked to retrieve element {idx}, '
                       'but the Sequence '
                       'has length {length}'.format(idx=idx, length=len(self)))
    if self.seed is not None:
      np.random.seed(self.seed + self.total_batches_seen)
    self.total_batches_seen += 1
    if self.index_array is None:
      self._set_index_array()
    index_array = self.index_array[self.batch_size * idx:self.batch_size * (
        idx + 1)]
    return self._get_batches_of_paired_samples(index_array)

  def __len__(self):
    return (self.n + self.batch_size - 1) // self.batch_size  # round up

  def on_epoch_end(self):
    self._set_index_array()

  def reset(self):
    self.batch_index = 0

  def _flow_index(self):
    # Ensure self.batch_index is 0.
    self.reset()
    while 1:
      if self.seed is not None:
        np.random.seed(self.seed + self.total_batches_seen)
      if self.batch_index == 0:
        self._set_index_array()

      current_index = (int)(self.batch_index * self.batch_size) % self.n


      # print('current_index = {} '.format(current_index))

      if self.n > current_index + self.batch_size:
        self.batch_index += 1
      else:
        self.batch_index = 0
      self.total_batches_seen += 1
      yield self.index_array[current_index:current_index + self.batch_size]

  def __iter__(self):
    return self

  def _get_batches_of_paired_samples(self, index_array):
    isNorm = False
    batch_x = np.zeros(
        tuple([len(index_array)] + [self.channel] + list(self.target_size)), dtype=np.float32)
    batch_y = np.zeros((len(index_array),), dtype=np.int)

    for i, j in enumerate(index_array):
        if random_float(0.0, 1.0) < 0.5:
            is_random_flip = True
        else:
            is_random_flip = False

        if random_float(0.0, 1.0) < 0.5:
            is_cut_out = True
        else:
            is_cut_out = False
        is_cut_out = True
        # print(i,j)
        ir_filename = self.filenames_to_class[j]['ir_path']
        depth_filename = self.filenames_to_class[j]['depth_path']

        label = self.filenames_to_class[j]['label']
        ir_filename = os.path.join(self.data_dir, ir_filename)
        depth_filename = os.path.join(self.data_dir, depth_filename)
        # print(ir_filename)

        ir_img = cv2.imread(ir_filename, 0)
        depth_img  = cv2.imread(depth_filename, 0)
        if self.isTrain:
            if is_random_flip:
                ir_img = cv2.flip(ir_img, 1)
                depth_img = cv2.flip(depth_img, 1)

            if is_cut_out:
                ir_img = get_cut_out(ir_img)
                depth_img = get_cut_out(depth_img)

        ir_img = cv2.resize(ir_img, (224, 224))
        depth_img = cv2.resize(depth_img, (224, 224))

        if (isNorm):
            ir_image_np = standardImg(ir_img)
            depth_image_np = standardImg(depth_img)
        else:
            ir_image_np = np.array(ir_img)
            depth_image_np = np.array(depth_img)

        if self.grayscale:
          ir_image_np = ir_image_np.reshape(self.target_size[1],self.target_size[0],1)
          depth_image_np = depth_image_np.reshape(self.target_size[1],self.target_size[0],1)
        img = np.concatenate((ir_image_np, depth_image_np), 2)
        img = np.transpose(img, axes=(2, 0, 1))
        x = img
        batch_x[i] = x
        batch_y[i] = int(label)
    # self.one_hot = False
    if self.one_hot:
        batch_y = to_one_hot(batch_y, self.num_class)
    return batch_x, batch_y




  def _get_batches_of_samples(self, index_array):
    batch_x = np.zeros(
        tuple([len(index_array)] + [self.channel] + list(self.target_size)), dtype=np.float32)
    batch_y = np.zeros((len(index_array),), dtype=np.int)

    for i, j in enumerate(index_array):
      img = load_img(os.path.join(self.data_dir, self.idx_to_fname[j]),
                   target_size=self.target_size, interpolation=self.interpolation, grayscale=self.grayscale)
      img = np.array(img)
      if not self.grayscale:
          img = np.transpose(img, axes=(2,0,1))
      x = img
      batch_x[i] = x
      batch_y[i] = int(self.filenames_to_class[self.idx_to_fname[j]])

    if self.one_hot:
        batch_y = to_one_hot(batch_y, self.num_class)

    return batch_x, batch_y

  def next(self):
    """For python 2.x.
    Returns:
        The next batch.
    """
    index_array = next(self.index_generator)
    return self._get_batches_of_paired_samples(index_array)

  def __next__(self):
    return self.next()





def data_generator_paired_images(
               imagenet_dirname,
               batch_size,
               random_order,
               seed=100,
               n=1000,
               filenames_to_class = 'training_data_paired.json',
               target_size=(224,224),
               interpolation='bilinear',
               grayscale = False,
               one_hot = True,
               num_class = 2,isTrain = True):

    dg = Data_Generator(data_dir=imagenet_dirname,batch_size=batch_size,shuffle=random_order,
                        seed=seed,n=n,filenames_to_class=filenames_to_class,
                        target_size=target_size,interpolation=interpolation,grayscale=grayscale,
                        one_hot=one_hot,num_class=num_class,isTrain=isTrain)

    while 1:
        batchx, batchy = next(dg)
        yield {'input':batchx,'output':batchy}






if __name__ == '__main__':


    # input_path = '/home/data01_disk/share/quant3Dliveness/grayFacesTrain'
    # json_file_name = 'training_grayscale_paired.json'
    # sg = data_generator_paired_images(
    # imagenet_dirname=input_path,
    # batch_size=32,
    # random_order=True,
    # filenames_to_class = json_file_name,
    # grayscale=True
    # )
    #
    #
    # for i in range(11000):
    #     data_batch = next(sg)
    #     batchx = data_batch['input']
    #     batchy = data_batch['output']
    #     print(i,batchx.shape, batchy.shape)
    train_data_path = '/home/data03_disk/YZhang/grayAugTrain'
    train_json_path = '/home/data03_disk/YZhang/grayAugTrain/aug_train_paired.json'
    test_data_path = '/home/data03_disk/YZhang/grayAugTest'
    test_json_path = '/home/data03_disk/YZhang/grayAugTest/aug_test_paired.json'

    batch_size = 64
    g_train = data_generator_paired_images(
        imagenet_dirname= train_data_path,
        batch_size= batch_size,
        random_order=True,
        filenames_to_class= train_json_path,
        grayscale=True
    )

    g_test = data_generator_paired_images(
        imagenet_dirname=test_data_path,
        batch_size=batch_size,
        random_order=False,
        filenames_to_class=test_json_path,
        grayscale=True
    )

    print('training batch:')
    for i in range(500):
        batch = next(g_train)
        batch_in = batch['input']
        batch_out = batch['output']
        print(i, batch_in.shape, batch_out.shape)
        if i > 500:
            break
    print('testing batch:')
    for j in range(500):
        batch = next(g_test)
        batch_in = batch['input']
        batch_out = batch['output']
        print(j, batch_in.shape, batch_out.shape)
