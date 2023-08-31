#-*- coding:utf-8 -*-

import os
import sys
import glob
import cv2
import shutil
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def decode_images(raw_dir):
    raw_path = raw_dir + '/*.bin'
    imgs =sorted(glob.glob(raw_path))
    # print(imgs)

    save_dir = raw_dir + "_png"
    # save_dir = raw_dir

    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
    for ig in imgs:
        print(ig)

        if "-depth" in ig:
            img = np.fromfile(ig, dtype=np.uint16)
            # img.shape = (1024, 768)
            img.shape = (1280, 768)
            img = img / 8
            img = img.astype(np.uint8)
            new_img = img
            name = ig.strip().split(".")[0]
            name = name.replace(raw_dir, save_dir)
            cv2.imwrite(name + ".png", new_img)

        if "-ir" in ig:
            img = np.fromfile(ig, dtype=np.uint8)
            img_len = len(img)
            # if img_len< 1280*768:
            #     continue
            # img.shape = (1024, 768)
            img.shape = (1280, 768)
            name = ig.strip().split(".")[0]
            name = name.replace(raw_dir, save_dir)
            cv2.imwrite(name + ".png", img)
        else:
            pass

if __name__ == '__main__':
    raw_dir = '/home/tao/workspace/FaceDatas/testImage0125'
    decode_images(raw_dir)
