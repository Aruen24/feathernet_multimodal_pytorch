import os
import numpy as np


# deal rgb+ir data   根据文件夹路径写到txt文件中
# ./casia_surf_train\fake\CLKJ_AS0005\101-color.jpg ./casia_surf_train\fake\CLKJ_AS0005\101-ir.jpg 0
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


def convert_file_path_to_txt(data_path,txtfile):
    with open(txtfile, 'w') as f:
        data_lines = []
        negs = 0
        posts = 0
        imgs = image_paths_list(data_path)
        for p_index, p_name in enumerate(imgs):
            # print(p_name)
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
                #data_lines.append((ir_path, rgb_path, label))
                f.write(str(rgb_path) + " " + str(ir_path) + " " + str(label) + '\n')
        #print('total sample is %d, postive sample is %d, negative sample is %d' % ((posts + negs), posts, negs))
        #return data_lines



convert_file_path_to_txt("0428_new_test", "0428_new_test.txt")
