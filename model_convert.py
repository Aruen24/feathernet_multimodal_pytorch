import os
import sys
import argparse
import numpy as np
import torch
import onnx
from pytorch import FeatherNet
from skimage import io
sys.path.append('/home/tao/picture/LCW/liveness_detection/face_align')
from MobilenetV2 import  My_Function_lib
os.environ['CUDA_VISIBLE_DEVICES']='-1'

def estimate(predictions, labels):
    from sklearn.metrics import confusion_matrix
    y_true = labels
    y_pred = predictions
    Mat_result = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sample_num = len(labels)
    acc = np.trace(Mat_result) / sample_num
    FAR = sum(iter(Mat_result[1:,0].reshape(-1,))) / (sum(iter(Mat_result[1:,:].reshape(-1,)))+ 1e-5)
    FRR = sum(iter(Mat_result[0][1:3])) / sum(iter(Mat_result[0][:]))
    Act_acc = (np.trace(Mat_result) + Mat_result[1][2] + Mat_result[2][1]) / sample_num
    result = np.hstack((FAR, FRR, acc, Act_acc))
    return result

def get_imgpath():
    dir_path='/home/data01_disk/lcw/code/img_data/snap/face'
    imgs=os.listdir(dir_path)
    imgs=list(filter(lambda x : x.endswith(('png')),imgs))
    imgs.sort()
    img_paths=[os.path.join(dir_path,img)for img in imgs]
    txt=dir_path+'/face.txt'
    f=open(txt,'w')
    for img_path in img_paths:
        f.write(img_path)
        f.write('\n')
def test(argvs):
    sys.path.append('/home/tao/picture/LCW/liveness_detection/face_align')
    from MobilenetV2 import utilizes
    Scence_data = utilizes.load_all_txts(argvs.test_scence_data)
    Face_data = utilizes.load_all_txts(argvs.test_face_data)

    label_true = '/media/tao/500G/test/class_false1/pc_T'
    label_false = '/media/tao/500G/test/class_false1/pc_F'
    data_num = len(Scence_data)
    print('data num:%d' % (data_num))
    # if not os.path.exists(label_false):
    #     os.makedirs(label_false)
    # if not os.path.exists(label_true):
    #     os.makedirs(label_true)
    pb_path = argvs.pb_path
    Input, Output, sess = My_Function_lib.call_PbModel(pb_path, argvs.In_node, argvs.Out_node)
    total_Pre_class = []
    total_labels = []
    total_predictions = []

    scence_img_paths, labels = utilizes.decode_img_data(Scence_data)
    test_labels = np.array(labels)
    test_data = np.transpose(np.vstack((scence_img_paths,Face_data,test_labels)))
    i=1
    for inf in test_data:
        scence_img = io.imread(inf[0])
        face_img=io.imread(inf[1])
        label=int(inf[2])
        scence_img = scence_img[:, :, :3]
        scence_img = (scence_img - np.mean(scence_img)) / np.std(scence_img)
        face_img = face_img[:, :, :3]
        face_img = (face_img - np.mean(face_img)) / np.std(face_img)
        img = np.concatenate((scence_img, face_img), 2)
        img = img.transpose(2, 0, 1).reshape(1,6,224,224)
        feed_dict1 = {Input: img}
        predictions = sess.run(Output, feed_dict=feed_dict1)
        preds_index = np.argmax(predictions, 1)
        total_Pre_class.append(preds_index)
        total_labels.append(label)
        total_predictions.append(predictions)
        if i%100==0:
          print('process:%d'%i)

        i+=1
    total_Pre_class=np.array(total_Pre_class)
    total_labels=np.array(total_labels)
    result = estimate(total_Pre_class, total_labels)

    # save_false_class_img(label_true,label_false,acc,labels,predictions,batch)
    print('False Rejection Rate (FRR) is {}'.format(result[1]))  # 拒认率
    print('False Acceptance Rate(FAR) is {}'.format(result[0]))  # 误识率
    print("test_acc:{:.5f}".format(result[2]))
    print('Entire_Acc rata {:.5f}'.format(result[3]))

def convert_onnx_pb(args):
    save_dir='/home/data01_disk/lcw/code/train_save_file/model/0903_0332/model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    onnx_path=os.path.join(save_dir,'feathernet_440.onnx')
    pb_path=os.path.join(save_dir,'feathernet_440.pb')
    pb_txt_path= os.path.join(save_dir,'feathernet_440.txt')

    #-------config the model--------
    device = torch.device('cpu')
    net = FeatherNet.FeatherNet(num_class=3, input_size=224, input_channels=6, se=True, avgdown=True)
    # import jimukeji_net
    # net = jimukeji_net.Net(num_class=3, input_channels=6)
    net=net.to(device)
    net.eval()
    model_dict=net.state_dict()
    map_location = lambda storage, loc: storage
    checkpoint = torch.load(args.model_path,map_location=map_location)
    start_epoch = checkpoint['epoch']
    state_dict=checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    for k,v in new_state_dict.items():
        if k in model_dict:
            model_dict[k]=v
        if k=='logits.1.weight':
            model_dict['predictions.0.weight']=v
        # if k=='logits.1.bias':
        #     model_dict['predictions.0.bias']=v

    net.load_state_dict(new_state_dict)

    #-------convert onnx-------
    input_names = ["Input0"]
    output_names = ["Output0"]
    x=torch.randn(440,6,224,224)
    # torch_out = torch.onnx._export(net, x, onnx_path, export_params=True, verbose=False,
    #                                 input_names=input_names, output_names=output_names)
    torch_out = torch.onnx._export(net, x, onnx_path, export_params=True, verbose=True,training=False)

    #-------convert pb-------
    sys.path.append('/home/data01_disk/lcw/code/onnx-tensorflow-master/build/lib')
    from onnx_tf.backend import prepare
    model = onnx.load(onnx_path)
    tf_rep =prepare(model,strict=False)
    tf_rep.export_graph(pb_path)
    My_Function_lib.pb_convert_pbtxt(pb_path, pb_txt_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    '''test data'''
    parser.add_argument('-test_scence_data', type=str, help='data for testing',
                        default='/home/data01_disk/lcw/code/img_data/snap/scence')
    parser.add_argument('-test_face_data', type=str, help='data for testing',
                        default='/home/data01_disk/lcw/code/img_data/snap/face')
    parser.add_argument('-model_path', type=str, default='/home/data01_disk/lcw/code/train_save_file/model/0903_0332/Feathernet_88.pkl')
    parser.add_argument('-pb_path',type=str,default='/home/data01_disk/lcw/code/train_save_file/model/0903_0332/model/feathernet.pb')

    parser.add_argument('--In_node',type=str,default='0:0')    #IteratorGetNext:0
    parser.add_argument('--Out_node', type=str, default='Softmax:0') #Mobilenet_V2/Logits/Predictions/Reshape_1:0
    argv=parser.parse_args()
    return argv

if __name__ == '__main__':
    #convert_onnx_pb(parse_arguments())
    test(parse_arguments())
    #get_imgpath()