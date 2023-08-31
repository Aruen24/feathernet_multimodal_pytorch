from __future__ import print_function
import os
import argparse
import torch
import FeatherNet
import torch.backends.cudnn as cudnn
import numpy as np
import cv2



parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='/home/disk01/wyw/feathernet_multimodal_pytorch/multimodal_rgb_ir_training_record_0424/model/0424_0737/Feathernet_160.pkl',
                    type=str, help='Trained state_dict file path to open')
#parser.add_argument('-m', '--trained_model', default='/home/disk04/YZhang/multimodal_training_record/model/1215_0955/Feathernet_91.pkl',
#                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--long_side', default=[224,224], help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('-img_size', type=int, help='the input size', default=224)
parser.add_argument('-class_num', type=int, help='class num', default=2)
parser.add_argument('-input_channels', type=int, help='the input channels', default=4)

args = parser.parse_args()


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


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # net = FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels,se=True, avgdown=True)
    net= FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels, is_training=False,se=True, avgdown=True)

    net = net.to(device)
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    net = net.to(device)
    print('Finished loading model!')

    # ------------------------ export -----------------------------
    #output_onnx = '/home/disk01/wyw/feathernet_multimodal_pytorch/liveness_multimode_ir.onnx'
    output_onnx = '/home/disk01/wyw/feathernet_multimodal_pytorch/liveness_multimode.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ['output0']
    #output_names = ['output0', 'output1', 'output2']
    inputs = torch.randn(1, args.input_channels, args.long_side[0], args.long_side[1]).to(device)

    # torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=True,
    #                                input_names=input_names, output_names=output_names,opset_version=11)
    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=True,
                                   input_names=input_names, output_names=output_names)

