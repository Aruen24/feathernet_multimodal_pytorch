#source activate py36_torch

#FeatherNet_m.py中修改block 中Relu6为h-swish 84、95、101行
#FeatherNetB中将SE换成CBAM

#train 修改trian_rgb_ir_liveness.py中8、153行看是否用修改后还是修改前feathernet网络   159-162行选择对应的训练测试数据加载
#import FeatherNet
#import FeatherNet_m
#训练rgb和ir数据，输入为四通道,训练数据可以加载文件夹，也可以加载txt文件路径
python trian_rgb_ir_liveness.py   #训练代码163行指定gpu
python trian_rgb_ir_liveness_cbam.py

#训练depth和ir数据，输入为二通道
python trian_depth_ir_liveness.py
python trian_depth_ir_liveness_cbam.py

#只训练ir数据
python trian_ir_liveness.py

#原始的是训练depth+ir输入为2通道
#现在是rgb+ir输入为四通道
#修改trian_rgb_ir_liveness.py或者train_rgb_ir_liveness_cbam.py中28行default改为4


# pytorch 1.7.1版本训练报异常，降到1.6.0训练正常


#test
python evaluate_3dliveness_datasets.py  #测试部分代码未作修改
python evaluate_3dliveness_rgb_ir_datasets.py
python evaluate_3dliveness_datasets_cbam.py  #测试部分代码未作修改
python evaluate_3dliveness_rgb_ir_datasets_cbam.py

python evaluate_3dliveness_ir_datasets.py  #测试ir数据

#onnx转换,转换后得到的onnx模型做python -m onnxsim后会多出tile算子，在君正上转换模型失败，可能是onnx-simplifier版本原因，君正平台上转换模型最好不要做优化，转换工具内部也都会优化图结构
python convert_to_onnx.py
python convert_to_onnx_multimode.py

