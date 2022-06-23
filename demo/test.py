import time

import torch
from mmcv.tensorrt import TRTWraper

input_shape = (1, 3, 224, 224)
input_img = torch.randn(*input_shape)
input_img_cpu = input_img.detach().cpu().numpy()
input_img_cuda = input_img.cuda()

trt_file = 'pare.trt'
input_names = ['input.1']
output_names = ['3245', '3401', '3322', '3323']
# Get results from TensorRT
trt_model = TRTWraper(trt_file, input_names, output_names)
for i in range(100):
    T = time.time()
    for i in range(1000):
        with torch.no_grad():
            trt_outputs = trt_model({input_names[0]: input_img_cuda})
    print(time.time() - T)
    trt_outputs = [trt_outputs[_].detach().cpu().numpy() for _ in output_names]
