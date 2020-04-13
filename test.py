from planer import read_onnx
import os
import cupy
import planer
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pal = planer.core(cupy)
net = read_onnx('re18')

# input should be float32


# the same folder should contain resnet18.txt, resnet18.npy
x = pal.random.randn(1, 3, 512, 512).astype('float32')
# y = net(x)
start = time.time()
for i in range(100):
    y = net(x)
print((time.time()-start)/100)
# net.show()
all_output_shape = [i.shape for i in y]
print(all_output_shape)
