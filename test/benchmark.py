#!/usr/bin/env python

import caffe
import numpy as np
from cv2 import *

caffe.set_mode_cpu()

net = caffe.Net('../../../../models/DeepMARCaffe/DeepMAR.prototxt',
                '../../../../models/DeepMARCaffe/DeepMAR.caffemodel',
                caffe.TEST)

img = imread('CAM01_2014-02-15_20140215161032-20140215162620_tarid0_frame218_line1.png')
img = resize(img, (227, 227))
img = np.array(img, dtype=np.float32)
img -= 128
img /= 256

blob = np.zeros((1, 227, 227, 3), dtype=np.float32)
blob[0, 0:img.shape[0], 0:img.shape[1], :] = img
channel_swap = (0, 3, 1, 2)
blob = blob.transpose(channel_swap)

forward_kwargs = {'data': blob}
net.forward(**forward_kwargs)
output = net.blobs['fc8']
print output.data[0][0:1024]

