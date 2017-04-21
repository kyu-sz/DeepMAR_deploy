#!/usr/bin/env python

import caffe
import numpy as np
from cv2 import *
import time

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

start = int(round(time.time() * 1000))
net.forward(**forward_kwargs)
end = int(round(time.time() * 1000))

output = net.blobs['fc8']
print output.data[0][0:1024]

print 'Cost time: {}ms'.format(end - start)

