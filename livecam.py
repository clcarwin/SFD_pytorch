import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

import net_s3fd
from detect_faces import detect_faces

parser = argparse.ArgumentParser(description='PyTorch face detect')
parser.add_argument('--net','-n', default='s3fd', type=str)
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--path', default='CAMERA', type=str)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


net = getattr(net_s3fd,args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()


if args.path=='CAMERA': 
    cap = cv2.VideoCapture(0)
with torch.no_grad():
    while(True):
        if args.path=='CAMERA': 
            ret, img = cap.read()
        else: 
            img = cv2.imread(args.path)

        imgshow = np.copy(img)
        start_time = time.time()
        bboxlist = detect_faces(net, img, 3)
        print(f"Running detect_faces took {1000*(time.time() - start_time):.1f}ms.  Found {len(bboxlist)} faces.")
        for b in bboxlist:
            x1,y1,x2,y2,s = b
            cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
        cv2.imshow('test',imgshow)

        if args.path=='CAMERA':
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            cv2.imwrite(args.path[:-4]+'_output.png',imgshow)
            if cv2.waitKey(0) or True: break
