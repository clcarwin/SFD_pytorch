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
from bbox import decode, nms

def detect(net:nn.Module, img:np.ndarray, minscale:int=3):
    """Setting minscale to 0 finds the smallest faces, but takes the longest.
    """
    start_time = time.time()
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float()).cuda()
    BB,CC,HH,WW = img.size()
    olist = net(img)
    print(f"Running CNN took {1000*(time.time() - start_time):.1f}ms")

    bboxlist = []
    for i in range(len(olist)//2): 
        olist[i*2] = F.softmax(olist[i*2])
    for i in range(minscale, len(olist)//2):
        #print(f"Going through olist {i} at {1000*(time.time() - start_time):.1f}ms.  bboxlist has {len(bboxlist)} entries")
        ocls,oreg = olist[i*2].data,olist[i*2+1].data
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):  # Run a sliding window over the whole thing...
            windex,hindex = Findex%FW,Findex//FW
            score = ocls[0,1,hindex,windex]
            if score<0.05: 
                continue
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]]).cuda()
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]]).cuda()
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = torch.Tensor(bboxlist)
    if 0==len(bboxlist): bboxlist=torch.zeros((1, 5))
    return bboxlist

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
        if args.path=='CAMERA': ret, img = cap.read()
        else: img = cv2.imread(args.path)

        imgshow = np.copy(img)
        start_time = time.time()
        bboxlist = detect(net, img, 3)
        print(f"Running detect took {1000*(time.time() - start_time):.1f}ms")

        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep,:]
        for b in bboxlist:
            x1,y1,x2,y2,s = b
            if s<0.5: continue
            cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
        cv2.imshow('test',imgshow)

        if args.path=='CAMERA':
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            cv2.imwrite(args.path[:-4]+'_output.png',imgshow)
            if cv2.waitKey(0) or True: break
