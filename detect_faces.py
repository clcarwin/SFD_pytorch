import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple
torch.backends.cudnn.benchmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

import net_s3fd
from bbox import decode, nms

def detect_faces(net:nn.Module, img:np.ndarray, minscale:int=3, ovr_threshhold:float=0.3,
                 score_threshhold:float=0.5) -> List[Tuple]:
    """returns an list of tuples describing bounding boxes: [x1,y1,x2,y2,score].
    Setting minscale to 0 finds the smallest faces, but takes the longest.
    """
    bboxlist = detect(net, img, minscale)
    keep_idx = nms(bboxlist, ovr_threshhold)
    bboxlist = bboxlist[keep_idx,:]
    out = []
    for b in bboxlist:
        x1,y1,x2,y2,s = b
        if s<0.5: 
            continue
        out.append((int(x1),int(y1),int(x2),int(y2),s))
    return out


def detect(net:nn.Module, img:np.ndarray, minscale:int=3) -> torch.Tensor:
    """returns an Nx5 tensor describing bounding boxes: [x1,y1,x2,y2,score].
    This will have LOTS of similar/overlapping regions.  Need to call bbox.nms to reconcile them.
    Setting minscale to 0 finds the smallest faces, but takes the longest.
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
        olist[i*2] = F.softmax(olist[i*2], dim=1)
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
    if len(bboxlist) == 0: 
        bboxlist=torch.zeros((1, 5))
    bboxlist = torch.Tensor(bboxlist)
    return bboxlist

