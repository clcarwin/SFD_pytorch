from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime,math
import argparse
import numpy as np

import scipy.io as sio
import zipfile
from net_s3fd import s3fd
from bbox import *


def detect(net,img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    BB,CC,HH,WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)/2): olist[i*2] = F.softmax(olist[i*2])
    for i in range(len(olist)/2):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

def flip_detect(net,img):
    img = cv2.flip(img, 1)
    b = detect(net,img)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist

def scale_detect(net,img,scale=2.0,facesize=None):
    img = cv2.resize(img,(0,0),fx=scale,fy=scale)
    b = detect(net,img)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = b[:, 0]/scale
    bboxlist[:, 1] = b[:, 1]/scale
    bboxlist[:, 2] = b[:, 2]/scale
    bboxlist[:, 3] = b[:, 3]/scale
    bboxlist[:, 4] = b[:, 4]
    b = bboxlist
    if scale>1: index = np.where(np.minimum(b[:,2]-b[:,0]+1,b[:,3]-b[:,1]+1)<facesize)[0] # only detect small face
    else: index = np.where(np.maximum(b[:,2]-b[:,0]+1,b[:,3]-b[:,1]+1)>facesize)[0] # only detect large face
    bboxlist = b[index,:]
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

wider_face_mat = sio.loadmat('./eval/wider_face_val.mat')
event_list = wider_face_mat['event_list']
file_list = wider_face_mat['file_list']

save_path = './eval/sfd_val_pytorch/'
dataset = '../../dataset/face/WIDER/WIDER_val.zip'
datazip = zipfile.ZipFile(dataset)

net = s3fd()
net.load_state_dict(torch.load('data/s3fd_convert.pth'))
net.cuda()
net.eval()

# for i in range(1000):
#     size = 1024+64*i; print(size)
#     detect(net,np.zeros((size,size,3)))

for index, event in enumerate(event_list):
    filelist = file_list[index][0]
    im_dir = event[0][0].encode('utf-8')
    if not os.path.exists(save_path + im_dir): os.makedirs(save_path + im_dir)

    for num, file in enumerate(filelist):
        im_name = file[0][0].encode('utf-8')
        zipname = '%s/%s.jpg' % (im_dir,im_name)
        
        data = np.frombuffer(datazip.read('WIDER_val/images/'+zipname),np.uint8)
        img = cv2.imdecode(data,1)

        imgshow = np.copy(img)
        b1 = detect(net,img)
        b2 = flip_detect(net,img)
        if img.shape[0]*img.shape[1]*4>3000*3000: b3 = np.zeros((1, 5))
        else: b3 = scale_detect(net,img,scale=2,facesize=100)
        b4 = scale_detect(net,img,scale=0.5,facesize=100)
        bboxlist = np.concatenate((b1,b2,b3,b4))

        keep = nms(bboxlist,0.3)
        keep = keep[0:750] # keep only max 750 boxes
        bboxlist = bboxlist[keep,:]

        # for b in bboxlist:
        #     x1,y1,x2,y2,s = b
        #     if s<0.5: continue
        #     cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
        # cv2.imshow('',imgshow)
        # cv2.waitKey(0)
        # continue

        f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
        f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir,im_name)))
        f.write('{:d}\n'.format(len(bboxlist)))
        for b in bboxlist:
            x1,y1,x2,y2,s = b
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1,y1,(x2-x1+1),(y2-y1+1),s))
        f.close()
        print('event:%d num:%d' % (index + 1, num + 1))