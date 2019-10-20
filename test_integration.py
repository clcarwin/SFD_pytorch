import cv2
import numpy as np
import pytest
import torch

from detect_faces import detect_faces
from net_s3fd import S3fd_Model

def test_ellen_selfie():
    model = S3fd_Model()
    try:
        state_dict = torch.load("s3fd_convert.pth")
        model.load_state_dict(state_dict)
    except:
        print("Failed to load pre-trained model for test")
        raise
    model.cuda()
    model.eval()
    with torch.no_grad():
        img = cv2.imread('data/test01.jpg')
        faces = detect_faces(model, img)
    assert len(faces) == 11
