# Base model of inception courtesy of Dinesh Palanisamy
# Website: dineshp.ai

# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
from threading import Thread
import argparse
import imutils
import cv2
#import pandas as pd
from random_word import RandomWords
import timeit
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import pyautogui


def putIterationsPerSec(frame, label):
    """ Add iterations per second text to lower-left corner of a frame. """

    cv2.putText(frame, "Gesture: " + label,
                (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame


# created a *threaded *video stream, allow the camera senor to warmup, and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
fps = FPS().start()

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torchvision
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from models import TSN
import transforms
from torch.nn import functional as F


def load_frames(frames, num_frames=8):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))


'''def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames'''


parser = argparse.ArgumentParser(description="test TRN on a single video")
# group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--video_file', type=str, default='')
parser.add_argument('--frame_folder', type=str, default='')
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='jester',
                    choices=['something', 'jester', 'moments', 'somethingv2'])
parser.add_argument('--rendered_output', type=str, default='test')
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weights', type=str,
                    default='pretrain/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')

args = parser.parse_args()

# Get dataset categories.
categories_file = 'pretrain/{}_categories.txt'.format(args.dataset)
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
net = TSN(num_class,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

checkpoint = torch.load(args.weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()

# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupScale(net.scale_size),
    transforms.GroupCenterCrop(net.input_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(net.input_mean, net.input_std),
])

label = ['Wait, say that again', 'Drumming Fingers', 'No gesture', 'Pull Hand In', 'Pull Two Fingers In', 'Push Hand Away', 'Push Two Fingers Away',
         'Roll Hand Backward', 'Roll Hand Forward', 'Shaking Hand', 'Slide Two Fingers Down', 'Slide Two Fingers Left', 'Slide Two Fingers Right', 
         'Slide Two Fingers Up', 'Stop Sign', 'Swipe Down', 'Swipe Left', 'Swipe Right', 'Swipe Up', 'Thumb Down', 'Thumb Up', 'Turn Hand Clockwise',
         'Turn Hand Anticlockwise', 'Zoom In With Full Hand', 'Zoom In With Two Fingers', 'Zoom Out With Full Hand', 'Zoom Out With Two Fingers']
index = np.arange(len(label))

'''def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.figure(figsize=(20,10))
    plt.bar(index, pred_data)
    plt.xlabel('Action', fontsize=14)
    plt.ylabel('Prediction %', fontsize=14)
    plt.xticks(index, label, fontsize=14, rotation=40, ha='right')
    plt.title('Gesture Recognition Accuracy')
    plt.show(block=False)
    #plt.show()

def animate(pred_data):
    #print(pred_data)
    plt.figure(figsize=(10,5))
    plt.bar(index, pred_data)
    plt.xlabel('Action', fontsize=14)
    plt.ylabel('Prediction %', fontsize=14)
    plt.xticks(index, label, fontsize=14, rotation=40, ha='right')
    plt.title('Gesture Recognition Accuracy')
    plt.show()
    #dataArray = pullData.split('\n')
    #xar = []
    #yar = []
    #for eachLine in dataArray:
    #    if len(eachLine)>1:
    #        x,y = eachLine.split(',')
    #        xar.append(int(x))
    #        yar.append(int(y))
    #ax1.clear()
    #ax1.plot(xar,yar)
fig = plt.figure()'''
pred = ""
bufferf = []
r = RandomWords()
ten_pred = []
mcount = 0##
start = timeit.default_timer()##
# loop over some frames...this time using the threaded stream
while True:
#for i in range(20):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    print('Classification #: ', mcount)##
    stop = timeit.default_timer()##
    print('Time: ', stop - start)##
    frame = vs.read()
    frame = putIterationsPerSec(frame, pred)

    # frame = cv2.resize(frame, (640, 480))

    crop_img = frame[0:720, 0:720]
    # h,w = crop_img.shape[:2]
    input_img = crop_img;
    input_pill = Image.fromarray(input_img)
    cv2.imshow("Frame", crop_img)
    key = cv2.waitKey(1) & 0xFF

    if (len(bufferf) < 16):
        bufferf.append(input_pill)
    else:
        input_frames = load_frames(bufferf)
        #print(input_frames)
        data = transform(input_frames)
        input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda(), volatile=True)

        # with torch.no_grad():
        logits = net(input)
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)
        np.save("prediction.npy", h_x.cpu())#, allow_pickle=True)
        #print(pred_data)
        #ani = FuncAnimation(fig, animate, interval=1000)
        #plt.show()What is the answer to life, the universe and everything?        
        pred = categories[idx[0]]
        #print(h_x)the preds
        ten_pred.append(pred)

        '''if(mcount>20 and mcount%16==0):
            #true_pred = max(set(ten_pred), key=ten_pred.count)
            if(pred == 'Push Hand Away'):
                pyautogui.moveTo(600, 75)
            elif(pred == 'Thumb Up'):
                pyautogui.press('enter')
            elif(pred == 'Thumb Down'):
                pyautogui.press('esc')
            elif(pred == 'Swipe Down'):
                pyautogui.moveRel(None,100)
            elif(pred == 'Swipe Up'):
                pyautogui.moveRel(None,-100)
            elif(pred == 'Swipe Left'):
                pyautogui.moveRel(-100,None)
            elif(pred == 'Swipe Right'):
                pyautogui.moveRel(100,None)
            elif(pred == 'Slide Two Fingers Down'):
                pyautogui.doubleClick()
            elif(pred == 'Pull Hand In'):
                #ten_pred = []
                #word = r.get_random_word(hasDictionaryDef="true")
                #if(ten_pred == []):
                pyautogui.typewrite('What is the answer to life, the universe and everything?')#word, interval=0.10)
                ten_pred = []
            ten_pred = []'''


        if(mcount>20 and mcount%16==0):
            true_pred = max(set(ten_pred), key=ten_pred.count)
            if(true_pred == 'Push Hand Away'):
                pyautogui.moveTo(600, 75)
            elif(true_pred == 'Thumb Up'):
                pyautogui.press('enter')
            elif(true_pred == 'Thumb Down'):
                pyautogui.press('esc')
            elif(true_pred == 'Swipe Down'):
                pyautogui.moveRel(None,100)
            elif(true_pred == 'Swipe Up'):
                pyautogui.moveRel(None,-100)
            elif(true_pred == 'Swipe Left'):
                pyautogui.moveRel(-100,None)
            elif(true_pred == 'Swipe Right'):
                pyautogui.moveRel(100,None)
            elif(true_pred == 'Slide Two Fingers Down'):
                pyautogui.doubleClick()
            elif(true_pred == 'Pull Hand In'):
                #ten_pred = []
                #word = r.get_random_word(hasDictionaryDef="true")
                #if(ten_pred == []):
                pyautogui.typewrite('What is the answer to life, the universe and everything?')#word, interval=0.10)
                ten_pred = []
            ten_pred = []        

        bufferf[:-1] = bufferf[1:];
        bufferf[-1] = input_pill
        # check to see if the frame should be displayed to our screen

    # update the FPS counter
    fps.update()
    mcount = mcount + 1##
    '''if(mcount>20 and mcount%10==0):
        true_pred = max(set(ten_pred), key=ten_pred.count)
        if(true_pred == 'Push Hand Away'):
            pyautogui.moveTo(600, 70)
        elif(true_pred == 'Slide Two Fingers Down'):
            pyautogui.doubleClick()
        elif(true_pred == 'Pull Hand In'):
            ten_pred = []
            #word = r.get_random_word(hasDictionaryDef="true")
            pyautogui.typewrite('What is the answer to life, the universe and everything?')#word, interval=0.10)
            ten_pred = []
        elif(true_pred == 'Thumb Up'):
            pyautogui.press('enter')
        ten_pred = []'''



