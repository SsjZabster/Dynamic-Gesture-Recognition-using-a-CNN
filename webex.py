import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import os

label = ['Doing something?', 'Drum Fingers', 'No gesture', 'Pull Hand In', 'Pull Two Fingers In', 'Push Hand Away', 'Push Two Fingers Away',
         'Roll Hand Backward', 'Roll Hand Forward', 'Shake Hand', 'Slide Two Fingers Down', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 
         'Slide Two Fingers Up', 'Stop Sign', 'Swipe Down', 'Swipe Left', 'Swipe Right', 'Swipe Up', 'Thumb Down', 'Thumb Up', 'Turn Hand Clockwise',
         'Turn Hand Anticlockwise', 'Zoom In With Full Hand', 'Zoom In With Two Fingers', 'Zoom Out With Full Hand', 'Zoom Out With Two Fingers']

#pred_data = [1.2439e-03, 1.9670e-03, 9.3795e-01, 1.1780e-03, 8.4933e-04, 4.7713e-04,
#        1.5490e-03, 5.9750e-05, 5.0666e-05, 1.7601e-03, 1.2124e-02, 6.3848e-04,
#        7.8682e-04, 7.7008e-03, 6.8628e-04, 4.8785e-03, 1.6826e-03, 1.4956e-03,
#        2.0409e-03, 5.9328e-04, 4.7968e-03, 5.5651e-05, 1.0579e-04, 9.1814e-04,
#        3.3939e-03, 3.5396e-03, 7.4737e-03]
#pred_data = np.multiply(np.round(pred_data, 4),100)
      
index = np.arange(len(label))

#os.stat("prediction.npy").st_size == 0

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,1,1)
plt.tight_layout()

plt.xlabel('Prediction %', fontsize=13)
plt.ylabel('No of Movies', fontsize=13)
#plt.xticks(, pred_data, fontsize=10, rotation=30)
plt.yticks(index, label, fontsize=13, rotation=30)
plt.title('Gesture Recognition Accuracy')

def animate(i):
    #pullData = open("sampleText.txt","r").read()
    #dataArray = pullData.split('\n')
    #xar = []
    #yar = []
    #for eachLine in dataArray:
    #    if len(eachLine)>1:
    #        x,y = eachLine.split(',')
    #        xar.append(int(x))
    #        yar.append(int(y))
    if(os.path.getsize("/home/zab/Desktop/TRN/prediction.npy") > 10):
        pred_data = np.load("prediction.npy",allow_pickle=True)
        pred_data = np.round(np.multiply(pred_data, 100),2)
        ax1.clear()
        plt.xlabel('Prediction %', fontsize=13)
        plt.ylabel('No of Movies', fontsize=13)
        #plt.xticks(, pred_data, fontsize=10, rotation=30)
        plt.yticks(index, label, fontsize=13, rotation=30)
        plt.title('Gesture Recognition Accuracy')
        plt.tight_layout()
        ax1.barh(index, pred_data)


ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()














'''import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


label = ['Doing other things', 'Drumming Fingers', 'No gesture', 'Pulling Hand In', 'Pulling Two Fingers In', 'Pushing Hand Away', 'Pushing Two Fingers Away',
         'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand', 'Sliding Two Fingers Down', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 
         'Sliding Two Fingers Up', 'Stop Sign', 'Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up', 'Thumb Down', 'Thumb Up', 'Turning Hand Clockwise',
         'Turning Hand Counterclockwise', 'Zooming In With Full Hand', 'Zooming In With Two Fingers', 'Zooming Out With Full Hand', 'Zooming Out With Two Fingers']

pred_data = [1.2439e-03, 1.9670e-03, 9.3795e-01, 1.1780e-03, 8.4933e-04, 4.7713e-04,
        1.5490e-03, 5.9750e-05, 5.0666e-05, 1.7601e-03, 1.2124e-02, 6.3848e-04,
        7.8682e-04, 7.7008e-03, 6.8628e-04, 4.8785e-03, 1.6826e-03, 1.4956e-03,
        2.0409e-03, 5.9328e-04, 4.7968e-03, 5.5651e-05, 1.0579e-04, 9.1814e-04,
        3.3939e-03, 3.5396e-03, 7.4737e-03]
#pred_data = np.multiply(np.round(pred_data, 4),100)
pred_data = np.round(np.multiply(pred_data, 100),2)
      
index = np.arange(len(label))





def barlist(n): 
    return [1/float(n*k) for k in range(1,28)]

fig=plt.figure(figsize=(10,15))
#plt.figure(figsize=(20,10))
plt.bar(index, pred_data)
plt.xlabel('Action', fontsize=14)
plt.ylabel('Prediction %', fontsize=14)
plt.xticks(index, label, fontsize=14, rotation=40, ha='right')
plt.title('Gesture Recognition Accuracy')



n=100 #Number of frames
x=label
#plt.figure(figsize=(10,5))
plt.bar(index, pred_data)
graph = plt.bar(x,pred_data)

def animate(i):
    y=pred_data
    for i in range(len(graph)):
        graph[i].set_height(y[i])

anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,
                             interval=100)

#anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
plt.show()'''
