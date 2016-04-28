"""
    the program implements the sliding window algorithm to detect the face in a image 
"""
import cv,cv2
import mxnet as mx
import numpy as np
import os

def get_most_impossible(pres):
    max = 0
    result = [0,0]
    for pre in pres:
        if(max<pre[2]):
            max = pre[2]
            result = [pre[0],pre[1]]
    return result
def get_face(img,scale,result):
    cropImg = img[result[0]:result[0]+scale[0],result[1]:result[1]+scale[1]]
    cv2.imwrite(filename='single/face.jpg', img=cropImg)
def read_list(path_in):
    with open(path_in) as fin:
        for line in fin.readlines():
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            break
    return item

def write_one_img(s):
    record = mx.recordio.MXRecordIO('predict/predict.rec', 'w')
    #record.reset()
    record.write(s)
    record.close()   
    print '--------over-------ssdsfdfsd----'
    
def writeOneImg(item):
    img = cv2.imread('piredict/'+item[1])
    header = mx.recordio.IRHeader(0, item[2], item[0], 0)
    s = mx.recordio.pack_img(header, img)
    record = mx.recordio.MXRecordIO('piredict/piredict.rec', 'w')
    record.write(s)    

"""load the model"""
prefix = "data/lenetweights"
model = mx.model.FeedForward.load(prefix, epoch=1, ctx=mx.cpu())

#read the original picture
originalImg = cv2.imread(filename='single/test.jpg', flags=3)
#the scale of the patch
scale = (64,64)
#the height of the patch
patchHeight = scale[0]
#the weight of the patch
patchWeight = scale[1]
#the height of the original picture
originalHeight = originalImg.shape[0]
#the weight of the original picture
originalWeight = originalImg.shape[1]

count = 0
preds = []
record = mx.recordio.MXRecordIO('predict/predict.rec', 'w')
for i in range(originalHeight-patchHeight+1):
    for j in range(originalWeight-patchWeight+1):
        count +=1
        cropImg = originalImg[i:i+patchHeight,j:j+patchWeight]
        header = mx.recordio.IRHeader(0,0,0,0)#the head information of the picture.
        s = mx.recordio.pack_img(header, img=cropImg)
        record.reset()
        record.write(s)
        record.close()
        batch_size=1
        data_shape = (3,scale[0],scale[1])
        predict_data = mx.io.ImageRecordIter(
            path_imgrec = "predict/predict.rec",
            mean_img="predict/predict.bin",
            data_shape  = data_shape,
            batch_size  = 1)
        predictions = model.predict(predict_data)
        for prediction in predictions:
            print 'prediction'
            print prediction
            pre=np.argsort(prediction)[::-1]
            most_possible = pre[0]
            preds.append((i,j,prediction[most_possible]))
        record.open()
        #cv2.imwrite(filename='split/'+str(i)+'_'+str(j)+'.jpg', img=cropImg)
print 'completed!To detect the face,'+str(count)+' pictures are generated.'
point = get_most_impossible(pres=preds)
print point
get_face(originalImg, scale, result=point)
