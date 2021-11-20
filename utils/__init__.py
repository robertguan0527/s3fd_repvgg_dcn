import csv
import os

import cv2

def csv_write(path,data):
    f = open(path, 'a') 
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def csv_writerows(path,data):
    f = open(path, 'w') 
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()

def dawrects(img, rects_gt,rects_pred):
    for box in rects_gt:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
    for box in rects_pred:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
    cv2.imshow("result",img)
    cv2.waitKey(0)



def write_pic(img, rects_gt,rects_pred,path,pic_name):
    for box in rects_gt:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
    for box in rects_pred:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),1)
    cv2.imwrite(os.path.join(path,pic_name),img)
    print(f"pic_{pic_name}_write sucess...")