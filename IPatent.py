from IObjZoneDetect import IObjZoneYOLOV3Detect
from ICharRecognize import CharRecognize
import os
import cv2
import numpy as np
from PIL import Image

class IPatent:
    def __init__(self,model_dir, gpu_id=0):
        if not model_dir.endswith("/"):
            model_dir += "/"
        cfg_file = model_dir + "patent/yolov3patent.cfg"
        weight_file = model_dir + "patent/yolov3patent.weights"
        self.__detector = IObjZoneYOLOV3Detect(cfg_file,weight_file,gpu_id)

        char_weightfile = model_dir + "patent/crnn.pth"
        self.__recognizer = CharRecognize(char_weightfile,gpu_id)

    def recognize(self, im):
        boxes = self.__detector.detect(im)
        for box in boxes:
            zone = box["zone"]
            roi = im[zone[1]:zone[3],zone[0]:zone[2]]
            if box["cls"]==1:
                roi = np.rot90(roi,3)
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            cv2.imshow("im", roi)
            roi = Image.fromarray(roi)
            str1 = self.__recognizer.recognize(roi)
            print(str1)
            box["license"] = str1
            if cv2.waitKey(0)==0:
                break

        return boxes

if __name__=="__main__":
    model_dir = "/home/zqp/models/"
    detector = IPatent(model_dir)
    picdir = "/media/zqp/data/data/patent/VOCdevkit/VOC2007/JPEGImages/"

    for picname in os.listdir(picdir):
        im = cv2.imread(picdir+picname)

        boxes = detector.recognize(im)
