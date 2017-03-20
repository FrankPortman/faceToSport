import cv2
import sys
import os

if not "Extracted" in os.listdir("."):
    os.mkdir("Extracted")
    
if len(sys.argv) < 2:
    print "Usage: python detectFace.py 'image path'"
    sys.exit()
    
image_path=sys.argv[1]
cascade="faceCascade.xml"

face_cascade=cv2.CascadeClassifier(cascade)

image=cv2.imread(image_path)
image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image_grey,scaleFactor=1.1,minNeighbors=5,minSize=(130,130),flags=0)

for x,y,w,h in faces:
    sub_img=image[y-40:y+h+10,x-40:x+w+40]
    os.chdir("../imgs/Extracted")
    cv2.imwrite(os.path.basename(image_path)+"-extracted.png",sub_img)
    os.chdir("../")


