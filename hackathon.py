#dependencies - cmake, dlib, face_recognition, numpy, openCV
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#3 steps are there
#loading images and converting into RGB
#as we are getting it in BGR but library understands as RGB
#we will get the encoding of the detected image and find that
# whether an image is there or not
path='Images'#path for our images
images=[]#list of all images that we will import
classNames=[]#we will create a list to name all our images so as to distinguish them
myList=os.listdir(path)
print(myList)#it will show the list of each image
for cls in myList:
    curImg=cv2.imread(f'{path}/{cls}')#imread function to read all images iteratively (cls stores the images)
    images.append(curImg)#append our curimg one by one
    classNames.append(os.path.splitext(cls)[0])#we want just the name not the extension so we will split it by retreiving only first element

print(classNames)#it will show image name without etension only name

def findEncodings(images):#it will find encodings for our image
    encodeList=[]#contain encodings of each image
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#converting BGR to RGB
        encode=face_recognition.face_encodings(img)[0]#finding the encodings
        # (A face encoding is basically a way to represent the face using a set of 128 computer-generated measurements)
        # for each image
        encodeList.append(encode)#appending encodings to our list
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+')as f:#read and write at the same time(r+)
        myDataList=f.readlines()#reading lines so that it will not enter value
                                # of same person detected
        nameList=[]#list of image/person
        for line in myDataList:
            entry=line.split(',')#splitting based on comma so that name and time are separated
            nameList.append(entry[0])#entering all names in this list
        if name not in nameList:#if name not in the list
            now=datetime.now()#check time when image detected
            dtString=now.strftime('%H:%M:%S')#format for time only
            f.writelines(f'\n{name},{dtString}')#writing name and time


encodeListKnown=findEncodings(images)#encoding complete
print('Encoding Complete')

cap=cv2.VideoCapture(0)#capturing the images from our web camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)#resizing so as to speed up the process(scale=0.25,0.25)( 1/4 the original size)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #facelocation in the image with 4 coordinates left right top bottom
    facesCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceloc, in zip(encodeCurFrame,facesCurFrame):
        # grab encoding from current
        # frame and face loction too
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        #comparing encodelistknown with encodeface
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #distnace of each one of them with the detected image
        #lowest the distance more perfect match it is
        matchIndex=np.argmin(faceDis)#finding lowest distance among
        # all the known images

        if matches[matchIndex]:#if there is a perfect match then,
            name=classNames[matchIndex].upper()#displaying the name of the image
            #print(name)
            y1,x2,y2,x1=faceloc#location of the image
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4#as we have scaled down our image
            # so we need
            # to undo that operation by multiplying the values by 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)#drawing rectangle
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #displaying text on rectangle
            markAttendance(name)
        else:
            name="Unknown"
            #name = classNames[matchIndex].upper()  # displaying the name of the image
            # print(name)
            y1, x2, y2, x1 = faceloc  # location of the image
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # as we have scaled down our image
            # so we need
            # to undo that operation by multiplying the values by 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # drawing rectangle
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

