# FaceRecognition using OpenCV

You need to import all the files that are provided in this repository
1) Images
2) attendance.csv
3) hackathon.py

# Images folder
Now Images folder contains images of various celebrities which can be considered as a training dataset.
You can add your own images in this Images folder.

# hackathon.py file
hackathon.py file is the main python file consisting of the python code that is required to execute this program.
The moment you will run this file, the webcam will automatically capture the image of a person in front of it.
If the webcam will not detect any face then it will wait for a person to be in front of it.
Now if you show an image to the webcam or maybe a person sits in front of the camera, it will detect the face and
display the name of person based on the training data repository (i.e Images folder).

# attendance.csv file
The attendance.csv file will be used to detect the Name and Time at which the image was captured ensuring that the attendance is marked.

# Libraries required
1) cv2 (Opencv-python)
2) face-recognition
3) datetime
4) numpy
5) pandas
6) os

# How you can use this project
You can use this project either for only face detection and recognition or for attendance purpose too.
I have already mentioned some comments in the python file so that you will get to know about the execution 
of each and every line in the program.
