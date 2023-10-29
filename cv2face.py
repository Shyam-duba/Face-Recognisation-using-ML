import cv2
import numpy as np
import os

# first start the web cam and get the data from it default '0' for system webcams
cap = cv2.VideoCapture(0)

# load the classifier here we are detecting faces so we are using haarcascade_frontalface
detector = cv2.CascadeClassifier("C:\\Users\\shyam\\Downloads\\haarcascade_frontalface_default (1).xml")

# images for storing faces
images = []
name  = input("enetr your name :")

# storing respected names for the faces
names = []
while True:
    ret, frame = cap.read() # read the image from the cam returns true if possible else false

    if ret :
        faces = cv2.detectMultiScale(frame) # detects the faces in the frame

        for face in faces:
            x,y,w,h = face

            # extracting the face from the frame
            cut_face = frame[y:y+h,x:x+w]

            # reshaping the face to the same size for all of the samples
            shaped_cut_face = cv2.resize(cut_face,(100,100))

            #gray scaling the picture to make it an 1-d array
            gray_face = cv2.cvtColor(shaped_cut_face,cv2.COLOR_BGR2GRAY)


            cv2.imshow("window",gray_face)

            #face_array = np.array(gray_face).flatten()
            #print(face_array)



    key = cv2.waitKey(30)

    if key == ord('q'):
        break

    if key == ord('c') : # c means take the image and add it to the datset for whuich the model is going to train on
        images.append(gray_face.flatten())
        names.append([name])

file = 'facerecognisation.npy'

X = np.array(images)
y = np.array(names)

data = np.hstack([y,X]) # adding the x and y coloumn wise
if os.path.exists(file): # returns true if the file exists
    old = np.load(file)
    data = np.vstack([old,data]) # adding the new row horizontally to the dataset

np.save(file,data) # saving the data to the file
cap.release()
cv2.destroyAllWindows()



