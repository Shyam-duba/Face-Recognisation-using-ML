import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = np.load("C:\\Users\\shyam\\PycharmProjects\\face_print\\facerecognisation.npy") # dataset contains 10001 columns oth coloum represents class label and rest of the coloumns represent image

X = data[:, 1:].astype(int) # extrcating image features
y = data[:, 0] # class label

Knn = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

Knn.fit(X_train,y_train)

print(Knn.score(X,y))
print(Knn.predict(X_test))

# first start the web cam and get the data from it default '0' for system webcams
cap = cv2.VideoCapture(0)

# load the classifier here we are detecting faces so we are using haarcascade_frontalface
detector = cv2.CascadeClassifier("C:\\Users\\shyam\\Downloads\\haarcascade_frontalface_default (1).xml")


while True:
    ret, frame = cap.read() #read the image from the cam returns true if possible else false

    if ret :
        faces = detector.detectMultiScale(frame) # detects the faces in the frame

        for face in faces : # do something for everey face
            x,y,w,h = face

            face_cut = frame[y:y + h, x:x+w] # extracting the face from the whole frame

            shaped_face_cut = cv2.resize(face_cut,(100,100)) # reshaping it to limited no of features

            gray_face_cut = cv2.cvtColor(shaped_face_cut,cv2.COLOR_BGR2GRAY) # gray scaling the image bcz shape of the gray scled image is 2-d array

            x = np.array(gray_face_cut.flatten()) # falttening the 2-d array each index implies each feature

            name = Knn.predict([x]) # making the prediction of the person's  name by the face

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0)) # ploting a rectangle in the frame for the face only

            cv2.putText(frame,str(name[0]),(x+10,y-10),cv2.FONT_ITALIC,2,(255,0,0),2) #displaying the predicted name on the top of the
                                                                                                                 # rectangle


        cv2.imshow("My window",frame) #now showing the frame in the window
    key = cv2.waitKey(1)

    if key == ord('q'): # waiting for the response from the Keyboard
        break

cap.release()
cv2.destroyAllWindows()



