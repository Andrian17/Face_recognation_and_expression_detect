import cv2, numpy as np

# TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Membuat model
model = Sequential()
# Setup Model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('./data/face_expression_detect.h5')

# mencegah penggunaan openCL dan pesan logging yang tidak perlu
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
# membuat kamus yang memberikan setiap label emosi (urutan abjad)
emotion_dict = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Netral", 5: "Sedih", 6: "Terkejut"}

# Face Recognition 
faceRecog = cv2.face.LBPHFaceRecognizer_create()
# membaca model faceRecog
faceRecog.read("./data/my_faces.xml")

# open camera
camera = cv2.VideoCapture(0)

# dbName
# List/array ini menyesuaikan dengan Id Gambar yand direcord di file recordFile.py, Jika Id gambarnya 1 maka "Andrian" akan ditampilkan
names = ["Unknown", "Andrian", "Cimen"]

while True :
    statusCam, frameCam = camera.read()
    frameCam = cv2.flip(frameCam, 1)

    if not statusCam:
        break

    # Haarcascade
    faceCascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    grayFrame = cv2.cvtColor(frameCam, cv2.COLOR_BGR2GRAY)
    faceDetect = faceCascade.detectMultiScale(grayFrame, 1.2, 5)

    for (x,y,h,w) in faceDetect :
        frameCam = cv2.rectangle(frameCam, (x,y), (x+w, y+h), color=(255, 82, 91), thickness=2)
        roy_gray = grayFrame[y:y + h, x:x + w]

        # face expression detect
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roy_gray, (48,48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxIndex = int(np.argmax(prediction))

        # face_recognition detect
        id, confidence = faceRecog.predict(roy_gray) # confidence 0 = sempurna
        if (100 - confidence) > 50 :
            nameId = names[id]
            confidenceTxt = f"{round(100-confidence)}%"
        else :
            nameId = names[0]
            confidenceTxt = f"{round(100-confidence)}%"

        # message
        cv2.putText(frameCam, nameId, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frameCam, confidenceTxt, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frameCam, emotion_dict[maxIndex], (x+20, y-35), cv2.FONT_HERSHEY_SIMPLEX, 1, (130, 130, 255), 2, cv2.LINE_AA)
    cv2.imshow("Me", frameCam)
    key = cv2.waitKey(1) & 0xff
    keyEsc = 27
    if key == keyEsc or key == ord('q') :
        break
# end
camera.release()
cv2.destroyAllWindows()