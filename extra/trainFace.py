from email.mime import image
import os, cv2,  numpy as np
from PIL import Image

imgPath = "./faces/"
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# mendapatkan label (id gambar)
def getImageLabel (pathImg) -> list : 
    imagePaths = [os.path.join(pathImg, f) for f in os.listdir(pathImg)]
    faceSamples = []
    faceIDs = []
    i = 1
    for imagePath in imagePaths :
        PILImg = Image.open(imagePath).convert("L")
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        print(f"id Wajah {faceID}, ke : {i}")
        i += 1
        for (x,y,w,h) in faces :
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)
    return faceSamples, faceIDs

faceRecog = cv2.face.LBPHFaceRecognizer_create()

# Training
print("start training data ....")
faces, IDs = getImageLabel(imgPath)
faceRecog.train(faces, np.array(IDs))
# save model training
faceRecog.write('./data/and_faces.xml')