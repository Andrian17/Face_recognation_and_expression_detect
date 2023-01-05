from importlib.resources import path
import cv2, os

# membuat folder
imgPath = "./faces"
# if os.path(imgPath) :
idFace = int(input(f"Masukkan id wajah! (contoh: 1) : "))

# haarcasscade
faceCascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# open cam
cam = cv2.VideoCapture(0)
countImgFrom = 0
countImgTo = 60

while True :
    status, frameCam = cam.read()
    greyImg = cv2.cvtColor(frameCam, cv2.COLOR_BGR2GRAY)
    faceDetect = faceCascade.detectMultiScale(greyImg, 1.3, 5)
    for (x,y,h,w) in faceDetect :
        frameCam = cv2.rectangle(frameCam, (x,y), (x+w, y+h), (255,0,122), 1)
        imgName = f"face.{idFace}.{countImgFrom}.jpg"
        
        # save image
        if not os.path.isdir(imgPath) :
            os.makedirs(imgPath)
        cv2.imwrite(os.path.join(imgPath, imgName), frameCam)
        print(imgName)
    countImgFrom += 1
    key = cv2.waitKey(1) & 0xff
    keyEsc = 27
    if key == keyEsc or key == ord('q') :
        break
    elif countImgFrom  > countImgTo :
        break

cam.release()
cv2.destroyAllWindows()