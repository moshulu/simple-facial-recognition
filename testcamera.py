
import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import recognize as rec
import time
import matplotlib.pyplot as plt

camera = PiCamera()
camera.framerate = 32
camera.resolution = (1024,768)
rawCapture = PiRGBArray(camera, size=(1024,768))

time.sleep(0.1)

lbp_face_cascade = cv2.CascadeClassifier('opencv-source/data/lbpcascades/lbpcascade_frontalface.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('data/model.yml')

for frame in camera.capture_continuous(rawCapture, format="bgr"):
	image = frame.array

	image = np.array(image)

	if image is None:
		continue
		
	image = rec.predict(image)

	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) &0xFF

	rawCapture.truncate(0)

	if key == ord("q"):
		break



#test_img1 = cv2.imread("untrained-data/image1.jpg")
#test_img2 = cv2.imread("untrained-data/image2.jpg")
#test_img3 = cv2.imread("untrained-data/image3.jpg")
#test_img4 = cv2.imread("untrained-data/image4.jpg")
#test_img5 = cv2.imread("untrained-data/image5.jpg")
#test_img6 = cv2.imread("untrained-data/image6.jpg")
#predicted_img1 = rec.predict(test_img1)
#predicted_img2 = rec.predict(test_img2)
#predicted_img3 = rec.predict(test_img3)
#predicted_img4 = rec.predict(test_img4)
#predicted_img5 = rec.predict(test_img5)
#predicted_img6 = rec.predict(test_img6)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

#ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))

#cv2.imshow("Matt test", predicted_img1)
#cv2.imshow("hugh jackman test", predicted_img2)
#cv2.imshow("mitch test", predicted_img3)
#cv2.imshow("test", predicted_img4)
#cv2.imshow("johnnydepp test", predicted_img5)
#cv2.imshow("elon test", predicted_img6)
cv2.waitKey(0)
cv2.destroyAllWindows()
