#import required libraries
#import OpenCV library
import cv2
#import matplotlib library
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import os
import numpy as np

subjects = ["", "Matt", "Mitch"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-source/data/lbpcascades/lbpcascade_frontalface_improved.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]
	
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    try:
	    label = face_recognizer.predict(face)
    except:
        print("nothing found")
        return img
 
    #get name of respective label returned by face recognizer
    print("label: " , label[0])
    label_text = subjects[label[0]]
    
    confidence = float(label[1])
    print(confidence)
    if confidence > 85:
        draw_rectangle(img, rect)
        draw_text(img, "not sure who this is", rect[0], rect[1]-5)
        return img
		
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    print("\n======================\n\nWE FOUND YOU!!\n\n========================\n")

    return img
		

########################################

lbp_face_cascade = cv2.CascadeClassifier('opencv-source/data/lbpcascades/lbpcascade_frontalface_improved.xml')

face_recognizer.read('data/model.yml')

#test_img1 = cv2.imread("untrained-data/image1.jpg")
#test_img2 = cv2.imread("untrained-data/image2.jpg")
#test_img3 = cv2.imread("data/smitch-images/mitch_image1.jpg")
#predicted_img1 = predict(test_img1)
#predicted_img2 = predict(test_img2)
#predicted_img3 = predict(test_img3)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
#ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))

#cv2.imshow("Matt test", predicted_img1)
#cv2.imshow("hugh jackman test", predicted_img2)
#cv2.imshow("mitch test", predicted_img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

