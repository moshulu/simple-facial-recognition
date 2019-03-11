#import required libraries
#import OpenCV library
import cv2
#import matplotlib library
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import os
import numpy as np

subjects = ["", "Matt", "Mitch"]

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

def prepare_training_data(data_folder_path):
	dirs = os.listdir(data_folder_path)
	
	faces = []
	labels = []

	for dir_name in dirs:
		if(dir_name == "model.yml"):
			continue
		#format is <name>-images
		label = dir_name.split('-')[0]
		
		if label == "smatt":
			label = 1
		elif label == "smitch":
			label = 2
		else:
			label = 0
		
		subject_dir_path = data_folder_path + "/" + dir_name
		image_names = os.listdir(subject_dir_path)
		
		for image_name in image_names:
			print(image_name)
			image_path = subject_dir_path + "/" + image_name
			image = cv2.imread(image_path)
			#cv2.imshow("Training on image...", image)
			#cv2.waitKey(100)
			
			face, rect = detect_face(image)
			
			if face is not None:
				faces.append(face)
				labels.append(label)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()
	
	return faces, labels

########################################

lbp_face_cascade = cv2.CascadeClassifier('opencv-source/data/lbpcascades/lbpcascade_frontalface_improved.xml')

print("preparing data...")
faces, labels = prepare_training_data('data')
print("data prepared.")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.write('data/model.yml')
print("\ndata successfully trained.\n")





