import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('Locate the captured image to be detected')
faces_detected,gray_img=fr.faceDetection(test_img)
print("The detected face:",faces_detected)

for (x,y,w,h) in faces_detected:
   cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)
   
#resized_img=cv2.resize(test_img,(1000,700))
#cv2.imshow("Internship Project in DDU",resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows

faces,faceID=fr.labels_for_training_data('Locate Path of the folder where we train our images')
face_recognizer=fr.train_classifier(faces,faceID)
name={0:"Unlicenced",1:"Licenced"}

for face in faces_detected:
	(x,y,w,h)=face
	roi_gray=gray_img[y:y+h,x:x+h]
	lable,confidence=face_recognizer.predict(roi_gray)
	print("confidence:",confidence)
	print("lable:",lable)
	fr.draw_rect(test_img,face)
	predicted_name=name[lable]
	fr.put_text(test_img,predicted_name,x,y)


resized_img=cv2.resize(test_img,(600,400))
cv2.imshow("Final Project in ASTU",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

