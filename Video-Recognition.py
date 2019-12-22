import cv2
import os
import numpy as num
  
face_cascade=cv2.CascadeClassifier('#LOCATE THE PATH WHERE YOUR HARR DETECTION ALGORITHM FOUND')

capture = cv2.VideoCapture(0)

while(True):
 	#Capture frame-by-frame
	rectangle, FRM = capture.read()
	imgc =cv2.cvtColor(FRM, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(imgc, scaleFactor=1.5, minNeighbors=5) 
       
	for(x , y, w, h) in faces:
	    print(x ,y ,w ,h ) 
 	    roi_imgc = imgc[y:y+h, x:x+w]
	    roi_color = FRM[y:y+h, x:x+w]
	    img = "#LOCATE THE PATHE WHERE YOUR CAPTURED IMAGE TO BE SET"
	    cv2.imwrite(img, roi_imgc)
            color = (55,255,55)
	    stroke = 2
	    end_cord_x = x + w
	    end_cord_y = y + h
	    cv2.rectangle(FRM,(x, y), (end_cord_x, end_cord_y),color,stroke) 


	#display the resulting frame
	cv2.imshow('Live Stream',FRM)
	if cv2.waitKey(20) & 0xFF == ord('S'):
	   break
#when everything is done, release the capture
capture.release()
cv2.destroyAllWindows()
