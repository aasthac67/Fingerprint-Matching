import cv2
import numpy as np

imgo = cv2.imread('./input/f1.jpeg')
cv2.imshow("imgo",imgo)

#Removing the background
height, width = imgo.shape[:2]

#Create a mask holder
mask = np.zeros(imgo.shape[:2],np.uint8)

#Grab Cut the object
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#Hard Coding the Rect… The object must lie within this rect.
rect = (10,10,width-30,height-30)
cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask = np.where((mask==2)|(mask==0),0,1).astype("uint8")
img1 = imgo*mask[:,:,np.newaxis]

#Get the background
background = cv2.absdiff(imgo,img1)

#Change all pixels in the background that are not black to white
background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

#Add the background and the image
final = background + img1

#To be done – Smoothening the edges….

cv2.imshow('image', final )
cv2.imwrite("input.jpg",final)

cv2.waitKey(0)
cv2.destroyAllWindows()