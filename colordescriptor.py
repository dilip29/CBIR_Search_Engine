import numpy as np
import cv2

class ColorDescriptor:
    def __init__(self,bins):
        self.bins=bins      #store no. of bins(or no. of steps ) used in 3D histogram, for larger dataset higher bins is preferable

    def describeImage(self,image):
        #convert the image file from BGR to HSV (Hue , Saturation, value) which is more perceivable to humans
        image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        # dimensions to calculate the center of image
        (height,width)=image.shape[:2]
        (cX,cY)=(int(width*0.5),int(height*0.5))

        features=[]
        # divide the image into 4 rectangular segements i.e (top-left,top-right,bottom-right,bottom-left)
        segments=[(0,cX,0,cY),(cX,width,0,cY),(cX,width,cY,height),(0,cX,cY,height)]

        # construct  a elliptical mask at the center cX,cY of the image
        (axesX,axesY)=(int(width*0.75)/2,int(height*0.75)/2)

        # ellipMask a empty matrix of same size of image
        ellipMask=np.zeros(image.shape[:2],dtype=np.uint8)
        cv2.ellipse(ellipMask,(cX,cY),(int(axesX),int(axesY)),0,0,360,255,-1)

        for (startX,startY,endX,endY) in segments:
            cornerMask=np.zeros(image.shape[:2],dtype=np.uint8)
            cornerMask=cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
            # subtract the elliptical mask from each rectangular mask to obtain required area for analysis
            cornerMask=cv2.subtract(cornerMask,ellipMask)

            hist=self.histogram(image,cornerMask)
            features.extend(hist)


        # add the features for elliptical mask also to the features list

        hist=self.histogram(image,ellipMask)
        features.extend(hist)


        return features


    def histogram(self,image,mask):
        # obtaining the 3D histogram fro the masked area of image using the no. of bins for each HSV
        hist=cv2.calcHist([image],[0,1,2],mask,self.bins,[0,180,0,256,0,256])

        # normalise the histogram
        cv2.normalize(hist,hist)
        hist=hist.flatten()

        return hist

'''
image=cv2.imread("103100.png")
cd=ColorDescriptor((2,2,2))
print(len(cd.describeImage(image)))
'''
