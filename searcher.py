from colordescriptor import ColorDescriptor
import numpy as np
import csv
import cv2
class Searcher:
    def __init__(self,indexPath):
        self.indexPath=indexPath    #indexPath is the path of the features_index.csv file
                                    #which holds the features vectors

    def search(self,queryimage_Features,limit=15):
        # limit is the maximum no. of similar images output for the query image
        # queryimage_Features is the feature vector for the queried image

        results={} # results is the dicitonary to store the results of searched images_list
        with open(self.indexPath) as file:
            reader=csv.reader(file)

            for row in reader:
                features=[float(i) for i in row[1:]]
                # obtain the chi squared distnace between the 2 histograms, smaller the chi squared distance
                # is similar the image is with the queried image
                distance=self.chi2_distance(features,queryimage_Features)
                print("Distance :",distance)


                results[row[0]]=distance
        file.close()
        # sort the results in ascending order so that relevant images are in the begining
        results=sorted([(value,key) for (key,value) in results.items()])



        return results[:limit]


    def chi2_distance(self,Histogram1,Histogram2,eps=1e-10):
        # defining the chi squared function
        distance=0.5*np.sum([ ( (a-b)**2) / (a+b+ eps) for (a,b) in zip(Histogram1,Histogram2)])
        print("Distance",distance)
        return distance
'''
cd=ColorDescriptor((2,2,2))
image=cv2.imread("101400.png")
features=cd.describeImage(image)
sd=Searcher("features_index.csv")
print(sd.search(features,10))
'''
