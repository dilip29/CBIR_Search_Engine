from colordescriptor import ColorDescriptor
import cv2
import glob
import csv
import pandas as pd

cd=ColorDescriptor((6,8,3))
path="Dataset/"
images_list=[]

output_file=open("features_index.csv","w")
#df=pd.read_csv("features_index.csv")
#rows,cols=df.shape


for file in glob.glob(path+"*.jpg"):
    imageID=file[file.rfind("/")+1:]
    image=cv2.imread(file)

    features=cd.describeImage(image)
    print(imageID," processed ")
    features=[str(f) for f in features]
    output_file.write("%s,%s\n"%(imageID,",".join(features)))

output_file.close()

#print("Rows=",rows,"  Cols=",cols)

