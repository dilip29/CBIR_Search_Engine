from colordescriptor import ColorDescriptor
import cv2
import glob
import csv
cd=ColorDescriptor((2,2,2))
path="Dataset/"
images_list=[]
output_file=open("features_index.csv","w")

for file in glob.glob(path+"*.png"):
    imageID=file[file.rfind("/")+1:]
    image=cv2.imread(file)

    features=cd.describeImage(image)
    features=[str(f) for f in features]
    output_file.write("%s,%s\n"%(imageID,",".join(features)))

output_file.close()
