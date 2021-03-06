from colordescriptor import ColorDescriptor
from searcher import Searcher
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import argparse
import numpy as np

def show_images(images, cols = 1):
    n_images = len(images)
    titles=[]
    titles.append("Query Image")
    for i in range(1,n_images + 1):
        titles.append('Result Image (%d)' % i)

    fig = plt.figure()
    for n,(image,title) in enumerate(zip(images,titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a.axis('off')
        plt.imshow(image)

        a.set_title(title)
    print(np.array(fig.get_size_inches()) * (n_images))
    fig.set_size_inches(np.array(fig.get_size_inches()) * (n_images))
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument(
  '--query',
  type=str,
  required=True,
  help="""
  Path to the query image
  """)
cmd_args, unparsed = parser.parse_known_args()

cd=ColorDescriptor((8,12,3))
#query_image=input("Enter the Query Image :  ")
query_image_path=cmd_args.query
query_image=cv2.imread(query_image_path)
features=cd.describeImage(query_image)

sd=Searcher("features_index.csv")
results=sd.search(features,15)
images_set=[]
images_set.append(mpimg.imread(query_image_path))
#show_images(images_set,1)

print("\n\nImages Matched")
for result in results:
    # result_image=cv2.imread("Dataset/oxford5k_images/"+result[1])
    result_image = mpimg.imread("Dataset/oxford5k_images/"+result[1])
    print(result[1])
    images_set.append(result_image)



'''
print(filenames) #or glob or any other way to describe filenames
for i in range(6):
    f=open("Dataset/"+filenames[i])
    image=Image.open(f)
    ax[i%2][i//2].imshow(image)
    f.close()
'''

show_images(images_set,4)
