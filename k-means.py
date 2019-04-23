
from  sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np


def centroid_histo(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist,_)=np.histogram(clt.labels_,bins=numLabels)
    hist = hist.astype(np.float32)
    hist=hist/hist.sum()
    return hist


def plot_colors(hist,centroids):
    bar = np.zeros((50, 300, 3), dtype = np.int8)
    startX=0
    for (percent, color) in zip(hist, centroids):
        endX=startX+(percent*300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),color.astype("uint8").tolist(), -1)
        startX = endX
    return bar

    



image=cv2.imread("101402.jpg")
#image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

'''
plt.figure()
plt.axis("off")
plt.imshow(image)
plt.show()
'''


image=image.reshape((image.shape[0]*image.shape[1],3))



clust=KMeans(n_clusters=7)
#print(clust)
clust.fit(image)

hist=centroid_histo(clust)
print(sorted(hist,reverse=False))

bar=plot_colors(hist,clust.cluster_centers_)

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


#hist=centroid_histo(clust)




