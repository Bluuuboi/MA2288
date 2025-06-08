import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage as ndi #Multi-dim image processing, we will use convolution from here
from skimage.filters import gabor_kernel

picture = plt.imread('Fig_Sary.jpg')
# print(picture)
# print(picture.shape)
plt.imshow(picture)
plt.show()

#Save each channel in its own 2D matrix
#Recall: Each channel = R/G/B for colours
R = picture[:,:,0] #Red channel
G = picture[:,:,1] #Green channel
B = picture[:,:,2] #Blue channel
fig,subs = plt.subplots(1,3)
subs[0].imshow(R, cmap=cm.Greys_r)
subs[1].imshow(G, cmap=cm.Greys_r)
subs[2].imshow(B, cmap=cm.Greys_r)
plt.show()

#convert initial image to grayscale
gray=np.around(0.2989*R+0.5870*G+0.1140*B).astype(np.uint8)
gray_image=np.dstack((gray,gray,gray))
plt.imshow(gray_image)
plt.show()

# choose the frequency desired in the texture
frequency=0.025
# f=0.05 captured nice texture for grayscale image
# f=0.1 for green channel

#Gabor filter has a real and imaginary component
# create the filter with the chosen frequency and different orientations
filter_matrix_0=gabor_kernel(frequency, theta=0, bandwidth=1, dtype=np.complex128)
filter_matrix_45=gabor_kernel(frequency, theta=np.pi/4, bandwidth=1, dtype=np.complex128)
filter_matrix_90=gabor_kernel(frequency, theta=np.pi/2, bandwidth=1, dtype=np.complex128)
filter_matrix_135=gabor_kernel(frequency, theta=3*np.pi/4, bandwidth=1, dtype=np.complex128)

# get the real part of the filter
filter_matrix_0=np.real(filter_matrix_0)
filter_matrix_45=np.real(filter_matrix_45)
filter_matrix_90=np.real(filter_matrix_90)
filter_matrix_135=np.real(filter_matrix_135)

# choose the image to filter
image=G # the green channel
# image=gray # one channel of the grayscale image (all channels are equal anyway)

# filter the image by convolution with each filter
filtered_image_0= ndi.convolve(image, filter_matrix_0, mode='wrap')
filtered_image_45= ndi.convolve(image, filter_matrix_45, mode='wrap')
filtered_image_90= ndi.convolve(image, filter_matrix_90, mode='wrap')
filtered_image_135= ndi.convolve(image, filter_matrix_135, mode='wrap')

fig,subs = plt.subplots(nrows=1, ncols=3, figsize=(25,8))
subs[0].imshow(G,cmap=cm.Greys_r)
subs[1].imshow(filter_matrix_0,cmap=cm.Greys_r)
subs[2].imshow(filtered_image_0,cmap=cm.Greys_r)
plt.show()

fig,subs = plt.subplots(nrows=1, ncols=3, figsize=(25,8))
subs[0].imshow(G,cmap=cm.Greys_r)
subs[1].imshow(filter_matrix_45,cmap=cm.Greys_r)
subs[2].imshow(filtered_image_45,cmap=cm.Greys_r)
plt.show()

fig,subs = plt.subplots(nrows=1, ncols=3, figsize=(25,8))
subs[0].imshow(G,cmap=cm.Greys_r)
subs[1].imshow(filter_matrix_90,cmap=cm.Greys_r)
subs[2].imshow(filtered_image_90,cmap=cm.Greys_r)
plt.show()

fig,subs = plt.subplots(nrows=1, ncols=3, figsize=(25,8))
subs[0].imshow(G,cmap=cm.Greys_r)
subs[1].imshow(filter_matrix_135,cmap=cm.Greys_r)
subs[2].imshow(filtered_image_135,cmap=cm.Greys_r)
plt.show()



