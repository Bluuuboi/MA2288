import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

picture = plt.imread('pic.jpg')
print('Type of the image : ' , type(picture))
print(f'Shape of the image : {picture.shape}')
print(f'Image Height {picture.shape[0]}')
print(f'Image Width {picture.shape[1]}')
print(f'Dimension of Image {picture.ndim}')
print(f'Size of Image {picture.size}')
# plt.imshow(picture)
# plt.show()

# Let's save each channel in its own two dimensional matrix
R=picture[:,:,0] # red channel
G=picture[:,:,1] # green channel
B=picture[:,:,2] # blue channel

# Let's display these three channels. I will plot first two on grey scale.
fig,subs = plt.subplots(nrows = 1, ncols=3, figsize=(15,5))
subs[0].imshow(R,cmap=cm.Greys_r)
subs[0].set_title('R',fontsize=30)
subs[1].imshow(G,cmap=cm.Greys_r)
subs[1].set_title('G',fontsize=30)
subs[2].imshow(B,cmap=cm.Greys_r)
subs[2].set_title('B',fontsize=30)
# plt.show()

# If we want to see Red/Green/Blue tints for each,then we have to layer each of them with two zero layers, since a computer color image is made up of three layers.
# Define new zero channel
z=np.zeros(np.shape(R))
# scale all the pixels to numbers between 0 and 1 (instead of between 0 and 255, unless you are sure those were all integers)
R1=R/255
G1=G/255
B1=B/255
# red channel stays, others zero
red_tint=np.dstack((R1,z,z))
# green channel stays, others zero
green_tint=np.dstack((z,G1,z))
# blue channel stays, others zero
blue_tint=np.dstack((z,z,B1))
# split the figure into 3 subplots
fig, subs=plt.subplots(nrows = 1, ncols=3, figsize=(15,5))
#plt.setp(subs, xticks=[], xticklabels=[],yticks=[],yticklabels=[])
subs[0].imshow(red_tint)
subs[0].set_title('R',fontsize=30)
subs[1].imshow(green_tint)
subs[1].set_title('G',fontsize=30)
subs[2].imshow(blue_tint)
subs[2].set_title('B',fontsize=30)
# plt.show()

#Now, we perform SVD for each channel.
fig,subs = plt.subplots(nrows = 1, ncols=3, figsize=(15,5))

# Let's keep the first 25 singular values
r=25
# red channel
U,sigma,Vt=np.linalg.svd(R)
print(f"The original number of nonzero singular values is {len(sigma)}")
new_sigma=sigma[0:r]
new_Sigma=np.diag(new_sigma)
new_U=U[:,0:r]
new_Vt=Vt[0:r,:]
product_r=new_U.dot(new_Sigma.dot(new_Vt))
subs[0].imshow(product_r,cmap=cm.Greys_r)
subs[0].set_title('Reduced R',fontsize=25)

# green channel
U,sigma,Vt=np.linalg.svd(G)
new_sigma=sigma[0:r]
new_Sigma=np.diag(new_sigma)
new_U=U[:,0:r]
new_Vt=Vt[0:r,:]
product_g=new_U.dot(new_Sigma.dot(new_Vt))
subs[1].imshow(product_g,cmap=cm.Greys_r)
subs[1].set_title('Reduced G',fontsize=25)

# blue channel
U,sigma,Vt=np.linalg.svd(B)
new_sigma=sigma[0:r]
new_Sigma=np.diag(new_sigma)
new_U=U[:,0:r]
new_Vt=Vt[0:r,:]
product_b=new_U.dot(new_Sigma.dot(new_Vt))
subs[2].imshow(product_b,cmap=cm.Greys_r)
subs[2].set_title('Reduced B',fontsize=25)

# plt.show()

#Now we layer all three channels to see the compressed color image.
product_r_1=product_r.astype(int)
product_g_1=product_g.astype(int)
product_b_1=product_b.astype(int)
reduced_rank_image=np.dstack((product_r_1,product_g_1,product_b_1))

fig,subs=plt.subplots(nrows=1, ncols=2, figsize=(10,5))
subs[0].imshow(picture)
subs[0].set_title('Original', fontsize=30)
subs[1].imshow(reduced_rank_image)
subs[1].set_title(f'Reduced Rank', fontsize=30)

print(f'The original image had {picture.size} pixels')
print(f'The reduced rank image has {reduced_rank_image.size} pixels but it requires much less storage.')

plt.show()