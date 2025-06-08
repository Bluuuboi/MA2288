import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#NARROW RECTANGLE MATRIX C
# Input the matrix and compute its singular value decomposition
C=np.array([[9,2],[24,8],[39,14]])
# U,sigma,Vt=np.linalg.svd(C)
# print("U=\n",U)
# print("sigma=\n",sigma)
# # store sigma in a diagonal matrix that has the same shape as C
# Sigma=np.zeros(C.shape) # Sigma has the same shape as A
# m=np.amin(C.shape) # pick the smaller number between the number of rows and columns
# Sigma[0:m,0:m]=np.diag(sigma) # place the singular values on the diagonal of Sigma
# print("Sigma=\n",Sigma)
# print("Vt=\n",Vt)
#
# # Let's visualize the above product
# # split the figure into 4 subplots
# fig,subs=plt.subplots(nrows = 1, ncols=4, figsize=(10,5))
#
# # remove the x and y ticks and labels from all the subplots
# plt.setp(subs, xticks=[], xticklabels=[],yticks=[],yticklabels=[])
#
# # plot the matrices
# subs[0].imshow(C,cmap=cm.Greys_r)
# subs[0].set_title('C',fontsize=30)
# subs[1].imshow(U,cmap=cm.Greys_r)
# subs[1].set_title('U',fontsize=30)
# subs[2].imshow(Sigma,cmap=cm.Greys_r)
# subs[2].set_title('$\Sigma$',fontsize=30)
# subs[3].imshow(Vt,cmap=cm.Greys_r)
# subs[3].set_title('$V^t$',fontsize=30)
#
# fig.text(0.31,0.5,'=',horizontalalignment='center',
#          verticalalignment='center', fontsize=30, color='k')
# plt.show()

#WIDE RECTANGLE MATRIX A
A=np.array([[-1,3,-5,4,18],[1,-2,4,0,-7],[2,0,4,-3,-8]])
# U,sigma,Vt=np.linalg.svd(A)
# print("U=\n",U)
# print("sigma=\n",sigma)
# Sigma=np.zeros(A.shape)
# m=np.amin(A.shape)
# Sigma[0:m,0:m]=np.diag(sigma)
# print("Sigma=\n",Sigma)
# print("Vt=\n",Vt)
#
# #Check if we get back A again
# product=U.dot(Sigma.dot(Vt))
# print("product=\n",product)
#
# # Let's visualize the above product
# # split the figure into 4 subplots
# fig,subs=plt.subplots(nrows = 1, ncols=4, figsize=(10,5))
# # remove the x and y ticks and labels from all the subplots
# plt.setp(subs, xticks=[], xticklabels=[],yticks=[],yticklabels=[])
# subs[0].imshow(A,cmap=cm.Greys_r)
# subs[0].set_title('A', fontsize=30)
# subs[1].imshow(U,cmap=cm.Greys_r)
# subs[1].set_title('U', fontsize=30)
# subs[2].imshow(Sigma,cmap=cm.Greys_r)
# subs[2].set_title('$\Sigma$', fontsize=30)
# subs[3].imshow(Vt,cmap=cm.Greys_r)
# subs[3].set_title('$V^t$', fontsize=30)
#
# fig.text(0.31,0.5,'=',horizontalalignment='center',
#          verticalalignment='center', fontsize=30, color='k')
#
# plt.show()

