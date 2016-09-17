## -*- coding: utf-8 -*-
#"""
#Created on Thu Aug  4 10:52:13 2016
#
#@author: rahma
#"""
#
from HOG_functions import sobel, hog, read_data
import os
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import numpy as np
#
##%% Data
os.chdir('/home/spark/Documents/Recherches/Theano/Datasets')
dataset='mnist.pkl.gz'
datasets = read_data(dataset)
# Data
X_df = pd.DataFrame(datasets[0][0])
y = datasets[0][1]

#%% Transformer les données

# Sobel operator for gradient and angle mat
G_mat, A_mat = sobel(X_df.values, n_line = 1500)

#plt.close('all')
##Visualiser l'intensité et l'orientation du gradient
#n_line = 1500 #Exemple pris au hasard pour la visualisation
#visu_sample(X_df.values,G_mat,A_mat,n_line)

#%% HOG Transform

#Train data
bin_nb = 12
hog_mat = []
for n_l in range(np.shape(G_mat)[0]):
    # gradient and angle matrices
    G = np.reshape(G_mat[n_l,:],(28,28))
    A = np.reshape(A_mat[n_l,:],(28,28))
    A = 360*(A + pi)/(2*pi)
    hist_list1 = hog(G,A,4,bin_nb)
    hist_list2 = hog(G,A,7,bin_nb)
    hist_list3 = hog(G,A,14,bin_nb)
    hist_list = np.concatenate([hist_list1,hist_list2,hist_list3],axis=1)
    hog_mat.append(hist_list)
    if(n_l%500==0):
        print n_l

hog_train = np.array(hog_mat)[:,0,:]
# Adding weights: 4 for hist_list1, 2 for hist_list2 and 1 for hist_list3
hog_train = np.concatenate([4.0*hog_train[:,:432],2.0*hog_train[:,432:432+108],hog_train[:,432+108:]],axis=1)



#%%
##########
#Save HOG#
##########
#from tempfile import TemporaryFile
#outfile = TemporaryFile()
#np.save(outfile, hog_train)


#%%
"""
La construction du HOG prends beaucoup de temps. Après sa construction elle est sauvegardé 
pour son utilisation en apprentissage. La matrice HOG représente les featrues.
Dans ce script on teste ces feature avec un SVM avec un noyau intersection pour
voir rapidement son pouvoir discriminatif (plus rapide qu'un réseau de neurone)
"""

import sklearn.metrics as me
from sklearn.svm import SVC, LinearSVC, libsvm
from sklearn.cross_validation import train_test_split

hog_train = np.genfromtxt('/home/spark/Documents/Recherches/Theano/Datasets/HOG_train')


plt.close('all')

def inter_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.sum(np.min(np.array([x,y]),axis=0))
    return gram_matrix

X_tr, X_te, y_tr, y_te = train_test_split(hog_train,y,test_size=0.2)
print('on a le split')
#clf = SVC(C=0.01,kernel='linear')
#clf = LinearSVC(C=0.0001)
clf = SVC(kernel=inter_kernel)

clf.fit(X_tr,y_tr)
print('le model est construit')
pred = clf.predict(X_te)

print "Precision: ", me.accuracy_score(y_te,pred)

y_false = y_te[np.where(y_te!=pred)[0]]
pred_false = pred[np.where(y_te!=pred)[0]]

plt.hist(y_false)



