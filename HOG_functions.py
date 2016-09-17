# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:53:08 2016

@author: rahma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg


def read_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    import cPickle, gzip

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    print('... loading data')
    
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval 
    
    
def atan2(x,y):
    angl = 2*np.arctan(y/(np.sqrt(x**2+y**2)+x))
    angl[angl!=angl]=0
    return angl
    
def sobel(X, n_line = None):
    ''' Construction du filtre de Sobel
    '''    
    n_samples, n_features = X.shape
    G = []
    Angl = []
    op1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    op2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])    
    for i in range(n_samples):
        A = np.reshape(X[i,:],(28,28))
        Gx = sg.convolve(op1,A)[:-2,:-2]
        Gy = sg.convolve(op2,A)[:-2,:-2]
#        if (n_line != None) and (i ==n_line):
#            plt.figure()
#            plt.subplot(121)
#            plt.imshow(Gx, cmap=plt.cm.gray, interpolation='none')
#            plt.subplot(122)
#            plt.imshow(Gy, cmap=plt.cm.gray, interpolation='none')
#            
#            
        g = np.sqrt(Gx**2+Gy**2)
       
        angl = atan2(Gx,Gy) 
        G.append(list(np.reshape(g,(np.shape(g)[0]**2,1))))
        Angl.append(list(np.reshape(angl,(np.shape(angl)[0]**2,1))))
    return np.array(G)[:,:,0], np.array(Angl)[:,:,0]

def visu_sample(X,G,A, n_line):
    plt.figure()
    
    plt.subplot(131)
    plt.imshow(X[n_line,:].reshape(28,28), cmap=plt.cm.gray, interpolation='none')
    plt.subplot(132)
    plt.imshow(G[n_line,:].reshape(28,28), cmap=plt.cm.gray, interpolation='none')
    plt.subplot(133)
    plt.imshow(A[n_line,:].reshape(28,28), cmap=plt.cm.gray, interpolation='none')
  

def block_per_line(cell_size):
    count = 1
    temp = 2*cell_size
    while(temp<28):
        count += 1
        temp += cell_size       
    return count


def hog(G,A,cell_size,bin_nb):
    ''' Construction du histogramme
    '''     
    # Divide image into cells of cell_size**2 size
    hist_list = []
    hist_val = np.linspace(0,360,bin_nb)
    # Count the number of overlapping block per line/column
    blk_line = block_per_line(cell_size)
    # Compute hist for each block
    for k in range(blk_line):
        for l in range(blk_line):
            A_blk = A[k*cell_size:(k+2)*cell_size,l*cell_size:(l+2)*cell_size]
            G_blk = G[k*cell_size:(k+2)*cell_size,l*cell_size:(l+2)*cell_size]
            hist = np.zeros(bin_nb) 
            for line in range(np.shape(A_blk)[0]):
                for col in range(np.shape(A_blk)[1]):          
                    angl = A_blk[line,col]
                    grad = G_blk[line,col]
                    indices_two_closest = np.argsort((hist_val - angl)**2)[:2]
                    two_closest_val = hist_val[indices_two_closest]
                    angl_two_closest_diff = np.abs(two_closest_val - angl)
                    ratio_two_closest_val = 1 -  angl_two_closest_diff/(np.sum(angl_two_closest_diff))
                    hist[indices_two_closest] += grad*ratio_two_closest_val*two_closest_val
            hist_list.append(hist)
    hist_list = np.array(hist_list)
    hist_list = np.reshape(hist_list,(1,np.size(hist_list)))
    return hist_list
    

def pre_pro(X_val,buff=30):
    X = X_val.copy()
    col_nb = np.shape(X)[1]
    X[X<0.0] = 0.0
    begin_points = np.argmin(X<0.85,axis=1) - buff
    begin_points[begin_points<=0.0] = 0
    end_points = col_nb - np.argmin(X[:,::-1]<0.85,axis=1) + buff
    end_points[end_points>=col_nb-1] = col_nb -1
    for i in range(np.shape(X)[0]):
        X[i,:begin_points[i]] = 0.0
        X[i,end_points[i]:] = 0.0
    X[X<0.60] = 0.0
    return X
    
def plot_digit(X_df, y, value, fig_nb, n_rows=4, n_cols=5, max_iter=200):
    temp=0
    print value
    for k in range(max_iter):
        if(temp>=fig_nb):
            break
        if(y[k] == value):
            plt.subplot(n_rows, n_cols, temp+1)
            plt.imshow(np.reshape(X_df[k,:],(28,28)), cmap=plt.cm.gray, interpolation='none')
            plt.xticks(())
            plt.yticks(())
            temp += 1