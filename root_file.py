from PIL import Image
import numpy
import compute_pca
from numpy import *
import os
from sklearn.model_selection import train_test_split

# this function reads the image from the dataset
def read_image():
    Exclude_list=[".info"]
    basewidth=50
    hsize=50
    dict_list={}
    folder = './CroppedYale' # mention the folder name of the dataset here
    for filename in os.listdir(folder):
        temp=filename.endswith('.')
        if temp  in Exclude_list:
            break
        else:
            dict_list[filename]=[]
            inner_folder=folder+"/"+str(filename)
            for i in os.listdir(inner_folder):
                img = Image.open(os.path.join(inner_folder, i)).convert('L') # convert the image to grey scale
                img = img.resize((basewidth, hsize), Image.ANTIALIAS)
                arr=array(img)
                arr=arr.ravel()
                arr=numpy.matrix(arr)
                dict_list[filename].append(arr)
    return dict_list


# lets divide the dataset into training and test data
def divide_list(dict_list):
    d_train = {}
    d_test = {}
    for i in dict_list:
        d_train[i] = []
        d_test[i]=[]
        d_train[i],d_test[i] = train_test_split(dict_list[i],test_size=0.2,train_size=0.8)
    dmean = {}
    mean = {}
    for i in d_train:
        v= numpy.matrix(array(d_train[i])).T
        mean[i]=numpy.matrix.mean(v,axis=1)
        dmean[i]=v-numpy.matrix.mean(v,axis=1)
    return dmean,d_test,mean

# this function calls the PCA function for calculating the PCA for k times
def calculate_principal_component(dmean,k):
    d_final={}
    for j in range(1,k+1):
        for i in dmean:
            v=dmean[i]
            Q =compute_pca.calculate_pca(v, 10)# compute the L1-PCA for each training set
            if(i not in d_final):
                lst = []
                Q=numpy.matrix(Q).T
                lst.append(Q)
                d_final[i]=lst
            else:
                lst=d_final[i]
                Q=numpy.matrix(Q).T
                lst.append(Q)
                d_final[i]=lst
        # removing the contribution of computed PCA with the Data matrix and updating the matrix
        for i in dmean:
            v = dmean[i]
            lst=d_final[i]
            Q=lst[len(lst)-1]
            v=numpy.subtract(v, numpy.matmul(Q, numpy.matmul(Q.T,v)))
            dmean[i]=matrix(v)
    return d_final
