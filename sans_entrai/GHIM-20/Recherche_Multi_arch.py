from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.models import Model, load_model
import numpy as np
import os
import csv
import keras
import operator
import ntpath
import tensorflow as tf
import pickle
import warnings
from operator import itemgetter
"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)"""
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from shutil import copyfile
import os.path
from os import path
from matplotlib.pyplot import imread
#from keras.backend.tensorflow_backend import set_session
from scipy import spatial
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.xception import Xception, preprocess_input, decode_predictions #299*299
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions #224*224
from keras.applications.inception_v3 import InceptionV3, preprocess_input,decode_predictions# input shape= 299x299
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input,decode_predictions# input shape= 299x299
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions# input shape= 224x224 
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.applications.densenet import DenseNet201, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.applications.nasnet import NASNetMobile, preprocess_input
import time
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD

# For automated test log result directly in the csv
import csv
import sys
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import math
import argparse
import matplotlib
import imghdr
import pickle as pkl
import datetime
import urllib.request
from cycler import cycler
from PIL import Image, ImageEnhance
import zipfile
import shutil


#VAR
current_folder="."

nbr_batch_size=8 
dataset_name='NEW_GHIM' #@param ["NEW_GHIM"]
new_base_dir="NEW_GHIM_VAL_TRAIN"
dataset_path = os.path.join('./', dataset_name)
old_dataset="GHIM-20"

#FUNCTIONS
def get_feature(img_index,lfeatures):
    return lfeatures[img_index - 1]

def euclidianDistance(l1,l2):
    distance = 0
    length = min(len(l1),len(l2))
    for i in range(length):
        distance += pow((l1[i] - l2[i]), 2)
    return math.sqrt(distance)

def bruteForceDistance(l1,l2):
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(l1,l2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    dists=np.array([j.distance for j in matches])
    mean=np.mean(dists)
    moy=np.array([i  for i in dists if i <= mean])
    return np.mean(moy)

def chisquareDistance(l1,l2):
    return cv.compareHist(l1, l2, cv.HISTCMP_CHISQR)

def correlationDistance(l1,l2):
    return -cv.compareHist(l1, l2, cv.HISTCMP_CORREL)

def bhattacharyyaDistance(l1,l2):
    return cv.compareHist(l1, l2, cv.HISTCMP_BHATTACHARYYA)

def flannDistance(l1,l2):
    FLANN_INDEX_KDTREE = flann_index
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = flann_tree)
    search_params = dict(checks=flann_checks)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(l1,l2,k=flann_k)        
    dists=np.array([j.distance for j,l in matches if j.distance < 0.7*l.distance])
    return np.mean(dists)
    
def getkVoisins(img_index, k,dist_index,*args) :
    lfeatures=[]
    path=[]
    cpt=0
    for a in args:
        for i in range(len(a)):
            if cpt==0:
                lfeatures.append(a[i][1])
                path.append(a[i][0])
            else:
                lfeatures[i] = np.concatenate((lfeatures[i], a[i][1]), axis = None)
        cpt+=1
    ldistances = [] 
    start_time = time.time()
    for i in range(len(lfeatures)): 
        if dist_index == 1:
            dist = euclidianDistance(lfeatures[img_index], lfeatures[i]) 
        elif dist_index == 2:
            dist = bhattacharyyaDistance(lfeatures[img_index], lfeatures[i]) 
        elif dist_index == 3:
            dist = correlationDistance(lfeatures[img_index], lfeatures[i]) 
        elif dist_index == 4:
            dist = chisquareDistance(lfeatures[img_index], lfeatures[i]) 
        elif dist_index == 5:
            dist = bruteForceDistance(lfeatures[img_index], lfeatures[i]) 
        elif dist_index == 6:
            dist = flannDistance(lfeatures[img_index], lfeatures[i]) 
        ldistances.append([path[i], lfeatures[i], dist]) 
    ldistances.sort(key=operator.itemgetter(2)) 
    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i])
    end_time = time.time()
    temps = end_time - start_time
    return lvoisins, temps

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


def search (sortie,img_indexes,features1,features2,features3, features4,new_path):
    
    print(new_path)
    result_path=new_path
    #img_class=[[3110,3288,3305],[1970,1801,1775],[9998,9896,9898]]
    for img_index in img_indexes:
        
        ##############################
        ## Récupération des voisins ##
        ##############################
        #img_test = get_feature(img_index, features1 )
        warnings.filterwarnings('ignore')
        voisins, temps = getkVoisins(img_index, sortie, 1, features1, features2, features3, features4)
        
        #print("Temps d execution : %s secondes ---" % (time.time() - start_time))
        nom_image_plus_proches = voisins 
        nom_image_plus_proches_sans = []
        # Sauver les voisins dans un fichier txt
        with open(os.path.join(result_path,"img_"+str(img_index)+"_"+str(sortie)+"_voisins.txt"), 'w') as s: 
            for vsn in voisins:
                s.write((os.path.splitext(os.path.basename(vsn[0].split("_")[1]))[0]) + '\n')
                nom_image_plus_proches_sans.append(os.path.splitext(os.path.basename(vsn[0].split("_")[1]))[0]) 
        
        # Afficher l'image requête
        #plt.figure(figsize=(5, 5)) 
        #plt.imshow(imread(img_test[0]), cmap='gray', interpolation='none') 
        #plt.title("Image " + str(img_index)) 
        #plt.figure(figsize=(25, 25)) 
        #plt.subplots_adjust(hspace=0.2, wspace=0.2) 

        #########################################
        ## Calcul du rappel et de la précision ##
        #########################################
        MaP = np.array([])
        MaR = np.array([])
        rappel_precision = [] 
        rp = [] 
        position1=(int(img_index)-1)//500
        for j in range(sortie): 
            position2=(int(nom_image_plus_proches_sans[j])-1)//500
            if position1==position2: 
                rappel_precision.append("pertinant") 
            else: 
                rappel_precision.append("non pertinant") 

        for i in range(sortie): 
            j=i 
            val=0 
            while j>=0:
                if rappel_precision[j]=="pertinant": 
                    val+=1 
                j-=1 
            #               précision                 rappel
            rp.append(str((val/(i+1))*100)+" "+str((val/sortie)*100))
            MaP = np.append(MaP,(val/(i+1))*100)
            MaR = np.append(MaR,(val/sortie)*100)

            
        MaP = np.mean(MaP)
        MaR = np.mean(MaR)
        print("Classe : {}".format((img_index-1)//500))
        print("Moyenne des précisions - image {} : {}".format(img_index, MaP))
        print("Moyenne des rappels - image {} : {}".format(img_index, MaR))
        print("")
        with open(os.path.join(result_path,"TEMP_MAP"+str(img_index)+"_"+str(sortie)+".txt"), "w") as output:
            output.write("Moyenne des précisions - image {} : {}".format(img_index, MaP))
            output.write("Temps d execution : %s secondes ---" % (temps))
        
        with open(os.path.join(result_path,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt"), 'w') as s: 
            for a in rp: 
                s.write(str(a) + '\n')

        ######################
        ## Tracer la courbe ##
        ######################
        x = [] 
        y = []
        fichier = os.path.join(result_path,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt")
        with open(fichier) as csvfile: 
            plots = csv.reader(csvfile, delimiter=' ') 
            for row in plots: 
                y.append(float(row[0])) # précision
                x.append(float(row[1])) # rappel
        fig = plt.figure() 
        line, =plt.plot(x,y,'C1', label=new_path,marker='o' ) 
        plt.xlabel('Rappel') 
        plt.ylabel('Précison') 
        plt.title("R/P - image "+str(img_index)) 
        plt.legend()
        fig.savefig(os.path.join(result_path,"img_"+str(img_index)+"_"+str(sortie)+".png"))
    
    
    ########################################
    ## Calcul du RP moyen pour une classe ##
    ########################################
    nbr_img = len(img_indexes)
    col = 2
    RP = [[[0 for k in range(col)] for j in range(nbr_img)] for i in range(sortie)]
    i = 0
    for img_index in img_indexes:
        fichier = os.path.join(result_path,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt")
        with open(fichier) as csvfile: 
            temp = csv.reader(csvfile, delimiter=' ')
            j = 0
            for row in temp:
                RP[j][i][0] = row[0]
                RP[j][i][1] = row[1]
                j += 1
            i += 1
    
    x = []
    y = []
    for row in RP:
        y.append((float(row[0][0]) + float(row[1][0]) + float(row[2][0]))/3)
        x.append((float(row[0][1]) + float(row[1][1]) + float(row[2][1]))/3)
    fig = plt.figure() 
    line, =plt.plot(x,y,'C1', label=new_path,marker='o' ) 
    plt.xlabel('Rappel') 
    plt.ylabel('Précison') 
    plt.title("R/P classe " + str((img_index-1)//500)) 
    plt.legend()
    fig.savefig(os.path.join(result_path,"RP_"+str(img_index)+"_"+str(sortie)+"_classe"+str((img_index-1)//500)+".png"))

feature_folder="Features_Train_dense_par4"

classifier1="DenseNet121"
file_name=feature_folder+"/"+classifier1+"/features.txt"
with open(file_name, 'rb') as filehandle:  
    # read the data as binary data stream
    features1 = pickle.load(filehandle)

classifier2="DenseNet169"
file_name=feature_folder+"/"+classifier2+"/features.txt"
with open(file_name, 'rb') as filehandle:  
    # read the data as binary data stream
    features2 = pickle.load(filehandle)

classifier3="DenseNet201"
file_name=feature_folder+"/"+classifier3+"/features.txt"
with open(file_name, 'rb') as filehandle:  
    # read the data as binary data stream
    features3 = pickle.load(filehandle)

classifier4="VGG16"
file_name=feature_folder+"/"+classifier4+"/features.txt"
with open(file_name, 'rb') as filehandle:  
    # read the data as binary data stream
    features4 = pickle.load(filehandle)

#classifier5="VGG19"
#file_name=feature_folder+"/"+classifier5+"/features.txt"
#with open(file_name, 'rb') as filehandle:  
    # read the data as binary data stream
#    features5 = pickle.load(filehandle)



img_class=[[3110,3288,3305],[1970,1801,1775],[9998,9896,9898]]


new_path=feature_folder+"_"+classifier1+"_"+classifier2+"_"+classifier3+"_"+classifier4
if not os.path.exists(new_path): 
    os.makedirs(new_path)

sortie_list =  [20,50,100,500]
for i_list in range(0,len(sortie_list)) :
    sortie=sortie_list[i_list]
    for img_indexes in img_class:
        search(sortie, img_indexes,features1,features2,features3, features4 ,new_path)
