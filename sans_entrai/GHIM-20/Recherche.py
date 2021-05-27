from multiprocessing import Pool, cpu_count
from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential,Model, load_model
import numpy as np
import os
import csv
#import keras
import operator
import ntpath
import tensorflow as tf
import pickle
import warnings
from operator import itemgetter
from tensorflow.keras.optimizers import Adam
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
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten,Lambda
#from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions #299*299
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#from tensorflow.keras.applications.efficientNetB7 import efficientNetB7
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions #224*224
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input,decode_predictions# input shape= 299x299
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input,decode_predictions# input shape= 299x299
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions# input shape= 224x224 
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
import argparse
# For automated test log result directly in the csv
import csv
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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


#FUNCTIONS
def get_feature(img_index,lfeatures):
    return lfeatures[img_index - 1]

def euclidianDistance(l1,l2):
    distance = 0
    length = min(len(l1),len(l2))
    for i in range(length):
        distance += pow((l1[i] - l2[i]), 2)
    return math.sqrt(distance)
    
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
        ldistances.append([path[i], lfeatures[i], dist]) 
    end_time = time.time()
    search_time= (end_time - start_time)
    ldistances.sort(key=operator.itemgetter(2)) 
    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i])
    return lvoisins , search_time

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def search (sortie,img_indexes,features,result_path,classifier,old_dataset):
    
    for img_index in img_indexes:
        
        ##############################
        ## Récupération des voisins ##
        ##############################
        #(nom_image,sdgsfgdfgdfh,3110)
        img_test = get_feature(img_index, features)
        warnings.filterwarnings('ignore')
        voisins , temps = getkVoisins(img_index, sortie, 1, features)
        
        print("Temps d execution : %s secondes ---" % (temps))

        nom_image_plus_proches = voisins 
        nom_image_plus_proches_sans = []
        # Sauver les voisins dans un fichier txt
        with open(os.path.join(result_path,"img_"+str(img_index)+"_"+str(sortie)+"_voisins.txt"), 'w') as s: 
            for vsn in voisins:
                s.write((os.path.splitext(os.path.basename(vsn[0].split("_")[1]))[0]) + '\n')
                nom_image_plus_proches_sans.append(os.path.splitext(os.path.basename(vsn[0].split("_")[1]))[0]) 
        #########################################
        ## Calcul du rappel et de la précision ##
        #########################################
        if(sortie == 20):
            plt.figure(figsize=(5, 5))
            #print(img_test[0]) --> OK dans GHIM-20
            plt.imshow(imread(img_test[0]), cmap='gray', interpolation='none')
            title1 = "Image req"
            plt.title(title1)
        if(sortie == 20):
            plt.figure(figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2, wspace=0.2)
        MaP = np.array([])
        MaR = np.array([])
        rappel_precision = [] 
        rp = [] 
        position1=(int(img_index)-1)//500
        for j in range(sortie): 
            position2=(int(nom_image_plus_proches_sans[j])-1)//500
            if(sortie == 20):
                plt.subplot(sortie/4,sortie/5,j+1)
                plt.imshow(imread(old_dataset+"/"+str(position2)+"_"+str(nom_image_plus_proches_sans[j])+".jpg"), cmap='gray', interpolation='none')
            if position1==position2: 
                rappel_precision.append("pertinant") 
            else: 
                rappel_precision.append("non pertinant") 
        if(sortie == 20):
            plt.savefig(result_path+"/Output_image_"+str(img_index)+"_top"+str(sortie)+"_"+classifier+".pdf")
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
        line, =plt.plot(x,y,'C1', label=classifier,marker='o' ) 
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
    line, =plt.plot(x,y,'C1', label=classifier,marker='o' ) 
    plt.xlabel('Rappel') 
    plt.ylabel('Précison') 
    plt.title("R/P classe " + str((img_index-1)//500)) 
    plt.legend()
    fig.savefig(os.path.join(result_path,"RP_"+str(img_index)+"_"+str(sortie)+"_classe"+str((img_index-1)//500)+".png"))


classifier_list1 = ["Xception","VGG16","VGG19","ResNet50","InceptionV3","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201"]
classifier_list2 = ["0","1","2","3","4","5","6","7","8","9"]
dossier = None
old_dataset = None

def recherche_image(classifier_list2):
    for id_classifier in range(0,len(classifier_list2)):
        classifier=classifier_list2[id_classifier]
        classifier = classifier_list1[int(classifier)]
        model_filename = 'model-'+classifier+'.h5'

        if classifier=="InceptionResNetV2" or classifier=="SqueezeNet" or classifier=="Xception" or classifier=="InceptionV3" or classifier=="InceptionResNetV2"  :
            img_height = 299
            img_width = 299
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)
        elif classifier=="VGG16" or classifier=="VGG19" or classifier=="ResNet50" or classifier=="MobileNet" or classifier=="NASNetMobile" or classifier =="DenseNet121" or classifier =="DenseNet169" or classifier=="DenseNet201":
            img_height = 224
            img_width = 224
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)
        elif classifier=="NASNetLarge" :
            img_height = 331
            img_width = 331
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)  
        
        features = []
        folder_features="Features_Sans"
        classifier_path=folder_features+"/"+classifier
        features_path = classifier_path +"/output"
        if os.path.exists(features_path) == False:
            os.makedirs(features_path)
        features_file= classifier_path +"/features.txt"
        with open(features_file, 'rb') as filehandle:
            features=pickle.load(filehandle)
        
        img_class=[[3110,3288,3305],[1970,1801,1775],[9998,9896,9898]]
        sortie_list =  [20,50,100,500]
        for i_list in range(0,len(sortie_list)) :
            sortie=sortie_list[i_list]
            for img_indexes in img_class:
                search(sortie, img_indexes,features,features_path,classifier,old_dataset)

#Recherche
if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--dossier', type=str,default='Results',
        help='path to image requete '
    )
    parser.add_argument(
        '--base', type=str,default='GHIM-20',
        help='modele  '
    )

    FLAGS = parser.parse_args()
    datasets = FLAGS.base
    dossier = FLAGS.dossier
    old_dataset="../../DataSets/"+datasets

    p = Pool(cpu_count())
    print("====================Train set====================")
    _ = p.map(recherche_image,classifier_list2)
        