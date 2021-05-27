from multiprocessing import Pool, cpu_count
from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions #299*299
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#from keras.applications.efficientNetB7 import efficientNetB7
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


#VAR

#old_dataset="cars_cools"
dataset_name='../new_cars_cools'
dataset_path = os.path.join('./', dataset_name)
new_base_dir="new_cars_cools_val_train"
#classes_path = "./classes.txt"
#csv_path = 'result.csv'





#FUNCTIONS
def get_feature(img_index,lfeatures):
    #print(lfeatures[0][0])
    return os.path.basename(lfeatures[0][img_index][0])

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
        print(len(a[0]))
        for i in range(len(a[0])):
            if cpt==0:
                lfeatures.append(a[0][i][1])
                path.append(a[0][i][0])
            else:
                lfeatures[i] = np.concatenate((lfeatures[i], a[0][i][1]), axis = None)
        cpt+=1
    
    ldistances = []
    print(len(lfeatures))
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
    end_time = time.time()
    search_time= (end_time - start_time)
    ldistances.sort(key=operator.itemgetter(2)) 
    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i])
    return lvoisins , search_time

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


def search (sortie,img_indexes,features,result_path,classifier,datasets):
    
    #result_path = "results/" + choice_fin
    #if not os.path.exists(result_path): 
    #    os.makedirs(result_path)
    
    #print(choice)
    print(result_path)
    
    new_folder=result_path
    if os.path.exists(new_folder) == False:
        os.makedirs(new_folder)
    for img_index in img_indexes:
        
   
        img_test = get_feature(img_index, features)
    
        warnings.filterwarnings('ignore')
        voisins , temps = getkVoisins(img_index, sortie, 1, features)
        
        nom_image_plus_proches = voisins 
        nom_image_plus_proches_sans = []
        # Sauver les voisins dans un fichier txt
        with open(os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+"_voisins.txt"), 'w') as s: 
            for vsn in voisins:
                #print(os.path.basename(vsn[0]))
                s.write(os.path.basename(vsn[0]) + '\n')

                nom_image_plus_proches_sans.append(os.path.basename(vsn[0])) 
        
         #Afficher l'image requête
        sous_dossier=img_test.split("_")[0]+"_"+img_test.split("_")[1]
        if(sortie <= 20):
            plt.figure(figsize=(5, 5))
            plt.imshow(imread(datasets+"/"+sous_dossier+"/"+img_test), cmap='gray', interpolation='none')
            plt.title("Image requête")
        if(sortie <= 20):
            plt.figure(figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        
        #########################################
        ## Calcul du rappel et de la précision ##
        #########################################
        MaP = np.array([])
        MaR = np.array([])

        rappel_precision = [] 
        rp = [] 
        print(img_test)
        position1=img_test.split('_')[0]
        position3=img_test.split('_')[1]
        for j in range(sortie): 
            if(sortie == 5):
                plt.subplot(2,4,j+1)
                print(nom_image_plus_proches[j])
                plt.imshow(imread(nom_image_plus_proches[j][0]), cmap='gray', interpolation='none')
            if(sortie == 10):
                plt.subplot(3,4,j+1)
                print(nom_image_plus_proches[j])
                plt.imshow(imread(nom_image_plus_proches[j][0]), cmap='gray', interpolation='none')
            if(sortie == 15):
                plt.subplot(4,4,j+1)
                print(nom_image_plus_proches[j])
                plt.imshow(imread(nom_image_plus_proches[j][0]), cmap='gray', interpolation='none')
            if(sortie == 20):
                plt.subplot(5,4,j+1)
                print(nom_image_plus_proches[j])
                plt.imshow(imread(nom_image_plus_proches[j][0]), cmap='gray', interpolation='none')
            
            cools=os.path.basename(nom_image_plus_proches_sans[j])
            n_marque, n_modele, marque, modele, num = cools.split('_')
            if(sortie <= 20):
                title = "Image proche n° "+str(j)+" (n° "+ num +")"
                plt.title(title)
            if position1==n_marque and position3==n_modele: 
                rappel_precision.append("pertinant") 
            else: 
                rappel_precision.append("non pertinant") 
        if(sortie <= 20):
            plt.savefig(new_folder+"/Output_image_"+str(img_index)+"_top"+str(sortie)+"_"+classifier+".pdf")
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
        print("Classe : {}".format(position1))
        print("Moyenne des précisions - image {} : {}".format(img_test, MaP))
        print("Moyenne des rappels - image {} : {}".format(img_test, MaR))
        print("")
        with open(os.path.join(new_folder,"TEMP_MAP"+str(img_index)+"_"+str(sortie)+".txt"), "w") as output:
            output.write("MAP - image {} : {}".format(img_test, MaP))
            output.write("Temps de recherche : %s secondes ---" % (temps))
        
        with open(os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt"), 'w') as s: 
            for a in rp: 
                s.write(str(a) + '\n')

        ######################
        ## Tracer la courbe ##
        ######################
        x = [] 
        y = []
        fichier = os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt")
        with open(fichier) as csvfile: 
            plots = csv.reader(csvfile, delimiter=' ') 
            for row in plots: 
                y.append(float(row[0])) # précision
                x.append(float(row[1])) # rappel
        fig = plt.figure() 
        line, =plt.plot(x,y,'C1', label=classifier,marker='o' ) 
        plt.xlabel('Rappel') 
        plt.ylabel('Précison') 
        plt.title("R/P - image "+str(img_test)) 
        plt.legend()
        fig.savefig(os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+".png"))
    
    
    ########################################
    ## Calcul du RP moyen pour une classe ##
    ########################################
    nbr_img = len(img_indexes)
    col = 2
    RP = [[[0 for k in range(col)] for j in range(nbr_img)] for i in range(sortie)]
    i = 0
    for img_index in img_indexes:
        fichier = os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt")
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
    plt.title("R/P classe " + str(position1)) 
    plt.legend()
    fig.savefig(os.path.join(new_folder,"RP_"+str(position1)+"_"+str(sortie)+"_classe"+str(position1)+".png"))


classifier_list1 = ["VGG16","MobileNet","Xception","ResNet50","DenseNet169"]
layers= ["block5_conv3","conv_pw_13","conv2d_3","conv5_block3_3_conv","conv5_block32_2_conv"]
classifier_list2 = ["0","1","2","3","4"]
dossier = None
old_dataset = None
def recherche_image(classifier_list2):
    for id_classifier in range(0,len(classifier_list2)):
        classifier=classifier_list2[id_classifier]
        classifier = classifier_list1[int(classifier)]
    

        features = [] #Stocker les caractérstiques
        folder_features="Features_"+dossier
        classifier_path=folder_features+"/"+classifier
        result_path = classifier_path +"/outputs"
        if os.path.exists(result_path) == False:
            os.makedirs(result_path)

        features = []
        features_path = classifier_path +"/features"
        #print(features_path)
        with open(classifier_path+"/features.txt", 'rb') as filehandle:  
            # read the data as binary data stream    
            features.append(pickle.load(filehandle))
            
        features=sorted(features,key=itemgetter(0))
        
        img_class=[[9283,9285, 9299],[12382, 12451, 12443],[1801, 1919, 1841]]
        

        #recherche 

        sortie_list =  [5, 10, 20,50,100]#@param {type:"number"}
        for i_list in range(0,len(sortie_list)) :
            sortie=sortie_list[i_list]
            for img_indexes in img_class:
                search(sortie, img_indexes,features,result_path,classifier,old_dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--dossier', type=str,default='./Results',
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

    #recherche 
    p = Pool(cpu_count())
    print("====================Train set====================")
    _ = p.map(recherche_image,classifier_list2)



        
    
