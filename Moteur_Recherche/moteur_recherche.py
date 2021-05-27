
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
import sys
import argparse
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
    
def getkVoisins(img_test_feature, k,dist_index,*args) :
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
    for i in range(len(lfeatures)): 
        if dist_index == 1:
            dist = euclidianDistance(img_test_feature, lfeatures[i]) 
        elif dist_index == 2:
            dist = bhattacharyyaDistance(img_test_feature, lfeatures[i]) 
        elif dist_index == 3:
            dist = correlationDistance(img_test_feature, lfeatures[i]) 
        elif dist_index == 4:
            dist = chisquareDistance(img_test_feature, lfeatures[i]) 
        elif dist_index == 5:
            dist = bruteForceDistance(img_test_feature, lfeatures[i]) 
        elif dist_index == 6:
            dist = flannDistance(img_test_feature, lfeatures[i]) 
        ldistances.append([path[i], lfeatures[i], dist]) 
    ldistances.sort(key=operator.itemgetter(2)) 
    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i])
    return lvoisins

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


def search (sortie,features1,new_path,position1,img_test,img_test_feature):
    
    print(new_path)
    result_path=new_path
    start_time = time.time()
    ##############################
    ## Récupération des voisins ##
    ##############################
    #img_test = get_feature(img_index, features1)
    warnings.filterwarnings('ignore')
    voisins = getkVoisins(img_test_feature, sortie, 1, features1)
    end_time = time.time()
    #print("Temps d execution : %s secondes ---" % (time.time() - start_time))
    nom_image_plus_proches = voisins 
    nom_image_plus_proches_sans = []
    # Sauver les voisins dans un fichier txt
    with open(os.path.join(result_path,"img_requete_"+str(sortie)+"_voisins.txt"), 'w') as s: 
        for vsn in voisins:
            s.write((os.path.splitext(os.path.basename(vsn[0].split("_")[1]))[0]) + '\n')
            nom_image_plus_proches_sans.append(os.path.splitext(os.path.basename(vsn[0].split("_")[1]))[0]) 
    
    # Afficher l'image requête
    plt.figure(figsize=(5, 5)) 
    plt.imshow(imread(img_test[0]), cmap='gray', interpolation='none') 
    plt.title("Image " + str(img_index)) 
    plt.figure(figsize=(25, 25)) 
    plt.subplots_adjust(hspace=0.2, wspace=0.2) 

    #########################################
    ## Calcul du rappel et de la précision ##
    #########################################
    MaP = np.array([])
    MaR = np.array([])
    rappel_precision = [] 
    rp = [] 
    #position1=(int(img_index)-1)//500
    for j in range(sortie): 
        position2=(int(nom_image_plus_proches_sans[j])-1)//500
        print("position 1 : "+str(position1) )
        print("position 2 : "+str(position2) )
        if int(position1)==int(position2): 
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
        MaR = np.append(MaR,(val/500)*100)

        
    MaP = np.mean(MaP)
    MaR = np.mean(MaR)
    print("Classe : {}".format(position1))
    print("Moyenne des précisions - image requete : {}".format( MaP))
    print("Moyenne des rappels - image requete : {}".format( MaR))
    print("")
    with open(os.path.join(result_path,"TEMP_MAP_requete_"+str(sortie)+".txt"), "wb") as output:
        pickle.dump("Moyenne des précisions - image requete : {}".format( MaP), output)
        pickle.dump("Temps d execution : %s secondes ---" % (end_time - start_time), output)
    
    with open(os.path.join(result_path,"img_requete_"+str(sortie)+"_RP.txt"), 'w') as s: 
        for a in rp: 
            s.write(str(a) + '\n')

    ######################
    ## Tracer la courbe ##
    ######################
    x = [] 
    y = []
    fichier = os.path.join(result_path,"img_requete_"+str(sortie)+"_RP.txt")
    with open(fichier) as csvfile: 
        plots = csv.reader(csvfile, delimiter=' ') 
        for row in plots: 
            y.append(float(row[0])) # précision
            x.append(float(row[1])) # rappel
    fig = plt.figure() 
    line, =plt.plot(x,y,'C1', label=new_path,marker='o' ) 
    plt.xlabel('Rappel') 
    plt.ylabel('Précison') 
    plt.title("R/P - img_requete ") 
    plt.legend()
    fig.savefig(os.path.join(result_path,"img_requete_"+str(sortie)+".png"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--image', type=str,default='./image_requete/I30_Aurelie.jpeg',
        help='path to image requete '
    )
    parser.add_argument(
        '--model', type=str,default='VGG16',
        help='modele  '
    )

    FLAGS = parser.parse_args()
    classifier = FLAGS.model
    image_requete =FLAGS.image
    print(image_requete)
    classifier="DenseNet121"
    file_name="Features_Train/"+classifier+"/features.txt"
    with open(file_name, 'rb') as filehandle:  
        # read the data as binary data stream
        features1 = pickle.load(filehandle)

    model_final_of_classification ="Results/"+classifier+"/"+classifier+"_final.h5"
    model=load_model(model_final_of_classification)
    model2 = Model(inputs=model.input, outputs=model.layers[-2].output)


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

    
    classes = []
    with open("classes.txt", 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))
    image = load_img(image_requete, target_size=input_size)
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda x: x[1]) # Ajouter par Aurélie signature OUI
    (classe_img, prob) = result[0]

    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes

    # Model 2
    feature = model2.predict(image)
    feature = np.array(feature[0]) 
    #feature=np.vectorize(feature) 
    image_requete_clean=os.path.basename(image_requete).split(".")[0]
    new_path="Results/Notre_moteur/"+image_requete_clean
    if os.path.exists(new_path) == False:
        os.makedirs(new_path)
    sortie_list =  [20,50,100,500]#@param {type:"number"}
    for i_list in range(0,len(sortie_list)) :
        sortie=sortie_list[i_list]
        search(sortie,features1,new_path,classe_img,image_requete,feature)
