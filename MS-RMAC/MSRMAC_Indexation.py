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
import argparse
from shutil import copyfile
import os.path
from os import path
from matplotlib.pyplot import imread
from rmac import RMAC
#from keras.backend.tensorflow_backend import set_session
from scipy import spatial
import numpy as np
from rmac import RMAC
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

#FUNCTIONS
def get_feature(img_index,lfeatures):
    return lfeatures[img_index - 1]



def get_position (base_model,classifier):
    couches = []

    for layer in base_model.layers:
        couches.append(layer.name)
    
    position = []
    number = 0
    i=1
    if classifier == "VGG16":
        for x in reversed(couches):
            if x == "MaxPooling2D" and number < 5:
                position.append(i)
                number = number +1
            i=i+1
    elif classifier == "ResNet50":
        position.append(5)
        position.append(15)
        position.append(26)
        position.append(37)
        position.append(47)
       
    elif classifier == "MobileNet":
        position.append(6)
        position.append(12)
        position.append(19)
        position.append(25)
        position.append(31)
       
    elif classifier == "DenseNet169":
        position.append(6)
        position.append(13)
        position.append(20)
        position.append(27)
        position.append(34)
    elif classifier == "Xception":
        position.append(6)
        position.append(9)
    return position

def MSRMAC(base_model,image,position):
    
    features = []
    taille_tot=0 
    for i in position :
        if classifier == "VGG16":
            base_out = base_model.get_layer(base_model.layers[-i-1].name).output
        else :
            base_out = base_model.get_layer(base_model.layers[-i].name).output
        rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)
        if classifier == "VGG16":
            rmac_layer = Lambda(rmac.rmac, input_shape=base_model.layers[-i-1].output_shape, name="rmac_out_relu")
        else :
            rmac_layer = Lambda(rmac.rmac, input_shape=base_model.layers[-i].output_shape, name="rmac_out_relu")

        out = rmac_layer(base_out)
        model = Model(base_model.input, out)

   
        taille=model.output_shape[1]
        taille_tot = taille_tot + taille
        y = model.predict(image)
        y=np.array(y[0])
        features=np.hstack((features,y))
    #quit()
    return features, taille_tot

classifier_list = ["Xception","VGG16","MobileNet","ResNet50","DenseNet169"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--dossier', type=str,default='Cars_Results_Sortie',
        help='path to image requete '
    )
    parser.add_argument(
        '--base', type=str,default='new_cars_cools_rmac_plus',
        help='modele  '
    )
    
    FLAGS = parser.parse_args()
    datasets = FLAGS.base
    dossier = FLAGS.dossier
    old_dataset="../../DataSets/"+datasets


    for id_classifier in range(0,len(classifier_list)):
        classifier=classifier_list[id_classifier]
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

        
        result_path = "../../normal/Cars/"+dossier+"/"+classifier

        model_final_of_classification=os.path.join(result_path, classifier+'_final.h5')
        
        model=load_model(model_final_of_classification)
        position= get_position (model,classifier)
        features = [] #Stocker les caractérstiques
        folder_features="Features_"+dossier
        classifier_path=folder_features+"/"+classifier
        if os.path.exists(classifier_path) == False:
            os.makedirs(classifier_path)
            features_path = classifier_path +"/features"
            if os.path.exists(features_path) == False:
                os.makedirs(features_path)
            pas =0
            for j in os.listdir(old_dataset) :
                if not j.endswith(".jpg"):
                    continue
                print(j)
                label, _, _,_,number = j.split("_")
                number = number.split(".")[0]
                number = int(number)
                data = os.path.join(old_dataset, j)
                #print (data)
                if not data.endswith(".jpg"):
                    continue
                file_name = os.path.basename(data)
                # load an image from file
                image = load_img(data, target_size=input_size)
                # convert the image pixels to a numpy array
                image = img_to_array(image)
                # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                # predict the probability across all output classes
                # Model 1
                features_msrmac, taille = MSRMAC(model, image, position) 
                #features_msrmac = np.array(features_msrmac[0])  
                np.savetxt(features_path+"/"+os.path.splitext(file_name)[0]+".txt",features_msrmac)
                features.append((data,features_msrmac,number))
                print (pas)
                pas = pas+1
            features=sorted(features,key=itemgetter(2))
            with open(classifier_path+"/features.txt", "wb") as output:
                pickle.dump(features, output)
        
    
