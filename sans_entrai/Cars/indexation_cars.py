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
#from keras.applications.efficientNetB7 import efficientNetB7
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#VAR


path_data_sets="../../DataSets/"
old_dataset=path_data_sets+"cars_cools"
dataset_name=path_data_sets+'new_cars_cools'
dataset_path = os.path.join('./', dataset_name)
new_base_dir=path_data_sets+"new_cars_cools_val_train"



classifier_list = ["Xception","VGG16","VGG19","ResNet50","InceptionV3","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201","efficientNetB7","NASNetMobile"]
#classifier_list = ["Xception"]



#indexation 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--type', type=str,default='Sortie',
        help='Layers type'
    )
   

    FLAGS = parser.parse_args()
    layer_type = FLAGS.type

    for id_classifier in range(0,len(classifier_list)):
        classifier=classifier_list[id_classifier]
        model_filename = 'model-'+classifier+'.h5'

        if classifier=="InceptionResNetV2" or classifier=="SqueezeNet" or classifier=="Xception" or classifier=="InceptionV3" or classifier=="InceptionResNetV2" or classifier =="DenseNet121" or classifier =="DenseNet169" or classifier=="DenseNet201" :
            img_height = 299
            img_width = 299
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)
        elif classifier=="VGG16" or classifier=="VGG19" or classifier=="ResNet50" or classifier=="MobileNet" or classifier=="NASNetMobile" :
            img_height = 224
            img_width = 224
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)
        elif classifier=="NASNetLarge" :
            img_height = 331
            img_width = 331
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)  
        

        result_path = "Cars_Results_"+layer_type+"/"+classifier
        model_final_of_classification=os.path.join(result_path, classifier+'_final.h5')
    
        
        model=load_model(model_final_of_classification)
        model2 = Model(inputs=model.input, outputs=model.layers[-2].output)
        # indexation 

        features = [] #Stocker les caractérstiques
        folder_features="Cars_Features_Train_"+layer_type
        classifier_path=folder_features+"/"+classifier
        #features_path = classifier_path +"/features"
        if os.path.exists(classifier_path) == False:
            os.makedirs(classifier_path)
            features_path = classifier_path +"/features"
            if os.path.exists(features_path) == False:
                os.makedirs(features_path)
            pas =0
            for k in os.listdir(dataset_name) :
                data1 = os.path.join(dataset_name, k)
                print(data1)
                for j in os.listdir(data1):
                    if not j.endswith(".jpg"):
                        continue
                    print(j)
                    num_marque, num_modele, marque, modele, number  = j.split("_") 
                    number = number.split(".")[0]
                    number = int(number)
                    data = os.path.join(data1, j)
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
                    feature = model2.predict(image)
                    feature = np.array(feature[0])  
                    np.savetxt(features_path+"/"+os.path.splitext(file_name)[0]+".txt",feature)
                    features.append((data,feature,number))
                    #features.append((number,feature))
                    print (pas)
                    pas = pas+1
                features=sorted(features,key=itemgetter(0))
                with open(classifier_path+"/features.txt", "wb") as output:
                    pickle.dump(features, output)
    
