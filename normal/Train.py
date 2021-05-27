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
zip_data_bases="../GHIM_20.zip"
current_folder="."

nbr_batch_size=8 
dataset_name='NEW_GHIM' #@param ["NEW_GHIM"]
new_base_dir="NEW_GHIM_VAL_TRAIN"
dataset_path = os.path.join('./', dataset_name)
old_dataset="../GHIM-20"
classes_path = "./classes.txt"
csv_path = 'result.csv'

epochs = 100

seed = 1

classifier_list = ["ResNet50","InceptionV3","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201"]
train_input_paths,val_input_paths,train_labels,val_labels=[],[],[],[]


#FUNCTIONS

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


def new_format_classification (old_dataset,new_base):
    classe_list=[]
    for j in os.listdir(old_dataset) :
        data = os.path.join(dataset_path, j)

        if not data.endswith(".jpg"):
            continue
        file_name = os.path.basename(data)
        classe_list.append(file_name.split('_')[0])
    classe_list=list(set(classe_list))

    for k in range(0,len(classe_list)):
        if os.path.exists(new_base+"/"+str(classe_list[k])) == False:
            os.makedirs(new_base+"/"+str(classe_list[k]))

    for j in os.listdir(old_dataset) :
        data = os.path.join(old_dataset, j)

        if not data.endswith(".jpg"):
            continue
        file_name = os.path.basename(data)
        id=file_name.split('_')[0]
        shutil.copy2(data, new_base+"/"+str(id)+"/"+file_name)
    command = os.popen('ls NEW_GHIM > classes.txt')

def dowload_data_bases(url,zip_data_bases):
    print('Beginning file Databases with urllib2...')
    urllib.request.urlretrieve(url, zip_data_bases)
    with zipfile.ZipFile(zip_data_bases, 'r') as zip_ref:
        zip_ref.extractall(current_folder)


def val_train_bases(new_base_dir,dataset_name):
		# # Creating Train / Val / Test folders (One time use)
	bases = dataset_name

	test_files=0
	val_files=0
	train_files=0
	all_files=0
	train = None
	val = None
	test = None
	train_FileNames = None
	val_FileNames = None 
	test_FileNames = None
	train=new_base_dir +'/train'
	val=new_base_dir +'/val'
	test=new_base_dir +'/test'

	if os.path.exists(train) == False:
	  os.makedirs(train)
	if os.path.exists(val) == False:
	  os.makedirs(val)
	if os.path.exists(test) == False:
	  os.makedirs(test)


	# Creating partitions of the data after shuffeling
	#currentCls = bases
	src = dataset_name # Folder to copy images from

	for j in os.listdir(src):
	    class_path = os.path.join(src, j)
	    allFileNames=[]
	    for k in os.listdir(class_path):
	      allFileNames.append(j+"/"+k)
	    if os.path.exists(train+"/"+j) == False:
	      os.makedirs(train+"/"+j)
	    if os.path.exists(val+"/"+j) == False:
	      os.makedirs(val+"/"+j)
	    if os.path.exists(test+"/"+j) == False:
	      os.makedirs(test+"/"+j)
	    np.random.shuffle(allFileNames)
	    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
	                                                              [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
	    train_FileNames = [name for name in train_FileNames.tolist()]
	    train_files=train_files+len(train_FileNames)
	    for i in range (0,len(train_FileNames)):
	      copyfile(src+"/"+train_FileNames[i], train+"/"+train_FileNames[i])
	    
	    val_FileNames = [name for name in val_FileNames.tolist()]
	    val_files=val_files+len(val_FileNames)
	    for i in range (0,len(val_FileNames)):
	      copyfile(src+"/"+val_FileNames[i], val+"/"+val_FileNames[i])
	    
	    test_FileNames = [name for name in test_FileNames.tolist()]
	    test_files=test_files+len(test_FileNames)
	    for i in range (0,len(test_FileNames)):
	      copyfile(src+"/"+test_FileNames[i], test+"/"+test_FileNames[i])

	print('Total images: ', train_files+val_files+test_files)
	print('Training: ', train_files)
	print('Validation: ', val_files)
	print('Testing: ', test_files)
	return train_files,val_files,test_files,train,val,test


#entrainnement
if(path.exists(zip_data_bases)==False):
	dowload_data_bases("https://github.com/ACOOLS/Memoire/releases/download/1/GHIM-20.zip",zip_data_bases)
if os.path.exists("GHIM-20库") == True:
	if os.path.exists(old_dataset) == False:
		command = os.popen('mv GHIM-20库 GHIM-20')
time.sleep(20)
if os.path.exists(dataset_name) == False: 
	new_format_classification (old_dataset,dataset_path)
time.sleep(10)

command = os.popen('ls NEW_GHIM > classes.txt')
time.sleep(10)
num_classes = sum(1 for line in open(classes_path))

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
    elif classifier=="NASNetLarge" :
        img_height = 331
        img_width = 331
        input_size=(img_height,img_height)
        input_shape=(img_width,img_height,3)  
    
    
    log={
        'case':id_classifier,
        'model':model_filename,
        'classifier':classifier,
        'epochs':epochs,
        'batch_size':nbr_batch_size,
        'val_loss':-1,
        'val_acc':-1,
    }
 
    result_path = "Results/"+classifier
    model_final_of_classification=os.path.join(result_path, classifier+'_final.h5')
    if path.exists(model_final_of_classification) == False:
        train_files,val_files,test_files,train,val,test=val_train_bases(new_base_dir,dataset_name)
        
        train_datagen=None
        train_generator=None
        val_datagen=None
        validation_generator=None
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=40,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True,
                                              fill_mode='nearest')

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        if os.path.exists(result_path) == False:
          os.makedirs(result_path)
        model_path = os.path.join(result_path, model_filename)
        model_path_epoch=result_path+"/model_of_epochs"
        if os.path.exists(model_path_epoch) == False:
            os.makedirs(model_path_epoch)
        LOG_DIR = result_path+'/log'
        tbCallBack = TensorBoard(log_dir=LOG_DIR, histogram_freq=0,
                            write_graph=True,
                            write_grads=True,
                            batch_size=nbr_batch_size,
                            write_images=True)

        try:
            del base_model
        except NameError:
            print("error") 
        base_model = str_to_class(classifier)(include_top=False, weights='imagenet', input_shape=input_shape)  # La pouvez tester différentes architectures
        # create a custom top classifier
        print(base_model.output_shape[3])
        sortie=int(base_model.output_shape[3])
        sortie= sortie
        print(sortie)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(int(sortie), activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.inputs, outputs=predictions)
        
        for layer in model.layers:
            layer.trainable = True
        #choix de l'optimizer
        optimizer_choice_list = ["sgd","RMSprop","Adagrad","Adam","Adadelta","Adamax"]
        optimizer_choice=optimizer_choice_list[0]
        learning_rate = 0.001
        csv_path = result_path+'/'+classifier+'.csv'
        csv_logger = tf.keras.callbacks.CSVLogger(csv_path)
        
        if optimizer_choice=="sgd":
          optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.0, nesterov=False)
        elif optimizer_choice=="RMSprop":
          optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
        elif optimizer_choice=="Adagrad":
          optimizer=keras.optimizers.Adagrad(lr=learning_rate)
        elif optimizer_choice=="Adam":
          optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif optimizer_choice=="Adadelta":
          optimizer=keras.optimizers.Adadelta(lr=learning_rate, rho=0.9)
        elif optimizer_choice=="Adamax":
          optimizer=keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        patience= 4
        monitor_checkpoint='acc' 
        monitor_early_stopping='loss' 
        save_best='True'
        # Stopper, CheckPoint
        stopper=EarlyStopping(monitor=monitor_early_stopping,
                                      min_delta=0,
                                      patience=patience,
                                      verbose=1, 
                                      mode='auto')    
        ckpt_save =model_path_epoch+ '/'+'classification_ep{epoch}_valloss{val_loss:.3f}.h5'

        checkpoint = ModelCheckpoint(ckpt_save, monitor=monitor_checkpoint, verbose=1, save_best_only=save_best, mode='auto')

        #callback_list = [checkpoint, stopper]
        callback_list = [checkpoint,stopper,csv_logger,tbCallBack]
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        train_generator = train_datagen.flow_from_directory(train,
                                                  target_size=(img_height, img_width),
                                                  batch_size=nbr_batch_size,
                                                  class_mode='categorical')
        validation_generator = val_datagen.flow_from_directory(val,
                                                      target_size=(img_height, img_width),
                                                      batch_size=nbr_batch_size,
                                                      class_mode='categorical')

        history=model.fit_generator(train_generator,
                        steps_per_epoch=train_files // nbr_batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=val_files // nbr_batch_size,
                        verbose=1,
                        callbacks=callback_list)
        
        # Sauvegarder le modèle final
        model.save(model_final_of_classification)
