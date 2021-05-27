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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#VAR
nbr_batch_size=16

old_dataset="cars_cools"
dataset_name='../../DataSets/new_cars_cools'
dataset_path = os.path.join('./', dataset_name)
new_base_dir="../../DataSets/new_cars_cools_val_train"


classes_path = "./classes.txt"
csv_path = 'result.csv'

epochs = 150

seed = 1

#classifier_list = ["VGG19","ResNet50","InceptionV3","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201","NASNetLarge","NASNetMobile"]
classifier_list = ["DenseNet169"]
#classifier_list = ["Xception","VGG16","VGG19","ResNet50","InceptionV3","InceptionResNetV2","DenseNet201"]
train_input_paths,val_input_paths,train_labels,val_labels=[],[],[],[]
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


def new_format_classification (old_dataset,new_base):
    marque_cpt=0
    image_cpt=0
    for i in os.listdir(old_dataset) :
        marque = os.path.join(old_dataset, i)
        modele_cpt=0
        for j in os.listdir(marque) :
            modele = os.path.join(marque, j)
            for k in os.listdir(modele) :
                image_file=os.path.join(modele, k)
                image_cv=cv2.imread(image_file)
                if image_cv is not None :
                    print( "Image : "+ image_file)
                    new_classe= str(marque_cpt)+"_"+str(modele_cpt)
                    print("New classe : "+new_classe)
                    if os.path.exists(new_base+"/"+new_classe) == False:
                        os.makedirs(new_base+"/"+new_classe)
                    ii=i.replace(" ", "")
                    jj=j.replace(" ", "")
                    file_name = str(marque_cpt)+"_"+str(modele_cpt)+"_"+ii+"_"+jj+"_"+str(image_cpt)+".jpg"
                    print("new file name : "+file_name)
                    copyfile(image_file, new_base+"/"+new_classe+"/"+file_name)
                    image_cpt=image_cpt+1
            modele_cpt=modele_cpt+1
        marque_cpt=marque_cpt+1
    command = os.popen('ls ../../DataSets/new_cars_cools > classes.txt')



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
if os.path.exists(dataset_name) == False:
    os.makedirs(dataset_name)

#new_format_classification (old_dataset,dataset_path)
time.sleep(10)

num_classes = sum(1 for line in open(classes_path))

#train_input_paths,val_input_paths,train_labels,val_labels=get_classes(classes_path,num_classes)
for id_classifier in range(0,len(classifier_list)):
    classifier=classifier_list[id_classifier]
    model_filename = 'model-'+classifier+'.h5'
    img_height = 224
    img_width = 224
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
    
    result_path = "Cars_Results_224_DenseNet/"+classifier
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
        print(sortie)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(sortie, activation='relu')(x)
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
        
        opt = Adam(lr=0.0001)
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
        #model = multi_gpu_model(model, gpus=3)
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

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
    else :
        model=load_model(model_final_of_classification)
      

    
