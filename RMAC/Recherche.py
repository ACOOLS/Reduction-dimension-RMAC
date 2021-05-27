from multiprocessing import Pool, cpu_count
from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
import numpy as np
import os
import csv
#import keras
import operator
import ntpath

import pickle
import warnings
from operator import itemgetter

import argparse
import cv2

from shutil import copyfile
import os.path
from os import path
from matplotlib.pyplot import imread
#from keras.backend.tensorflow_backend import set_session
from scipy import spatial
import numpy as np

import time


# For automated test log result directly in the csv
import csv
import sys
import math
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


#classifier_list = ["ResNet50_add_16","Xception_block14_sepconv2"]
classifier_list = ["DenseNet169","MobileNet","Xception","ResNet50","VGG16"]

#FUNCTIONS
def get_feature(img_index,lfeatures):
    return lfeatures[img_index]

def euclidianDistance(l1,l2):
    length = min(len(l1),len(l2))
    dist = cv2.norm(l1[:length] - l2[:length], cv2.NORM_L2)
    return dist


    
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
        #sous_dossier=img_test.split("_")[0]+"_"+img_test.split("_")[1]
        #print(img_test[0])
        imge_current=img_test[0]
        #print(imge_current)
        if(sortie <= 20):
            plt.figure(figsize=(5, 5))
            plt.imshow(imread(imge_current), cmap='gray', interpolation='none')
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
        image_name=os.path.basename(img_test[0])
        #print(img_test)
        position1=image_name.split('_')[0]
        position3=image_name.split('_')[1]
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
                #print(nom_image_plus_proches[j])
                plt.imshow(imread(nom_image_plus_proches[j][0]), cmap='gray', interpolation='none')
            
            cools=os.path.basename(nom_image_plus_proches_sans[j])
            n_marque, n_modele, marque, modele, num = cools.split('_')
            if(sortie <= 20):
                title = "Image proche n° "+str(j)+" (n° "+ num +")"
                plt.title(title)
            if position1==n_marque : 
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
        print("Moyenne des précisions - image {} : {}".format(img_test[0], MaP))
        print("Moyenne des rappels - image {} : {}".format(img_test[0], MaR))
        print("")
        with open(os.path.join(new_folder,"TEMP_MAP"+str(img_index)+"_"+str(sortie)+".txt"), "w") as output:
            output.write("MAP - image {} : {}".format(img_test[0], MaP))
            output.write("Temps de recherche : %s secondes ---" % (temps))
        
        with open(os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt"), 'w') as s: 
            for a in rp: 
                s.write(str(a) + '\n')

        ######################
        ## Tracer la courbe ##
        ######################
        """x = [] 
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
        fig.savefig(os.path.join(new_folder,"img_"+str(img_index)+"_"+str(sortie)+".png"))"""
    
    
    ########################################
    ## Calcul du RP moyen pour une classe ##
    ########################################
    """nbr_img = len(img_indexes)
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
    fig.savefig(os.path.join(new_folder,"RP_"+str(position1)+"_"+str(sortie)+"_classe"+str(position1)+".png"))"""


#Recherche

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
        features = []
        folder_features="Features_"+dossier
        classifier_path=folder_features+"/"+classifier
        features_path = classifier_path +"/output2"
        if not os.path.exists(features_path): 
            os.makedirs(features_path)
        features_file= classifier_path +"/features.txt"
        with open(features_file, 'rb') as filehandle:
            features=pickle.load(filehandle)
        
        img_class=[[9283,9285, 9299],[12382, 12451, 12443],[1801, 1919, 1841]]
        sortie_list =  [20,100]
        for i_list in range(0,len(sortie_list)) :
            sortie=sortie_list[i_list]
            for img_indexes in img_class:
                search(sortie, img_indexes,features,features_path,classifier,old_dataset)
        