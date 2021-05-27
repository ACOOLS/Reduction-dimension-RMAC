import argparse
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import os
import os.path
from os import path
import time
import numpy as np
from numpy import linalg as LA
from rmac_plus_util import *
from tensorflow.keras.layers import Input
import pickle
import tensorflow as tf
from shutil import copyfile



L = 3
topResultsQE = 5
nFiles = 100
largeScaleRetrieval = False

classifier_list = ["VGG16","MobileNet","Xception","ResNet50","DenseNet169"]
layers= ["block5_pool","conv_pw_13_relu","block14_sepconv2_act","conv5_block3_out","conv5_block32_concat"]

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
        '--base', type=str,default='new_cars_cools_rmac_plus',
        help='modele  '
    )
    FLAGS = parser.parse_args()
    datasets = FLAGS.base
    dossier = FLAGS.dossier

    for id_classifier in range(0,len(classifier_list)):
        classifier=classifier_list[id_classifier]
        layer=layers[id_classifier]

        result_path = "../../normal/Cars/"+dossier+"/"+classifier
        model_final_of_classification=os.path.join(result_path, classifier+'_final.h5')
        try:
            del base_model
        except NameError:
            print("error") 
        base_model=load_model(model_final_of_classification)
        base_model.summary()
        #quit()
        #base_model = VGG16(weights=model_final_of_classification, include_top=False, input_shape=(None,None,3))
    
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(str(layer)).output)

        dataset=datasets
        print("-------------------------------------------------")
        print('Parameters')
        print('Dataset: ' + str(dataset))

        datasetPCA = datasets
        print('PCA dataset: ' + str(datasetPCA))
        print('Model : ' + str(classifier))
        print('Layer: ' + str(layer))
        print('R-MAC descriptors with ' + str(L) + ' scales')

        resolutionLevel = 3
        print('Multi-resolution activated (3 scales: original, +25%, -25% on the largest side)')


        print("Query expansion. Top results used for QE: " + str(topResultsQE))

        if (largeScaleRetrieval):
            print("Activate large scale retrieval of",nFiles,"k files")

        print("------------------------------------------------")

        path_results="RMAC_"+dossier+"/"+classifier
        if os.path.exists(path_results) == False:
            os.makedirs(path_results)
        url = path_results+"/" + dataset + "/" + classifier + "_L" + str(L)
        savingUrl = datasetPCA + "_"+str(classifier)

        url += "_multiResolution_pca" + datasetPCA
        top_results=path_results+"/moteur_de_recherche"
        if path.exists(path_results+'/W'+savingUrl+'.npy') == False:
            try:
                del PCAImages
            except NameError:
                print("PCAImages remove error") 
            PCAImages = readTraining(datasetPCA, False,0)
            print('PCA with '+str(len(PCAImages))+' images')
            try:
                del PCAMAC
            except NameError:
                print("PCAMAC remove error") 
            PCAMAC = extractFeatures(PCAImages, model, True, L, classifier, resolutionLevel)
            try:
                del W
                del Xm
            except NameError:
                print("W Xm remove error") 
            W, Xm = learningPCA(PCAMAC)
            np.save(path_results+'/W'+savingUrl+'.npy',W)
            np.save(path_results+'/Xm'+savingUrl+'.npy',Xm)
        else :
            #after first execution comment the above snippet for the creation of the matrix W e Xm, usefull for the next PCA
            try:
                del W
                del Xm
            except NameError:
                print("W Xm remove error") 
            W = np.load(path_results+'/W' + savingUrl + '.npy')
            #print(W)
            #quit()
            Xm = np.load(path_results+'/Xm' + savingUrl + '.npy')

        # ------------------ DB images: reading, descripting and whitening -----------------------
        try:
            del DbImages
        except NameError:
            print("DbImages remove error") 
        DbImages = readTraining(dataset, True)
        print('DB contains ' + str(len(DbImages)) + ' images')

        if(path.exists(path_results+"/DbMAC.txt")==True):
            try:
                del DbMAC
            except NameError:
                print("DbMAC remove error") 
            with open(path_results+"/DbMAC.txt", "rb") as output:
                DbMAC=pickle.load(output)
        else :
            try:
                del DbMAC
            except NameError:
                print("DbMAC remove error")
            t1 = time.process_time()
            DbMAC = extractFeatures(DbImages, model, True, L, classifier, resolutionLevel)
            print("PCA-whitening")
            with open(path_results+"/DbMAC.txt", "wb") as output:
                pickle.dump(DbMAC, output)
        DbMAC = apply_whitening(DbMAC, Xm, W)
        regions = np.copy(DbMAC)
        nRegions = regions.shape[0]//len(DbImages)
        DbMAC = sumPooling(DbMAC, len(DbImages), False)
        

        # ------------------- query images: reading, descripting and whitening -----------------------

        img_class=["6_7_Hyundai_i30break_9283","6_7_Hyundai_i30break_9285","6_7_Hyundai_i30break_9299","9_1_Audi_Q5_12382","9_1_Audi_Q5_12451","9_1_Audi_Q5_12443","1_3_Kia_ceedsw_1801","1_3_Kia_ceedsw_1919","1_3_Kia_ceedsw_1841"]
        for img_indexes in img_class:
            image_nom_src="../../DataSets/"+dataset+"/"+img_indexes+".jpg"
            if os.path.exists("query") == False:
                os.makedirs("query")
            image_nom_des="query/"+img_indexes+".jpg"
            copyfile(image_nom_src, image_nom_des)
            queryImages, bBox = readTest(dataset, full=True)
            print('QUERY are ' + str(len(queryImages)) + ' images')

            queryMAC = extractFeatures(queryImages, model, True, L, classifier, resolutionLevel,bBox)
            #edsf)d
            queryMAC = apply_whitening(queryMAC, Xm, W)
            queryMAC = sumPooling(queryMAC, 1, False)
            print("Query descriptors saved!")

            sortie_list =  [20,50,100,500]
            for i_list in range(0,len(sortie_list)) :
                sortie=sortie_list[i_list]
                finalReRank,temps = retrieveRegionsNEW(queryMAC, regions, topResultsQE,url, queryImages, DbImages, dataset,top_results,sortie,classifier)
                print("AVG query time:",temps,"s")
                to_save=os.path.join(path_results,"moteur_de_recherche/TEMP_"+img_indexes+"_"+str(sortie)+".txt")
                with open(to_save, "w") as output:
                    output.write("Temps d execution : %s secondes ---" % (temps))
            os.remove(image_nom_des) 



