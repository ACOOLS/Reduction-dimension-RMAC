from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import os
import argparse
#import tensorflow as tf


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

    parser.add_argument(
        '--classifier1', type=str,default='DenseNet169',
        help='modele 1  '
    )
    parser.add_argument(
        '--classifier2', type=str,default='VGG16',
        help='modele 2  '
    )
    parser.add_argument(
        '--classifier3', type=str,default='MobileNet',
        help='modele 3 '
    )

    FLAGS = parser.parse_args()
    datasets = FLAGS.base
    dossier = FLAGS.dossier
    classifier1=FLAGS.classifier1
    classifier2=FLAGS.classifier2
    classifier3=FLAGS.classifier3
    old_dataset="../../DataSets/"+datasets

    if datasets =="GHIM-20":
        sous_dossier="GHIM-20"
    else :
        sous_dossier="Cars"

    result_path1 = "../normal/"+sous_dossier+"/"+dossier+"/"+classifier1

    model_final_of_classification1=os.path.join(result_path1, classifier1+'_final.h5')
    model_1=load_model(model_final_of_classification1)

    result_path2 = "../normal/"+sous_dossier+"/"+dossier+"/"+classifier2

    model_final_of_classification2=os.path.join(result_path2, classifier2+'_final.h5')
    model_2=load_model(model_final_of_classification2)

    result_path3 = "../normal/"+sous_dossier+"/"+dossier+"/"+classifier3

    model_final_of_classification3=os.path.join(result_path3, classifier3+'_final.h5')
    model_3=load_model(model_final_of_classification3)

    NB_CLASSES = 20

    input_common = Input(shape=(224, 224, 3), name='input_common')

    model_1.name="classifier1"
    model_2.name="classifier2"
    model_3.name="classifier3"
    model_simple_output = model_1(input_common)
    model_complexe_output = model_2(input_common)
    model_complexe_output_2 = model_3(input_common)

    x = concatenate([model_simple_output, model_complexe_output,model_complexe_output_2])
    x = Dense((3 * NB_CLASSES), activation='relu')(x)
    x = Dense((3 * NB_CLASSES), activation='relu')(x)
    x = Dense((3 * NB_CLASSES)*3, activation='relu')(x)
    x = Dense((3 * NB_CLASSES)*3, activation='relu')(x)
    x = Dense(NB_CLASSES, activation='relu')(x)
    output = Dense(NB_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=input_common, outputs=output)

    model.summary()
    result_path_tot="../normal/"+sous_dossier+"/"+dossier+"/"+classifier1+"_"+classifier2+"_"+classifier3
    if os.path.exists(result_path_tot) == False:
        os.makedirs(result_path_tot)
    model_result_path_tot = result_path_tot+"/"+classifier1+"_"+classifier2+"_"+classifier3+"_final.h5"

    model.save(model_result_path_tot)


    model_json = model.to_json()
    with open(result_path_tot+"/model.json", "w") as json_file:
        json_file.write(model_json)
