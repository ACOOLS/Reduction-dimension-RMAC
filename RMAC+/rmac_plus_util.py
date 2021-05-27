from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import os
import os.path
from PIL import Image, ImageFile
import sys
import cmath
import numpy as np
from numpy import linalg as LA
import cv2
import csv
import operator
from sklearn.decomposition import PCA
import scipy
import math
import time
from sys import getsizeof
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from glob import glob
import shutil
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib.pyplot import imread
def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def readTraining(dataset, rotated=True, nFiles=0, debug=False):
    
    path = '../../DataSets/'+dataset+'/*.jpg'
    DbImages = np.sort(glob(path))  #da capire se funziona con Flickr1M
    return DbImages

def readTest(dataset, full=False, debug=False):
	bBox = []
	path = 'query/*.jpg'
	path_R= "query"
	queryImages=np.sort(glob(path))
	queryImages = np.sort(queryImages)
	print(queryImages)
	print("Creation of bBox list")
	for e in os.listdir(path_R):
		q_final = [ 0,1,0,1,0]
		bBox.append(q_final[1:])
	for i,q in enumerate(queryImages,0):
		img = cv2.imread(q)
		h = img.shape[0]
		w = img.shape[1]
		bBox[i][0] = 0
		bBox[i][2] = 1
		bBox[i][1] = 0
		bBox[i][3] = 1
	return queryImages,bBox

def calculateMAC(featureVector, listData): #max-pooling and l2-norm
	rows = featureVector.shape[1] * featureVector.shape[2]
	cols = featureVector.shape[3]
	features1 = np.reshape(featureVector, (rows, cols))
	features2 = np.amax(features1, axis = 0)
	features2 /= np.linalg.norm(features2, 2)
	listData.append(features2)

	return

def calculateRMAC(features, listData, L):
	W = features.shape[1]
	H = features.shape[2]
	# print("W",W,"H",H)

	for l in range(1,L+1):
		if (l==1):
			heightRegion = widthRegion = min(W,H)
			if (W<H):
				xRegions = 1
				yRegions = 2
			else:
				xRegions = 2
				yRegions = 1
		else:
			widthRegion = heightRegion = math.ceil(2*min(W,H)/(l+1))
			if (l==2):
				xRegions = 2
				yRegions = 3
			elif (l==3):
				xRegions = 3
				yRegions = 2

		if (widthRegion*xRegions < W): #not covered the image along width
			widthRegion = math.ceil(W/xRegions)
		if (heightRegion*yRegions < H):
			heightRegion = math.ceil(H/yRegions)

		coefW = W / xRegions
		coefH = H / yRegions

		# print("L:",l," w:",widthRegion," h:",heightRegion,"xRegions",xRegions,"yRegions",yRegions)

		for x in range(0,xRegions):
			for y in range(0,yRegions):
				initialX = round(x*coefW)
				initialY = round(y*coefH)
				finalX = initialX + widthRegion
				finalY = initialY + heightRegion
				if (finalX > W):
					finalX = W
					initialX = finalX - widthRegion
				if (finalY > H):
					finalY = H
					initialY =  finalY - heightRegion

				# print(" X ",initialX,":", finalX," Y ", initialY,":", finalY)

				featureRegion = features[:,initialX:finalX,initialY:finalY,:] #(old implementation)
				calculateMAC(featureRegion, listData)
	return

def resizeImg (img, i, delta):
    if delta != 0:
        w = img.size[0]
        h = img.size[1]
        newWidth = round(w + w*delta)
        newHeight = round(h + h*delta)
        img = img.resize((newWidth,newHeight))
    return img

def extractFeatures(imgs, model, RMAC, L, classifier, resolutionLevel, bBox=[], croppedActivations = False):
	listData = []
	deltas = [0, -0.25, 0.25]
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
	
	 
	for j in tqdm(range(0,len(imgs))):
		for i in range(0, resolutionLevel):
			img = load_img(imgs[j])
			img = resizeImg(img,i, deltas[i])
			input_size=(img.size[0],img.size[1])
			#print(input_size)
			img = load_img(imgs[j],target_size=input_size)
			x = img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			#print(x.shape)
			features = model.predict(x)
			#print(features)
			if (croppedActivations):
				startDim1 = math.floor(bBox[j][1]*features.shape[1])
				endDim1 = math.ceil(bBox[j][3]*features.shape[1])
				startDim2 = math.floor(bBox[j][0]*features.shape[2])
				endDim2 = math.floor(bBox[j][2]*features.shape[2])
				features = np.copy(features[:,startDim1:endDim1,startDim2:endDim2,:])
				# print(features.shape,"->", features2.shape)
			calculateMAC(features, listData)
			if (RMAC):
				calculateRMAC(features, listData, L)

	return listData

def learningPCA(listData):
	fudge = 1E-18
	X = np.matrix(listData)
	mean = X.mean(axis=0)
	# subtract the mean
	X = np.subtract(X, mean)
	# calc covariance matrix
	Xcov = np.dot(X.T,X)
	d,V = np.linalg.eigh(Xcov)
	test= d+fudge
	test2=[]
	for i in range (0,len(test)):
		print(cmath.sqrt(test[i]))
		test2.append(cmath.sqrt(test[i]))
	print(test)
	print(test2[0])
	#quit()
	D = np.diag(1. / np.asarray(test2))
	W = np.dot(np.dot(V, D), V.T)
	return W, mean

def apply_whitening(listData, Xm, W) :
	X = np.matrix(listData)
	X = np.subtract(X, Xm)
	Xnew = np.dot(X,W)
	Xnew /= LA.norm(Xnew,axis=1).reshape(Xnew.shape[0],1)
	return Xnew


def sumPooling(listData, numberImages, largeScaleRetrieval=False):
	newListData = []
	value = 0
	regions = listData.shape[0] // numberImages
	for i, elem in enumerate(listData, 1):
		value = np.add(value,elem)
		if (i%regions==0):
			value /= LA.norm(value, 2)
			newListData.append(value)
			value = 0
	if (not largeScaleRetrieval):
		print("Sum pooling of",regions,"regions. The descriptors are",len(newListData),"of shape",newListData[0].shape)
	return newListData

def extractAndWhiteningNEW(imgs, model, RMAC, L, resolutionLevel,Xm,W, limits=1000, pca=None):
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	tmpList = []
	finalList = []
	delta = 0.25
	for j in tqdm(range(0,len(imgs))):
		for i in range(0, resolutionLevel):
			img = image.load_img(imgs[j])
			img = resizeImg(img, i, delta)
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = model.predict(x)
			calculateMAC(features, tmpList)
			if (RMAC):
				calculateRMAC(features, tmpList, L)
		if ((j+1)%limits==0):
			tmpList = apply_whitening(tmpList, Xm, W)
			tmpList = sumPooling(tmpList, limits, True)
			finalList.extend(tmpList)
			tmpList = []

	print("Features len",len(finalList))
	return finalList

def write_results( url, queryImages,i, distances, DbImages, dataset, sortie,classifier,largeScaleRetrieval=False):
	nom_image_plus_proches = []
	nom_image_plus_proches_sans = []
	if (dataset=='new_cars_cools_rmac_plus'):
		if not os.path.exists(url):
			os.makedirs(url)
		print("query :")
		if queryImages[i] is not None :
			print(queryImages[i])
			file_query  = open(url+"/"+os.path.basename(queryImages[i])[:-4], "w")

			for elem in distances:
				file_query.write(os.path.basename(DbImages[elem[0]])[:-4])
				image_current=os.path.basename(DbImages[elem[0]])[:-4]
				nom_image_plus_proches.append(image_current)
				nom_image_plus_proches_sans.append(image_current.split("_")[0])
				file_query.write("\n")
			file_query.close()
			#########################################
			## Calcul du rappel et de la précision ##
			#########################################
			if(sortie <= 20):
				plt.figure(figsize=(5, 5))
				plt.imshow(imread(queryImages[i]), cmap='gray', interpolation='none')
				plt.title("Image requête")
			if(sortie <= 20):     
				plt.figure(figsize=(25, 25))
				plt.subplots_adjust(hspace=0.2, wspace=0.2)
			
			MaP = np.array([])
			MaR = np.array([])
			rappel_precision = [] 
			rp = [] 
			image_req = os.path.basename(queryImages[i])[:-4]
			position1, _,_,_,img_index=image_req.split("_")
			#position1=(int(img_index)-1)//500
			for j in range(sortie): 
				if(sortie <= 20):
					plt.subplot(sortie/4,sortie/5,j+1)
					plt.imshow(imread("../../DataSets/"+dataset+"/"+nom_image_plus_proches[j]+".jpg"), cmap='gray', interpolation='none')
				position2=(int(nom_image_plus_proches_sans[j]))
				title = "Image proche n°"+str(j)
				if(sortie <= 20):
					plt.title(title)
				#print("Position1 :"+ str(position1))
				#print("Position2 :"+ str(position2))
				if int(position1)== int(position2): 
					rappel_precision.append("pertinant") 
				else: 
					rappel_precision.append("non pertinant") 
			if(sortie <= 20):
				plt.savefig(url+"/Output_image_"+os.path.basename(queryImages[i])[:-4]+"_top"+str(sortie)+"_"+classifier+".pdf")
			#print(rappel_precision)
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
			print("Moyenne des précisions - image {} : {}".format(img_index, MaP))
			print("Moyenne des rappels - image {} : {}".format(img_index, MaR))
			print("")
			with open(os.path.join(url,"TEMP_MAP_"+str(img_index)+"_"+str(sortie)+".txt"), "w") as output:
				output.write("Moyenne des précisions - image {} : {}".format(img_index, MaP))
			
			with open(os.path.join(url,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt"), 'w') as s: 
				for a in rp: 
					s.write(str(a) + '\n')

			######################
			## Tracer la courbe ##
			######################
			x = [] 
			y = []
			fichier = os.path.join(url,"img_"+str(img_index)+"_"+str(sortie)+"_RP.txt")
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
			fig.savefig(os.path.join(url,"img_"+str(img_index)+"_"+str(sortie)+".png"))
	return


def retrieve(queryMAC, DbMAC, topResultsQE, url, queryImages, DbImages, dataset, largeScaleRetrieval=False):
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	reRank = []

	for i,q in enumerate(queryMAC,0):
		distances = {}
		qNP = np.asarray(q)
		for j,dbElem in enumerate(DbMAC,0):
			dbNP = np.asarray(dbElem)
			distances[j] = np.linalg.norm(qNP-dbNP)
		finalDict = sorted(distances.items(), key=operator.itemgetter(1))

		reRank.extend(list(finalDict)[:topResultsQE])

		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)

	return reRank

def retrieveRegionsNEW(queryMAC, regions, topResultsQE,url, queryImages, DbImages, dataset,top_results,sortie,classifier, largeScaleRetrieval=False):


	reRank = []

	nRegions = regions.shape[0]//len(DbImages)
	retrieval1 = time.process_time()
	for i,q in enumerate(queryMAC,0):
		print("i : " + str(i))
		distances = {}
		bestRegions = []
		qNP = np.asarray(q)
		for j,dbElem in enumerate(regions,0):
			dbNP = np.asarray(dbElem)
			indexDb = j//nRegions
			d = np.linalg.norm(qNP-dbNP)
			if (indexDb in distances):
				if (distances[indexDb][0]>d):
					distances[indexDb] = [d,j]
			else:
				distances[indexDb] = [d,j]
		finalDict = sorted(distances.items(), key=operator.itemgetter(1))
		#print(finalDict[:20])
		reRank.extend(list(finalDict)[:topResultsQE])
		
		write_results(top_results, queryImages, i, finalDict, DbImages, dataset, sortie,classifier, largeScaleRetrieval,)
	retrieval2 = time.process_time() - retrieval1
	temps=round(retrieval2/len(queryImages),2)
	return reRank,temps

def retrieveQE(queryMAC, DbMAC, topResultsQE, url, queryImages, DbImages, reRank, dataset, largeScaleRetrieval=False):

	url += '_avgQE'
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	finalReRank = []

	for i,q in enumerate(queryMAC,0):
		distances2 = {}
		qNewNP = np.asarray(q)
		for top_results in range(0,int(topResultsQE)):
			index = top_results+(topResultsQE*i)
			dbOLD = np.asarray(DbMAC[reRank[index][0]])
			qNewNP += dbOLD
		qNewNP = np.divide(qNewNP,float(topResultsQE))
		for j,dbElem in enumerate(DbMAC,0):
			dbNP = np.asarray(dbElem)
			distances2[j] = np.linalg.norm(qNewNP-dbNP)
		finalDict = sorted(distances2.items(), key=operator.itemgetter(1))

		finalReRank.extend(list(finalDict))
		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)

	return finalReRank

def retrieveQERegionsNEW(queryMAC, regions, topResultsQE, url, queryImages, DbImages, reRank, dataset, largeScaleRetrieval=False):

	url += '_avgQE'
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	finalReRank = []

	nRegions = regions.shape[0]//len(DbImages)

	for i,q in enumerate(queryMAC,0):
		distances2 = {}
		qNewNP = np.asarray(q)
		for top_results in range(0,int(topResultsQE)):
			index = top_results+(topResultsQE*i)
			dbOLD = np.asarray(regions[reRank[index][1][1]])
			qNewNP += dbOLD
		qNewNP = np.divide(qNewNP,float(topResultsQE))
		for j,dbElem in enumerate(regions,0):
			dbNP = np.asarray(dbElem)
			indexDb = j//nRegions
			d = np.linalg.norm(qNewNP-dbNP)
			if (indexDb in distances2):
				if (distances2[indexDb]>d):
					distances2[indexDb] = d
			else:
				distances2[indexDb] = d

		finalDict = sorted(distances2.items(), key=operator.itemgetter(1))
		finalReRank.extend(list(finalDict))
		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)
	return finalReRank
