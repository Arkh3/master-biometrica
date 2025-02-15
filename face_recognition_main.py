# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:25:58 2023

@author: rober, Sofía, Hajar, Andrés
"""




import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow.keras.models as Models

from pathlib import Path
from sklearn import preprocessing
from scipy.spatial import distance
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image


#Detector de caras basado en redes convolucionales 2D
detector_dnn = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

#CARGAMOS EL MODELO DE RECONOCIMIENTO FACIAL basado en Resnet-50 y entrenado con VGG-Face
model_file = 'resnet50.h5'
model = Models.load_model(model_file)
last_layer = model.get_layer('avg_pool').output
feature_layer = Flatten(name = 'flatten')(last_layer)
feature_extractor = Models.Model(model.input, feature_layer)

def extract_faces(file_img):
    #Función que a partir de una imagen, detecta y recorta la cara
    img =  cv2.imread(file_img, cv2.IMREAD_UNCHANGED)
    centro=[int(img.shape[1]/2),int(img.shape[0]/2)]
    (h, w) = img.shape[:2]
    inputBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1, (300, 300), (104, 177, 123))
    detector_dnn.setInput(inputBlob)
    detections = detector_dnn.forward()
    list_box=[] 
    distancia=[]
    #Detectar la cara y recortarla
    if detections.shape[2]<=0:
             print("Cara no detectada")
    else:
             for i in range(0, detections.shape[2]):
                     prediction_score = detections[0, 0, i, 2]
                     if prediction_score >0.8:
                      
                       box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                       iz,arri,dere,abajo=box1.astype("int")
                       if iz<0 or iz>img.shape[1]:
                           iz=0
         
                       if arri<0 or arri>img.shape[0]:
                          arri=0
             
                       if dere> img.shape[1]:
                          dere= img.shape[1]
                   
                       if abajo> img.shape[0]:
                          abajo= img.shape[0]
                       list_box.append([iz,arri,dere,abajo])
                       centro1=[int((dere+iz)/2),int((abajo+arri)/2)]
                       distancia.append(distance.euclidean(centro, centro1))
                       
             if len(distancia)>0:          
                 box=list_box[np.argmin(distancia)]
                 imagen_copia=img
                 imagen_copia=imagen_copia[box[1]:box[3],box[0]:box[2]]
                 list_box=[] 
                 distancia=[]
             else:
                 print("No detectado")
                 imagen_copia=[]
    return imagen_copia

def preprocess_input(x, data_format=None, version=2):
    #Función de pre-procesado de la imagen antes de ser introducida en el modelo resnet-50 
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def generate_embedding(img):
    #Función que genera un embedding a partir de una cara y el modelo entrenado
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    #img = image.load_img(files)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img, version = 2)
    emb = feature_extractor.predict(img, verbose=0)
    emb_norm = preprocessing.normalize(emb, norm = 'l2', axis = 1, copy = True,
                                       return_norm = False)
    return emb_norm


################################## CÓDIGO DE LA PRÁCTICA ##################################


def extract_features(image_path, crop_image=False):
    """ Given an image generate its features for comparison.
        NOTE: if the face already covers all of the image,
        extract_faces should be set to False. """ 
    
    if crop_image:
        image = extract_faces(image_path)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
    embedding = generate_embedding(image)
    embedding = np.asarray(embedding)

    return np.squeeze(embedding)


def compare_images(img_path1, img_path2, threshold=0.5, crop_image=False):
    """ Compara dos imágenes y devuelve si pertenecen a la misma persona """
    
    f1 = extract_features(img_path1, crop_image)
    f2 = extract_features(img_path2, crop_image)
    
    similarity = np.dot(f1, f2)
    
    if similarity >= threshold:
        print(f"Las imágenes pertenecen a la misma persona con similitud {similarity:.2f}")
        return True
    else:
        print(f"Las imágenes pertenecen a diferentes personas con similitud {similarity:.2f}")
        return False


# Crear base de datos de embeddings
def create_embeddings_db(bbdd_path):
    """ Crea una base de datos de embeddings a partir de las imágenes en bbdd_path """
    print(" Creando la base de datos de embeddings...")
    embeddings_db = {}
    bbdd_path = Path(bbdd_path) # Convertir a Path para manejar rutas correctamente
    for person in os.listdir(bbdd_path):
        person_path = bbdd_path / person  # Construcción segura de rutas
        
        if person_path.is_dir():
            embeddings_db[person] = []
            
            for img_name in os.listdir(person_path):
                img_path = person_path / img_name  # Construcción segura de rutas

                embedding = extract_features(img_path)
                embeddings_db[person].append(embedding)
    
    print(" Base de datos de embeddings creada con éxito.")
    return embeddings_db


def calculate_far_frr_plot(embeddings_db):
    """ Calcula FAR y FRR para diferentes umbrales y genera un gráfico. """

    thresholds=np.linspace(0, 1, 50)
    
    fars = []
    frrs = []
 
    people = list(embeddings_db.keys())

    for threshold in thresholds:
        false_accepts = 0
        false_rejects = 0
        total_genuine = 0
        total_impostor = 0

        for person in people:
            embeddings = embeddings_db[person]

            #  **Calcular FRR (False Rejection Rate)**
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j].T)  # Comparar imágenes de la MISMA persona
                    total_genuine += 1
                    if sim < threshold:
                        false_rejects += 1  # Error: No reconoce a la persona correcta

            #  **Calcular FAR (False Acceptance Rate)**
            for other_person in people:
                if other_person != person:
                    for emb1 in embeddings:
                        for emb2 in embeddings_db[other_person]:
                            sim = np.dot(emb1, emb2.T)  # Comparar imágenes de DIFERENTES personas
                            total_impostor += 1
                            if sim >= threshold:
                                false_accepts += 1  # Error: Acepta a la persona equivocada

        # Calcular tasas FAR y FRR
        far = false_accepts / total_impostor if total_impostor > 0 else 0
        frr = false_rejects / total_genuine if total_genuine > 0 else 0

        fars.append(far)
        frrs.append(frr)

    # **Generar el gráfico**
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fars, label="False Acceptance Rate (FAR)", color="red")
    plt.plot(thresholds, frrs, label="False Rejection Rate (FRR)", color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR vs FRR en función del umbral")
    plt.legend()
    plt.grid()
    plt.show()

    return fars, frrs


def calcular_histograma(embeddings_db, thresholds=np.linspace(0, 1, 50)):
    """ Calcula FAR y FRR para diferentes umbrales y genera un gráfico. """

    fars = []
    frrs = []
    same_person=[]
    different_person=[]
    
    people = list(embeddings_db.keys())

    for threshold in thresholds:
        false_accepts = 0
        false_rejects = 0
        total_genuine = 0
        total_impostor = 0

        for person in people:
            embeddings = embeddings_db[person]

            #  **Calcular FRR (False Rejection Rate)**
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j].T)  # Comparar imágenes de la MISMA persona
                    different_person.append(sim)
                    total_genuine += 1
                    if sim < threshold:
                        false_rejects += 1  # Error: No reconoce a la persona correcta

            #  **Calcular FAR (False Acceptance Rate)**
            for other_person in people:
                if other_person != person:
                    for emb1 in embeddings:
                        for emb2 in embeddings_db[other_person]:
                            sim = np.dot(emb1, emb2.T)  # Comparar imágenes de DIFERENTES personas
                            same_person.append(sim)
                            total_impostor += 1
                            if sim >= threshold:
                                false_accepts += 1  # Error: Acepta a la persona equivocada

        # Calcular tasas FAR y FRR
        far = false_accepts / total_impostor if total_impostor > 0 else 0
        frr = false_rejects / total_genuine if total_genuine > 0 else 0

        fars.append(far)
        frrs.append(frr)

    # hacer el histogramama de las dos similitudes, que estan metidos en el array, sam_person y different_person
    plt.figure(figsize=(8, 6))
    plt.hist(same_person, bins=50, alpha=0.5, color='b', label='same person')
    plt.hist(different_person, bins=50, alpha=0.5, color='r', label='different person')
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR vs FRR en función del umbral")
    plt.legend()
    
    return same_person, different_person