# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:25:58 2023

@author: rober
"""







import cv2
import numpy as np
from scipy.spatial import distance
import tensorflow.keras.backend as K
import tensorflow.keras.models as Models
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from sklearn import preprocessing


data_path = 'imagenes'


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
    
    emb = feature_extractor.predict(img)
    emb_norm = preprocessing.normalize(emb, norm = 'l2', axis = 1, copy = True,
                                       return_norm = False)
    
    return emb_norm


""""Comparamos si son la misma persona a través de una simple.Detectamos las caras de las imagenes. Imagen0 y Imagen1"""

#Ejemplo para dos imágenes
img_face1=extract_faces(data_path+"/0.jpg")
img_face2=extract_faces(data_path+"/1.jpg")

"Comparamos si es la misma persona"

embedding = generate_embedding(img_face1)
embedding = np.asarray(embedding)
f1 = np.squeeze(embedding)
embedding = generate_embedding(img_face2)
embedding = np.asarray(embedding)
f2 = np.squeeze(embedding)

# Similaridad basada en el producto vectorial. Valores altos significan alta similaridad
sim=np.dot(f1, f2.T)

if sim>=0.50:
    print("Misma persona")
else:
    print("Diferentes identidades") 


################ TAREA 1 #########################

# Revisa y entiende el código

# Compara con la imagen 2 o 3 con las anterires  


################ TAREA 2 #########################

# Genera una pequeña bbdd a partir de imágenes de inet
# que contenga 3 imágenes por individuo y al menos 5 individuos

# Genera un script para comparar de forma automáticas imágenes de la bbdd            
    