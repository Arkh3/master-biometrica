# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:25:58 2023

@author: rober, Sofía, Hajar, Andrés
"""


import os
import cv2
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow.keras.models as Models


from pathlib import Path
from scipy.spatial import distance

from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical



random.seed(42)
np.random.seed(42)

# Rutas a los archivos:
data_dir = os.path.join('..', 'data')
models_dir = os.path.join('..', 'models')

bbdd_10_ppl_dir = os.path.join(data_dir, 'bbdd_10_personas')
embeddings_10_ppl_dataset_path = os.path.join(data_dir, 'embeddings_10_ppl_dataset.pkl')
imgs_dataset_dir = os.path.join(data_dir,'DiveFace4K_120', '4K_120')
embeddings_dataset_path = os.path.join(data_dir,'main_embeddings_dataset.pkl')

deploy_path = os.path.join(models_dir, 'deploy.prototxt.txt')
detector_path = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
model_file = os.path.join(models_dir, 'resnet50.h5')


#Detector de caras basado en redes convolucionales 2D
detector_dnn = cv2.dnn.readNetFromCaffe(deploy_path, detector_path)

#CARGAMOS EL MODELO DE RECONOCIMIENTO FACIAL basado en Resnet-50 y entrenado con VGG-Face
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

def load_datasets():
    if os.path.isfile(embeddings_10_ppl_dataset_path):  
        print("Embeddings dataset already exists. Loading...")

    else:
        print("Embeddings dataset does not exists. Creating it...")
        create_embeddings_10_ppl_dataset(bbdd_10_ppl_dir, embeddings_10_ppl_dataset_path)

    if os.path.isfile(embeddings_dataset_path):
        print("Embeddings dataset already exists. Loading...")

    else:
        print("Embeddings dataset does not exists. Creating and loading the dataset...")
        create_embeddings_dataset(imgs_dataset_dir, embeddings_dataset_path)

    # Load pickle object
    with open(embeddings_10_ppl_dataset_path, "rb") as file:
        embeddings_10_ppl_dataset = pickle.load(file)
    
    # Load pickle object
    with open(embeddings_dataset_path, "rb") as file:
        embeddings_dataset = pickle.load(file)

    print("Embeddings datasets loaded.")
    
    return embeddings_10_ppl_dataset, embeddings_dataset



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


def create_embeddings_dataset(bbdd_path, dataset_path, num_embeddings_group=750):
    """ Crea una base de datos de embeddings a partir de las imágenes en bbdd_path """

    embeddings_db = {}
    bbdd_path = Path(bbdd_path)  
 
    for group in os.listdir(bbdd_path):
        group_path = bbdd_path / group  
 
        if group_path.is_dir():
            embeddings_db[group] = []
 
            # Obtener todas las personas (subcarpetas) en el grupo
            persons = [person for person in os.listdir(group_path) if os.path.isdir(group_path / person)]
 
            # Seleccionar num_embeddings_group personas aleatorias si hay más de num_embeddings_group
            selected_persons = random.sample(persons, num_embeddings_group) if len(persons) > num_embeddings_group else persons
 
            # Recorrer las num_embeddings_group personas seleccionadas
            for person in selected_persons:                
                person_path = group_path / person
 
                # Obtener las imágenes de la persona
                img_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg'))]
 
                # Seleccionar la primera imagen de la persona
                img_file = img_files[0]  # Elegir la primera imagen
                img_path = person_path / img_file
 
                embedding = extract_features(img_path)
                embeddings_db[group].append(embedding)
    
    with open(dataset_path, "wb") as file:
        pickle.dump(embeddings_db, file)


# Crear base de datos de embeddings
def create_embeddings_10_ppl_dataset(bbdd_10_ppl_dir, embeddings_10_ppl_dataset_path):
    """ Crea una base de datos de embeddings a partir de las imágenes en bbdd_path """
    embeddings_db = {}
    bbdd_path = Path(bbdd_10_ppl_dir) # Convertir a Path para manejar rutas correctamente
    for person in os.listdir(bbdd_path):
        person_path = bbdd_path / person  # Construcción segura de rutas
        
        if person_path.is_dir():
            embeddings_db[person] = []
            
            for img_name in os.listdir(person_path):
                img_path = person_path / img_name  # Construcción segura de rutas

                embedding = extract_features(img_path)
                embeddings_db[person].append(embedding)
    
    with open(embeddings_10_ppl_dataset_path, "wb") as file:
        pickle.dump(embeddings_db, file)


#### TAREA 1.1


def plot_farr_frr(fars, frrs, thresholds):
    # Graficar FAR vs FRR
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fars, label="False Acceptance Rate (FAR)", color="red")
    plt.plot(thresholds, frrs, label="False Rejection Rate (FRR)", color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR vs FRR")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def plot_histogram(same_person, different_person):
    # hacer el histogramama de las dos similitudes, que estan metidos en el array, sam_person y different_person
    plt.figure(figsize=(8, 6))
    plt.hist(same_person, alpha=0.5, color='b', label='same person', density=True)
    plt.hist(different_person, alpha=0.5, color='r', label='different person', density=True)
    plt.xlabel("Ocurrencias")
    plt.ylabel("Similitud")
    plt.title("Similitudes")
    plt.legend()


#### TAREA 1.2


def create_embeddings_subset(embeddings, num_embedding=50):
    """ Crea una base de datos de embeddings a partir de las imágenes en bbdd_path """
    embeddings_subset = {}
    
    for key in embeddings:
        embeddings_subset[key] = embeddings[key][:num_embedding]
        
    return embeddings_subset


#### TAREA 1.3
    
def plot_tsne(embeddings_2d, labels_array):
    # Plot the results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 1], embeddings_2d[:, 0], c=labels_array%3, cmap='viridis', alpha=0.7)
    plt.title("Visualización de embeddings con t-SNE (etnias).")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.show()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 1], embeddings_2d[:, 0], c=labels_array, cmap='viridis', alpha=0.7)
    plt.title("Visualización de embeddings con t-SNE (etnias y géneros).")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.show()



#### TAREA 2.1

def generate_test_train(embeddings_db):
    """
    Convierte un diccionario de embeddings en matrices X e y.
    Parámetros:
    - embeddings_db: dict, diccionario con claves que indican el grupo y valores con listas de embeddings.
 
    Retorna:
    - X: numpy array con los embeddings.
    - y: numpy array con las etiquetas de género (one-hot encoded).
    """
    X = []
    y = []
 
    for label, embeddings in embeddings_db.items():
        gender = 0 if label[0] == 'H' else 1  # Hombre (0), Mujer (1)
        for emb in embeddings:
            X.append(emb)
            y.append(gender)
 
    X = np.array(X).astype(float)
    y = to_categorical(y)  # Convertir a one-hot encoding

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
 
    return X_train, X_test, y_train, y_test


def divide_embeddings(embeddings_dataset):
    embeddings_divided = {'A': {"x_train": [],
                                "x_test": [],
                                "y_train": [],
                                "y_test": []},
                          'B': {"x_train": [],
                                "x_test": [],
                                "y_train": [],
                                "y_test": []},
                          'N': {"x_train": [],
                                "x_test": [],
                                "y_train": [],
                                "y_test": []}}
    
    for ethnicity in embeddings_divided.keys():
        # Generate test and train datasets for each ethnicity
        grupos_filtrados = [f'H{ethnicity}4K_120', f'M{ethnicity}4K_120']
        embeddings_dataset_ethnicity = {k: v for k, v in embeddings_dataset.items() if k in grupos_filtrados}

        x_train, x_test, y_train, y_test = generate_test_train(embeddings_dataset_ethnicity)

        embeddings_divided[ethnicity]["x_train"] = x_train
        embeddings_divided[ethnicity]["x_test"] = x_test
        embeddings_divided[ethnicity]["y_train"] = y_train
        embeddings_divided[ethnicity]["y_test"] = y_test
    
    return embeddings_divided


def generate_gender_model():
    # Define the model
    model = Sequential([
        Dense(2048, activation='relu', input_shape=(2048,)),  # First dense layer with ReLU
        Dense(2, activation='softmax')  # Output layer with softmax for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical_crossentropy if labels are one-hot encoded
                  metrics=['accuracy'])
    
    return model


def train_gender_model(x_train, y_train, x_test, y_test):
    model = generate_gender_model()
    
    # Train the model
    history = model.fit(
        x_train, y_train,  # Training data
        validation_data=(x_test, y_test),  # Validation data
        epochs=20,  # Number of epochs (adjust as needed)
        batch_size=32,  # Batch size (adjust for performance)
        verbose=0 # Show training progress
    )
    
    return model, history.history['accuracy'][-1], history.history['val_accuracy'][-1]


#### TAREA 2.2

def get_all_accuracies_table(gender_models, embeddings):
    accuracies = {}
    
    for key in gender_models:
        accuracies[key] = {}
        for key2 in embeddings:
            accuracies[key][key2] = None

    # Evaluate the model and retrieve accuracies
    for ethnicity, model in gender_models.items():
        for other_ethnicity in embeddings.keys():
            x_test = np.concatenate((embeddings[other_ethnicity]["x_train"], embeddings[other_ethnicity]["x_test"]))
            y_test = np.concatenate((embeddings[other_ethnicity]["y_train"], embeddings[other_ethnicity]["y_test"]))
        # Duda: se esta evauluando el modelo con datos de test y train a la vez ¿??
            test_loss, test_acc = model.evaluate(x_test,y_test, verbose=0)

            accuracies[ethnicity][other_ethnicity] = test_acc * 100

    # Format the accuracies in a table
    df = pd.DataFrame(accuracies)

    df.index = [f"Dataset {k}" for k in df.index]
    df.columns = [f"Model {k}" for k in df.columns]

    df_styled = df.style.background_gradient(cmap='Blues', low=0.1, high=0.2).set_caption("Tabla de Accuracies").format("{:.4f}").set_table_styles([{
                        'selector': 'table', 
                        'props': [('font-size', '20px'), ('width', '100%')]
                    }, {
                        'selector': 'th', 
                        'props': [('font-size', '20px'), ('text-align', 'center')]
                    }, {
                        'selector': 'td', 
                        'props': [('font-size', '16px'), ('text-align', 'center')]
                    }])
    return df_styled