# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:25:58 2023

@author: rober, Sofía, Hajar, Andrés
"""


import os
import cv2
import math
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
from sklearn.metrics import roc_curve, auc
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

def retrieve_scores_and_true_labels(embeddings_db):
    """ Plot the roc curve given the embeddings. """
    
    y_true = []
    scores = []
 
    people = list(embeddings_db.keys())

    # Calculate the similitudes and the true labels
    for person in people:
        embeddings = embeddings_db[person]

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                score = np.dot(embeddings[i], embeddings[j].T)  # Comparar imágenes de la MISMA persona
                scores.append(score)
                y_true.append(1)
            
            for other_person in people:
                if other_person != person:
                    for emb2 in embeddings_db[other_person]:
                        score = np.dot(embeddings[i], emb2.T)  # Comparar imágenes de DIFERENTES personas
                        scores.append(score)
                        y_true.append(0)

    return np.array(scores), np.array(y_true)


def plot_far_frr_curves(scores, y_true):
    """ Calcula FAR y FRR para diferentes umbrales y genera un gráfico. """
    
    if len(scores) != len(y_true):
        raise Exception(f"Scores and labels must have the same lenght. {len(scores)} vs {len(y_true)}")
    
    fars = []
    frrs = []
    
    thresholds=np.linspace(0, 1, 50)

    for threshold in thresholds:
        false_accepts = 0
        false_rejects = 0
        total_genuine = 0
        total_impostor = 0

        for i in range(len(scores)):
            score = scores[i]
            ground_truth = y_true[i]
            
            if ground_truth == 1:
                total_genuine += 1
                if score < threshold:
                    false_rejects += 1  # Error: No reconoce a la persona correcta
            else:
                total_impostor += 1
                if score >= threshold:
                    false_accepts += 1  # Error: Acepta a la persona equivocada

        # Calcular tasas FAR y FRR
        far = false_accepts / total_impostor if total_impostor > 0 else 0
        frr = false_rejects / total_genuine if total_genuine > 0 else 0

        fars.append(far)
        frrs.append(frr)

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


def plot_similitude_histogram(scores, y_true):
    # Tamaños de los bins proporcional al número de muestras
    bins_genuine = math.ceil(2 * (len(scores[y_true==1]) ** (1/3)))
    bins_impostor = math.ceil(2 * (len(scores[y_true==0]) ** (1/3)))
    
    plt.figure(figsize=(8, 6))
    plt.hist(scores[y_true==1], bins=bins_genuine ,alpha=0.5, color='b', label='same person', density=True)
    plt.hist(scores[y_true==0], bins=bins_impostor ,alpha=0.5, color='r', label='different person', density=True)
    plt.xlabel("Ocurrencias")
    plt.ylabel("Similitud")
    plt.title("Similitudes")
    plt.legend()

'''
def plot_roc_curve(scores, y_true):
    """ Plot the ROC curve. """
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
'''
def plot_roc_curve(scores, y_true, ax=None):
    """Plot the ROC curve, with optional Axes support for subplots."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True)


#### TAREA 1.2


def create_embeddings_subset(embeddings, num_embedding=50):
    """ Crea una base de datos de embeddings a partir de las imágenes en bbdd_path """
    embeddings_subset = {}
    
    for key in embeddings:
        embeddings_subset[key] = embeddings[key][:num_embedding]
        
    return embeddings_subset


#### TAREA 1.3

def plot_tsne_with_gender(embeddings_2d, labels_array):
    # Reverse mapping from numeric labels to group names
    group_labels = {0: 'HA', 1: 'MA', 2: 'HN', 3: 'MN', 4: 'HB', 5: 'MB'}  
    subgroup_labels = {'H': 0, 'M': 1}  
    
    plt.figure(figsize=(8, 6))
    
    # Loop through each unique label
    for group in subgroup_labels.keys():
        # Create a mask for the current group
        indices = [group == group_labels[index][0] for index in labels_array]
        plt.scatter(
            embeddings_2d[indices, 1],
            embeddings_2d[indices, 0],
            alpha=0.7,
            label=group
        )
    
    plt.title("t-SNE Visualization of gender groups")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Group")
    plt.show()


def plot_tsne_with_ethnic(embeddings_2d, labels_array):
    # Reverse mapping from numeric labels to group names
    group_labels = {0: 'HA', 1: 'MA', 2: 'HN', 3: 'MN', 4: 'HB', 5: 'MB'}  
    subgroup_labels = {'A': 0, 'N': 1, 'B': 2}  
    
    plt.figure(figsize=(8, 6))
    
    # Loop through each unique label
    for group in subgroup_labels.keys():
        # Create a mask for the current group
        indices = [group == group_labels[index][1] for index in labels_array]
        plt.scatter(
            embeddings_2d[indices, 1],
            embeddings_2d[indices, 0],
            alpha=0.7,
            label=group
        )
    
    plt.title("t-SNE Visualization of ethinic groups")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Group")
    plt.show()

    
def plot_tsne_complete(embeddings_2d, labels_array):
    # Reverse mapping from numeric labels to group names
    group_labels = {'HA': 0, 'MA': 1, 'HN': 2, 'MN': 3, 'HB': 4, 'MB': 5}  

    plt.figure(figsize=(8, 6))
    
    # Loop through each unique label
    for group in group_labels.keys():
        # Create a mask for the current group
        indices = labels_array == group_labels[group]
        plt.scatter(
            embeddings_2d[indices, 1],
            embeddings_2d[indices, 0],
            alpha=0.7,
            label=group
        )
    
    plt.title("t-SNE Visualization of ethnic and gender")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Group")
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
            test_loss, test_acc = model.evaluate(embeddings[other_ethnicity]["x_test"], embeddings[other_ethnicity]["y_test"], verbose=0)
            accuracies[ethnicity][other_ethnicity] = test_acc * 100

    # Format the accuracies in a table
    df = pd.DataFrame(accuracies)

    df.index = [f"Dataset {k}" for k in df.index]
    df.columns = [f"Model {k}" for k in df.columns]

    df_styled = df.style.background_gradient(cmap='Blues', low=0.1, high=0.2, axis=None).set_caption("Tabla de Accuracies").format("{:.4f}").set_table_styles([{
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


def get_all_roc_curves(gender_models, embeddings):
    for ethnicity, model in gender_models.items():
        print(f"Model {ethnicity}")
        
        predictions = []
        y_true_list = []
        
        for other_ethnicity in embeddings.keys():
            preds = model.predict(embeddings[other_ethnicity]["x_test"], verbose=0)
            predictions.append(preds)
            y_true_list.append(embeddings[other_ethnicity]["y_test"])
        
        # Concatenate the predictions and true labels along the first axis
        all_predictions = np.concatenate(predictions, axis=0)
        all_y_true = np.concatenate(y_true_list, axis=0)

        plot_roc_curve(all_predictions[:, 1], all_y_true[:, 1])
        
def predict_gender(gender_models, model_key, img_path, image=True):
    """
    Realiza la predicción de género a partir de una imagen usando el modelo especificado.
    Parámetros:
    - model_key (str): Clave del modelo a usar ('A', 'B' o 'N').
    - img_path (str): Ruta de la imagen a evaluar.
    Imprime la imagen, la predicción y la clase predicha.
    """
    if model_key not in gender_models:
        print("Error: Modelo no válido. Usa 'A', 'B' o 'N'.")
        return
    
    model = gender_models[model_key]

    # Cargar y mostrar la imagen original
    img = cv2.imread(img_path)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return
    # si iamge=True impimir la iamgen
    if image:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(2, 2)) 
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Imagen de entrada ({model_key})")
        plt.show()
    print(f"\nUsando el modelo '{model_key}':")
    # Extraer la cara de la imagen
    img_face = extract_faces(img_path)

    if img_face is None:
        print("No se detectó ninguna cara en la imagen.")
        return

    # Generar embedding de la cara detectada
    embedding = generate_embedding(img_face)

    if embedding is not None:
        embedding = np.asarray(embedding).reshape(1, -1)  # Asegurar la forma correcta (1, 2048)

        # Usar el modelo correspondiente para la predicción
        prediction = model.predict(embedding, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)

        print(f"\tPredicción mujer: {prediction[0][1]:.05f}")
        print(f"\tPredicción hombre: {prediction[0][0]:.05f}")
        print("\tClase predicha:", "Hombre" if predicted_class[0] == 0 else "Mujer")
    else:
        print("No se pudo generar el embedding de la imagen.")
        
        
#### TAREA 2.5

def generate_test_train2(embeddings_db):
    """
    Convierte un diccionario de embeddings en matrices X e y.
    Parámetros:
    - embeddings_db: dict, diccionario con claves que indican el grupo y valores con listas de embeddings.
 
    Retorna:
    - X: numpy array con los embeddings.
    - y: numpy array con las etiquetas (one-hot encoded) para género y etnia.
    """
    X = []
    y = []
    
    # Definir etiquetas para las 6 clases: 3 etnias x 2 géneros = 6 clases
    for label, embeddings in embeddings_db.items():
        gender = 0 if label[0] == 'H' else 1  # Hombre (0), Mujer (1)
        ethnicity = label[1]  # A = Blanco, B = Asiático, N = Negro
        
        # Asignamos el índice de clase según la etnia y el género
        if ethnicity == 'A':  # Blanco
            ethnicity_label = 0
        elif ethnicity == 'B':  # Asiático
            ethnicity_label = 1
        else:  # Negro
            ethnicity_label = 2
        
        # La etiqueta será una combinación de género y etnia
        label_combined = ethnicity_label * 2 + gender  # 6 clases posibles
        
        for emb in embeddings:
            X.append(emb)
            y.append(label_combined)
 
    X = np.array(X).astype(float)
    y = to_categorical(y, num_classes=6)  # Convertir a one-hot encoding para 6 clases

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
 
    return X_train, X_test, y_train, y_test


def generate_combined_model():
    # Definir el modelo
    model = Sequential([ 
        Dense(2048, activation='relu', input_shape=(2048,)), 
        Dense(6, activation='softmax')  # Capa de salida con 6 clases (combinación de género y etnia)
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])
    
    return model


def train_combined_model(x_train, y_train, x_test, y_test):
    model = generate_combined_model()

    history = model.fit(
        x_train, y_train,  
        validation_data=(x_test, y_test),  
        epochs=20,  
        batch_size=32,  
        verbose=0  
    )
    
    return model, history.history['accuracy'][-1], history.history['val_accuracy'][-1]


def predict_gender_ethnicity(gender_models, combined_model, img_path, image=True):
    """
    Realiza la predicción de género y etnia a partir de una imagen usando un único modelo combinado.
    Parámetros:
    - combined_model: el modelo entrenado que predice género y etnia.
    - img_path (str): Ruta de la imagen a evaluar.
    Imprime la imagen, la predicción y la clase predicha (género y etnia).
    """
    # Cargar y mostrar la imagen original
    img = cv2.imread(img_path)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return
    
    # Si image=True, imprimir la imagen
    if image:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(2, 2)) 
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Imagen de entrada")
        plt.show()

    print(f"\nUsando el modelo combinado:")
    # Extraer la cara de la imagen
    img_face = extract_faces(img_path)

    if img_face is None:
        print("No se detectó ninguna cara en la imagen.")
        return

    # Generar embedding de la cara detectada
    embedding = generate_embedding(img_face)

    if embedding is not None:
        embedding = np.asarray(embedding).reshape(1, -1)  # Asegurar la forma correcta (1, 2048)

        # Usar el modelo combinado para la predicción
        prediction = combined_model.predict(embedding, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)

        # Mostrar las probabilidades para cada clase
        print(f"\tPredicción para las clases (Género + Etnia):")
        print(f"\tPredicción (HA): {prediction[0][0]:.05f}")
        print(f"\tPredicción (MA): {prediction[0][1]:.05f}")
        print(f"\tPredicción (HB): {prediction[0][2]:.05f}")
        print(f"\tPredicción (MB): {prediction[0][3]:.05f}")
        print(f"\tPredicción (HN): {prediction[0][4]:.05f}")
        print(f"\tPredicción (MN): {prediction[0][5]:.05f}")
        
        # Mostrar la clase predicha (combinación de género y etnia)
        if predicted_class[0] == 0:
            print("\tClase predicha: Hombre Etnia A")
        elif predicted_class[0] == 1:
            print("\tClase predicha: Mujer Etnia A")
        elif predicted_class[0] == 2:
            print("\tClase predicha: Hombre Etnia B")
        elif predicted_class[0] == 3:
            print("\tClase predicha: Mujer Etnia B")
        elif predicted_class[0] == 4:
            print("\tClase predicha: Hombre Etnia N")
        elif predicted_class[0] == 5:
            print("\tClase predicha: Mujer Etnia N")
    else:
        print("No se pudo generar el embedding de la imagen.")


def get_all_roc_curves(gender_models, embeddings):
    """
    Genera un ROC para cada modelo contenido en 'gender_models'.
    Cada gráfico se ubica en un subplot distinto, distribuidos en 2 columnas.
    """
    # Número de modelos a graficar
    n_models = len(gender_models)

    # Ajustamos cuántas filas y columnas queremos (2 columnas en este ejemplo)
    ncols = 2
    # Calculamos las filas necesarias (redondeo hacia arriba)
    nrows = (n_models + ncols - 1) // ncols

    # Creamos la figura con subplots en un grid de (nrows x ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5 * nrows))

    # Si solo hay un subplot, 'axes' no será un array de arrays, sino un solo Axes
    # Convertimos a array para iterar con más facilidad
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.ravel()  # "aplanamos" la matriz de ejes a 1D

    # Iteramos sobre cada modelo y su índice
    for i, (ethnicity, model) in enumerate(gender_models.items()):
        #print(f"Procesando modelo: {ethnicity}")

        predictions = []
        y_true_list = []
        
        # Concatenamos predicciones y labels verdaderos de todos los "other_ethnicity"
        for other_ethnicity in embeddings.keys():
            preds = model.predict(embeddings[other_ethnicity]["x_test"], verbose=0)
            predictions.append(preds)
            y_true_list.append(embeddings[other_ethnicity]["y_test"])

        all_predictions = np.concatenate(predictions, axis=0)
        all_y_true = np.concatenate(y_true_list, axis=0)

        # Seleccionamos la columna 1 (asumiendo que la salida del modelo es [prob_clase0, prob_clase1])
        ax = axes[i]  # Seleccionamos el subplot correspondiente
        plot_roc_curve(all_predictions[:, 1], all_y_true[:, 1], ax=ax)

        # Ajustamos el título para este subplot
        ax.set_title(f"ROC - Modelo: {ethnicity}")

    # Si sobran ejes (por ejemplo, si #modelos es impar), podemos borrarlos
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    # Ajustamos el espaciado
    plt.tight_layout()
    plt.show()