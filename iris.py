#!/usr/bin/env python

# coding: utf-8

import streamlit as st
import joblib
import numpy as np

# Créer une application Streamlit
def main():
    
    # Afficher un titre
    st.title('Prédiction iris')

    # Chargement du modèle
    model = joblib.load('iris.pkl')
    
    #Définition d'une fonction pour la prédiction des varietés de fleur
    def predict_species(species_data):
        # Prédiction de la classe en utilisant le model
        prediction = model.predict(species_data.reshape(1, -1))[0]
    
        # Associer la valeur prédite à la classe correspondante
        if prediction == 0:
            return 'Virginica'
        elif prediction == 1:
            return 'Versicolor'
        else:
            return 'Setosa'
        
    

    # Obtenir les données de la fleur
    sepal_length=st.number_input('Longeur Sépale',min_value=0.0,max_value=10.0,value=3.2)
    sepal_width=st.number_input('Largeur Sépale',min_value=0.0,max_value=10.0,value=2.1)
    petal_length=st.number_input('Longeur Pétale',min_value=0.0,max_value=10.0,value=5.1)
    petal_width=st.number_input('Largeur Pétale',min_value=0.0,max_value=10.0,value=3.1)
    
    # Calculer les données qu'on avait créer avec le feature engineering: l'aire des sépales et des pétales
    
    sepal_area=sepal_width*sepal_length
    petal_area=petal_width*petal_length


    # Collecte des caractéristiques dans un tableau numpy
    species_data = np.array([sepal_length,sepal_width,petal_length,petal_width,sepal_area,petal_area])

    # Appel de la fonction de prédiction avec comme paramètre le tableau des caractéristiques collectées
    returned_prediction = predict_species(species_data)

    # Afficher la prédiction
    if st.button("Prédire"):
        st.success(f'iris {returned_prediction}')
        

#Appel de la fonction main()
if __name__ == '__main__':
    main()
