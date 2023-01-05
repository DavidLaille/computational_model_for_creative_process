import functions_v1 as fct
import pandas as pd
import math


"""
Infos du modèle word2vec pré-entraîné
    nom du modèle   : frWac_no_postag_no_phrase_700_skip_cut50.bin
    nombre de mots  : 184 373 mots
    no_postag       : la catégorie grammaticale n'est pas indiquée (pas de tag '_a', '_n' ou '_v')
    no_phrase       : le modèle ne contient que des mots (pas de phrase ou d'expressions)
    700             : les vecteurs sont de taille 700
    skip            : la méthode utilisée est la méthode skip-gram
    cut50           : seuls les mots qui apparaissaient 50 fois ou plus dans le corpus ont été conservés
"""
# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/"
location_word2vec_models = location_word2vec_models_mac

pathToModel = location_word2vec_models + "frWac_no_postag_no_phrase_700_skip_cut50_modified.bin"

word2vec_model = fct.get_model(pathToModel)
print("Modèle word2vec chargé avec succès.")

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('data/experimental_data/all_data.csv', sep=',')
print("Fichier all_data.csv chargé avec succès.")

df['similarity'] = 'nan'
# df.insert(loc=len(df.axes[1]), column='similarity', value='nan')
print(df.head())

print("#############################################")
for index in range(len(df)):
    # pour ne pas tester toutes les valeurs
    # if index > 5:
    #     break

    cue = df.cues[index]
    response = df.responses[index]
    print(f"{index} Mot-indice : {cue}  -  Réponse : {response}")

    if response != response:  # on vérifie que response n'est pas 'nan'
        continue

    if word2vec_model.has_index_for(response):  # on vérifie que le mot réponse est dans le modèle word2vec
        similarity = word2vec_model.similarity(cue, response)
        df['similarity'][index] = similarity

# print(df.head())
# enregistrement du dataframe modifié
df.to_csv('data/experimental_data/all_data_with_sim_from_word2vec.csv', sep=',')
print("Fichier sauvegardé avec succès.")
