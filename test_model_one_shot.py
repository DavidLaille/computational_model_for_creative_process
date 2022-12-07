import functions_v1 as fct
import pandas as pd
from computational_model import ComputationalModel

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

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('data/experimental_data/cues.csv', sep=',')
print("Fichier cues.csv chargé avec succès.")

# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/"
location_word2vec_models = location_word2vec_models_mac

pathToModel = location_word2vec_models + "frWac_no_postag_no_phrase_700_skip_cut50_modified.bin"
word2vec_model = fct.get_model(pathToModel)
print("Modèle word2vec chargé avec succès.")

model_type = 2

########################################################################################################################
# Initialisation des paramètres du modèle computationnel
adequacy_influence = 0.65

initial_goal_value = 1
discounting_rate = 0.03  # (1%)

memory_size = 3
vocab_size = 3000

nb_neighbours = 5
nb_max_steps = 100
method = 3

alpha = 0.5
gamma = 0.5

nb_try = 5

########################################################################################################################
for cue in df['cues']:
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    # # Initialisation de la goal_value
    # # La goal_value initiale aura la valeur d'agréabilité la plus haute + un petit qqch (3%, 5% ou 10% par ex.)
    # initial_goal_value = fct.get_max_likeability(word2vec_model, cue,
    #                                              adequacy_influence,
    #                                              vocab_size) + 0.03
    # print("Initial goal value : ", initial_goal_value)

    # Initialisation du modèle computationnel
    model = ComputationalModel(word2vec_model=word2vec_model, model_type=model_type,
                               adequacy_influence=adequacy_influence,
                               initial_goal_value=initial_goal_value, discounting_rate=discounting_rate,
                               memory_size=memory_size, vocab_size=vocab_size,
                               nb_neighbours=nb_neighbours, nb_max_steps=nb_max_steps, method=method,
                               alpha=alpha, gamma=gamma)

    paths, all_neighbours_data = model.launch_model(cue=cue, nb_try=nb_try)

    '''
    Sauvegarde des données dans des fichiers csv
       all_neighbours_data_{cue}.csv   : fichier csv contenant toutes les données des réseaux sémantiques créés
                                         et parcourus pour un mot-indice donné ("cue")
       paths_{cue}.csv                 : fichier csv contenant tous les chemins parcourus pour un mot-indice donné ("cue")
    '''
    # print(paths)
    paths_filename = f'data/dataframes/paths_{cue}.csv'
    paths.to_csv(paths_filename, index=False, sep=',')

    # print(all_neighbours_data)
    all_neighbours_data_filename = f'data/dataframes/all_neighbours_data_{cue}.csv'
    all_neighbours_data.to_csv(all_neighbours_data_filename, index=False, sep=',')
