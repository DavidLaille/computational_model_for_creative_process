import csv

import numpy as np
from matplotlib import pyplot as plt

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

# Chargement des mots-indices depuis le fichier csv
df_cues = pd.read_csv('data/experimental_data/cues.csv', sep=',')
print("Fichier cues.csv chargé avec succès.")
cues = df_cues['cues'].tolist()

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
discounting_rate = 0.05  # (1%)

memory_size = 3
vocab_size = 3000

nb_neighbours = 5
nb_max_steps = 100
method = 3

alpha = 0.5
gamma = 0.5

nb_try = 67

########################################################################################################################
# Création d'un dataframe pour récupérer les données générées par le modèle
cues_and_responses_cond1 = dict()
cues_and_responses_cond2 = dict()

for cue in cues:
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # # si on veut tester un seul mot-indice
    # word_to_test = "avis"
    # if cue != word_to_test:
    #     continue

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
    paths_filename = f'data/generated_data/dataframes/paths_{cue}.csv'
    paths.to_csv(paths_filename, index=False, sep=',')

    # print(all_neighbours_data)
    all_neighbours_data_filename = f'data/generated_data/dataframes/all_neighbours_data_{cue}.csv'
    all_neighbours_data.to_csv(all_neighbours_data_filename, index=False, sep=',')

    cues_and_responses_cond1[cue] = paths['step_1'].tolist()
    cues_and_responses_cond2[cue] = paths['best_word'].tolist()

########################################################################################################################
# Sauvegarde des données dans un csv nommé responses_by_cue_2_lines.csv
header = ['cues']
for num_participant in range(nb_try):
    header.append("p_" + str(num_participant + 1))
with open('data/generated_data/responses_by_cue_2_lines.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    row_first = []
    row_distant = []
    for cue in cues_and_responses_cond1.keys():
        row_first = cues_and_responses_cond1[cue]
        row_first.insert(0, cue)
        row_distant = cues_and_responses_cond2[cue]
        row_distant.insert(0, cue)

        writer.writerow(row_first)
        writer.writerow(row_distant)

########################################################################################################################
# Création et affichage du nombre d'occurrences pour chacune des réponses données par les participants
# en condition 1 (First) et en condition 2 (Distant)

# Initialisation des fichiers csv et création des headers
header = ['cues']
nb_max_response = nb_try
for i in range(nb_max_response):
    header.append("r_" + str(i+1))

f_nb_occurrences_first = open(f'data/generated_data/nb_occurrences_by_response_first.csv', 'w')
writer_first = csv.writer(f_nb_occurrences_first)
writer_first.writerow(header)

f_nb_occurrences_distant = open(f'data/generated_data/nb_occurrences_by_response_distant.csv', 'w')
writer_distant = csv.writer(f_nb_occurrences_distant)
writer_distant.writerow(header)

f_nb_occurrences_first.close()
f_nb_occurrences_distant.close()

for cue in cues:
    responses_cond1 = cues_and_responses_cond1[cue]
    responses_cond2 = cues_and_responses_cond2[cue]

    # Calcul du nombre d'occurrences des 1ers mots donnés par les participants (First)
    first_words = []
    first_words_nb_occurrences = []
    for response in responses_cond1:
        if response != response:  # if it's a 'nan'
            pass
            # print(f"Réponse : {response} - on ne l'ajoute pas à la liste des réponses")
        elif response not in first_words:
            first_words.append(response)
            first_words_nb_occurrences.append(responses_cond1.count(response))

    # print("###########################################################################################################")
    # print(f"Mots First : {first_words}")
    # print(f"Nb_occurrences mots First : {first_words_nb_occurrences}")
    # print("###########################################################################################################")

    df_first_words = pd.DataFrame({
        'mots': first_words,
        'nb_occurrences': first_words_nb_occurrences
    })
    df_first_words = df_first_words.sort_values(by=['nb_occurrences'], ascending=False, ignore_index=True)

    # Calcul du nombre d'occurrences des mots créatifs donnés par les participants (Distant)
    distant_words = []
    distant_words_nb_occurrences = []
    for response in responses_cond2:
        if response != response:  # if it's a 'nan'
            pass
            # print(f"Réponse : {response} - on ne l'ajoute pas à la liste des réponses")
        elif response not in distant_words:
            distant_words.append(response)
            distant_words_nb_occurrences.append(responses_cond2.count(response))

    # print("###########################################################################################################")
    # print(f"Mots Distant : {distant_words}")
    # print(f"Nb_occurrences mots Distant : {distant_words_nb_occurrences}")
    # print("###########################################################################################################")

    df_distant_words = pd.DataFrame({
        'mots': distant_words,
        'nb_occurrences': distant_words_nb_occurrences
    })
    df_distant_words = df_distant_words.sort_values(by=['nb_occurrences'], ascending=False, ignore_index=True)

    # Affichage des 1ers mots (First) et des mots créatifs (Distant) donnés par les participants
    # avec leur nombre d'occurrences
    height = 16
    width = 10
    fig_f_and_ch = plt.figure(figsize=(width, height))
    (ax1, ax2) = fig_f_and_ch.subplots(1, 2)
    fig_f_and_ch.suptitle(f"{cue} - Nb d'occurrences des 1ers mots et des mots créatifs générés par le modèle",
                          color='brown', fontsize=14)
    fig_f_and_ch.tight_layout(h_pad=4, w_pad=7)
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)

    x1_min = min(first_words_nb_occurrences)
    x1_max = max(first_words_nb_occurrences)
    x1_step = 1
    x2_min = min(distant_words_nb_occurrences)
    x2_max = max(distant_words_nb_occurrences)
    x2_step = 1
    grid1_x_ticks = np.arange(x1_min, x1_max, 1)
    grid2_x_ticks = np.arange(x2_min, x2_max, 1)

    ax1.set_xticks(grid1_x_ticks, minor=True)
    ax1.grid(axis='x', which='major', color='grey', linestyle='-', linewidth=0.75, alpha=0.5)
    ax1.grid(axis='x', which='minor', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.barh(y=df_first_words.mots[::-1], width=df_first_words.nb_occurrences[::-1])
    ax1.set_title('First words & Nb_occurrences')
    ax1.set(xlabel='nb_occurrences')

    ax2.set_xticks(grid2_x_ticks, minor=True)
    ax2.grid(axis='x', which='major', color='grey', linestyle='-', linewidth=0.75, alpha=0.5)
    ax2.grid(axis='x', which='minor', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.barh(y=df_distant_words.mots[::-1], width=df_distant_words.nb_occurrences[::-1])
    ax2.set_title('Distant words & Nb_occurrences')
    ax2.set(xlabel='nb_occurrences')

    ####################################################################################################################
    # Sauvegarde des figures obtenues
    file_name = f"data/generated_data/images/{cue}_first_and_distant_words.png"
    print(file_name)
    plt.savefig(file_name)
    plt.close(fig_f_and_ch)
    # plt.show()
    ####################################################################################################################

    ####################################################################################################################
    # Sauvegarde des données dans 2 fichiers csv
    # nb_occurrences_by_response_first.csv   : réponses données en condition 1 (First) + nb_occurrences
    # nb_occurrences_by_response_distant.csv : réponses données en condition 2 (Distant) + nb_occurrences

    ####################################################################################################################
    # Sauvegarde n°1 : fichier nb_occurrences_by_response_first.csv
    # Création des lignes à ajouter dans le fichier nb_occurrences_by_response_first.csv
    row_first_words = [cue]
    row_nb_occurrences_first_words = [cue + "_nb_occurrences"]
    for word in df_first_words['mots']:
        row_first_words.append(word)
    for word in df_first_words['nb_occurrences']:
        row_nb_occurrences_first_words.append(word)

    # print("###########################################################################################################")
    # print("First words : ", row_first_words)
    # print("Nombre d'occurrences des First words : ", row_nb_occurrences_first_words)
    # print("###########################################################################################################")

    # Ouverture et écriture dans le fichier nb_occurrences_by_response_first.csv
    f_nb_occurrences_first = open(f'data/generated_data/nb_occurrences_by_response_first.csv', 'a')
    writer_first = csv.writer(f_nb_occurrences_first)

    writer_first.writerow(row_first_words)
    writer_first.writerow(row_nb_occurrences_first_words)

    ####################################################################################################################
    # Sauvegarde n°2 : fichier nb_occurrences_by_response_distant.csv
    # Création des lignes à ajouter dans le fichier nb_occurrences_by_response_distant.csv
    row_distant_words = [cue]
    row_nb_occurrences_distant_words = [cue + "_nb_occurrences"]
    for word in df_distant_words['mots']:
        row_distant_words.append(word)
    for word in df_distant_words['nb_occurrences']:
        row_nb_occurrences_distant_words.append(word)

    # print("###########################################################################################################")
    # print("Distant words : ", row_distant_words)
    # print("Nombre d'occurrences des Distant words : ", row_nb_occurrences_distant_words)
    # print("###########################################################################################################")

    # Ouverture et écriture dans le fichier nb_occurrences_by_response_distant.csv
    f_nb_occurrences_distant = open(f'data/generated_data/nb_occurrences_by_response_distant.csv', 'a')
    writer_distant = csv.writer(f_nb_occurrences_distant)

    writer_distant.writerow(row_distant_words)
    writer_distant.writerow(row_nb_occurrences_distant_words)

    ####################################################################################################################
    # Fermeture des 2 fichiers csv
    f_nb_occurrences_first.close()
    f_nb_occurrences_distant.close()
    ####################################################################################################################
