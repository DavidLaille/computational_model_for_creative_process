import os
import csv

import numpy as np
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt

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

########################################################################################################################
# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/"
location_word2vec_models = location_word2vec_models_mac

# Liste des modèles word2vec (sans postag) disponibles
# Modèles lemmatisés issus des sites web français (en .fr)
word2vec_model_name1 = "frWac_no_postag_no_phrase_700_skip_cut50_modified_2.bin"
word2vec_model_name2 = "frWac_no_postag_no_phrase_500_cbow_cut100_modified_2.bin"

# Modèles non lemmatisés issus des sites web français (en .fr)
word2vec_model_name3 = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100_modified_2.bin"
word2vec_model_name4 = "frWac_non_lem_no_postag_no_phrase_500_skip_cut100_modified_2.bin"

# Modèles lemmatisés issus du Wikipédia français
word2vec_model_name5 = "frWiki_no_postag_no_phrase_500_cbow_cut10_modified_2.bin"
word2vec_model_name6 = "frWiki_no_postag_no_phrase_700_cbow_cut100_modified_2.bin"
word2vec_model_name7 = "frWiki_no_postag_no_phrase_1000_skip_cut100_modified_2.bin"

# Modèles non lemmatisés issus du Wikipédia français
word2vec_model_name8 = "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100_modified_2.bin"
word2vec_model_name9 = "frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100_modified_2.bin"

# Modèles élaborés par l'équipe DaSciM (Polytechnique Paris - X)
word2vec_model_name10 = "originals/fr_w2v_web_w5_modified_2.bin"
word2vec_model_name11 = "originals/fr_w2v_fl_w5_modified_2.bin"
word2vec_model_name12 = "originals/fr_w2v_web_w20_modified_2.bin"
# word2vec_model_name13 = "originals/fr_w2v_fl_w20.bin"

pathToWord2vecModel = location_word2vec_models + word2vec_model_name2
word2vec_model = KeyedVectors.load_word2vec_format(pathToWord2vecModel, binary=True, unicode_errors="ignore")
print("Modèle word2vec chargé avec succès.")

########################################################################################################################
# Type de modèle computationnel utilisé
model_type = 2

nb_participants = 10

paths_all_participants = pd.DataFrame()
all_neighbours_data_all_participants = pd.DataFrame()

########################################################################################################################
# Initialisation des paramètres du modèle computationnel
adequacy_influences = np.random.normal(0.65, 0.10, nb_participants)
print("Influences de l'adequacy dans le calcul de la likeability : ", adequacy_influences)

initial_goal_values = np.random.normal(1, 0.03, nb_participants)
print("Valeurs de but (goal_values) initiales : ", initial_goal_values)
discounting_rates = np.random.normal(0.07, 0.03, nb_participants)
print("Taux de décroissance de la valeur de but : ", discounting_rates)

memory_sizes = np.random.choice(range(5, 10), nb_participants)
print("Tailles de la mémoire de travail : ", memory_sizes)
vocab_sizes = np.random.choice(range(500, 30000), nb_participants)
print("Tailles du lexique des participants : ", vocab_sizes)

nbs_neighbours = np.random.choice(range(2, 10), nb_participants)
print("Nombre de voisin créé pour chaque participant : ", nbs_neighbours)
methods = np.random.choice((3, 4), nb_participants)
print("Méthode utilisée pour chaque participant : ", methods)

alpha = 0.5
gamma = 0.5

nb_try = 1

########################################################################################################################
# Création d'un dataframe pour récupérer les données générées par le modèle
cues_and_responses_cond1 = dict()
cues_and_responses_cond2 = dict()

for cue in cues:
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    paths_all_participants = pd.DataFrame()
    all_neighbours_data_all_participants = pd.DataFrame()

    for num_participant in range(nb_participants):
        ####################################################################################################################
        # Initialisation des paramètres du modèle computationnel
        adequacy_influence = adequacy_influences[num_participant]

        initial_goal_value = initial_goal_values[num_participant]
        discounting_rate = discounting_rates[num_participant]
        if discounting_rate < 0.01:
            discounting_rate = 0.01

        memory_size = memory_sizes[num_participant]
        vocab_size = vocab_sizes[num_participant]

        nb_neighbours = nbs_neighbours[num_participant]
        method = methods[num_participant]
        ####################################################################################################################

        # Initialisation du modèle computationnel
        model = ComputationalModel(word2vec_model=word2vec_model, model_type=model_type,
                                   adequacy_influence=adequacy_influence,
                                   initial_goal_value=initial_goal_value, discounting_rate=discounting_rate,
                                   memory_size=memory_size, vocab_size=vocab_size,
                                   nb_neighbours=nb_neighbours, method=method,
                                   alpha=alpha, gamma=gamma)

        paths, all_neighbours_data = model.launch_model(cue=cue, nb_try=nb_try)

        '''
        Sauvegarde des données dans des fichiers csv
           all_neighbours_data_{cue}.csv   : fichier csv contenant toutes les données des réseaux sémantiques créés
                                             et parcourus pour un mot-indice donné ("cue")
           paths_{cue}.csv                 : fichier csv contenant tous les chemins parcourus pour un mot-indice donné ("cue")
        '''

        os.makedirs(f'data/generated_data/dataframes/sujet_{num_participant}', exist_ok=True)

        # print(paths)
        paths_filename = f'data/generated_data/dataframes/sujet_{num_participant}/paths_{cue}.csv'
        paths.to_csv(paths_filename, index=False, sep=',')

        # print(all_neighbours_data)
        all_neighbours_data_filename = f'data/generated_data/dataframes/sujet_{num_participant}/all_neighbours_data_{cue}.csv'
        all_neighbours_data.to_csv(all_neighbours_data_filename, index=False, sep=',')

        # On ajoute le numéro du participant dans les dataframes
        paths.insert(0, 'num_participant', num_participant)
        all_neighbours_data.insert(0, 'num_participant', num_participant)
        # print(f"Paths for cue {cue}", paths)
        # print(f"Neighbours data for cue {cue}", all_neighbours_data)

        paths_all_participants = pd.concat([paths_all_participants, paths], ignore_index=True)
        all_neighbours_data_all_participants = pd.concat([all_neighbours_data_all_participants, all_neighbours_data], ignore_index=True)

    cues_and_responses_cond1[cue] = paths_all_participants['step_1'].tolist()
    cues_and_responses_cond2[cue] = paths_all_participants['best_word'].tolist()

print("All paths for all participants : ", paths_all_participants)
paths_all_participants_filename = f'data/generated_data/dataframes/tests/paths_{nb_participants}.csv'
paths_all_participants.to_csv(paths_all_participants_filename, index=False, sep=',')
print("All neighbours data for all participants : ", all_neighbours_data_all_participants)
all_neighbours_data_all_participants_filename = f'data/generated_data/dataframes/tests/all_neighbours_data_{nb_participants}.csv'
all_neighbours_data_all_participants.to_csv(all_neighbours_data_all_participants_filename, index=False, sep=',')

########################################################################################################################
# Sauvegarde des données dans un csv nommé responses_by_cue_2_lines.csv
header = ['cues']
for num_participant in range(nb_participants):
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
nb_max_response = nb_participants
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
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    responses_cond1 = cues_and_responses_cond1[cue]
    responses_cond2 = cues_and_responses_cond2[cue]

    # Calcul du nombre d'occurrences des 1ers mots donnés par les participants (First)
    first_words = []
    first_words_nb_occurrences = []
    for response in responses_cond1:
        if response == cue:
            pass
        elif response != response:  # if it's a 'nan'
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
        if response == cue:
            pass
        elif response != response:  # if it's a 'nan'
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
