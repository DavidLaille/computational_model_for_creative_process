import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Chargement des mots-indices depuis le fichier csv
all_data = pd.read_csv('data/experimental_data/all_data.csv', sep=',')
print("Fichier all_data.csv chargé avec succès.")

# Chargement des mots-indices depuis le fichier csv
cues = pd.read_csv('data/experimental_data/cues.csv', sep=',')
print("Fichier cues.csv chargé avec succès.")

########################################################################################################################
cues = cues['cues'].tolist()
all_responses = all_data['responses'].tolist()

print("###############################################################################################################")
print("Cues : ", cues)
print("Responses : ", all_responses)
print("Nb Cues : ", len(cues))
print("Nb Responses : ", len(all_responses))
print("###############################################################################################################")

########################################################################################################################

########################################################################################################################
# Création de sous-ensembles de données (suivant la condition de passation)
# Condition 1 : First
# Condition 2 : Distant
all_data_cond1 = all_data[all_data['condition'] == 1]
all_data_cond2 = all_data[all_data['condition'] == 2]

print("###############################################################################################################")
print("Nombre de réponses en condition 1 (First) : ", len(all_data_cond1))
print("Nombre de réponses en condition 2 (Distant) : ", len(all_data_cond2))
print("Nombre de réponses au total : ", len(all_data))
print("###############################################################################################################")

all_responses_cond1 = all_data_cond1['responses'].tolist()
all_responses_cond2 = all_data_cond2['responses'].tolist()
########################################################################################################################

########################################################################################################################
# On récupère la liste des identifiants de chaque participant
id_participants = list()
for id_participant in all_data['id_participant']:
    if id_participant not in id_participants:
        id_participants.append(int(id_participant))
print("Liste des id des participants : ", id_participants)
########################################################################################################################

########################################################################################################################
# Test préalable pour vérifier les données expérimentales
for id_participant in id_participants:
    all_data_participant = all_data[all_data['id_participant'] == id_participant]
    cues_participant = all_data_participant['cues'].tolist()
    unique_cues_participant = list()
    nb_occurrences_cues_participant = list()
    for cue in cues:
        if cue not in unique_cues_participant:
            unique_cues_participant.append(cue)
            nb_occurrences_cues_participant.append(cues_participant.count(cue))
            # print("Mot-indice : ", cue)
            # print("Nb_occurrences : ", cues_participant.count(cue))

    if 1 in nb_occurrences_cues_participant or 3 in nb_occurrences_cues_participant:
        print(f"Participant {id_participant} : Problème !")

########################################################################################################################

########################################################################################################################
# Création d'un table de données regroupant les réponses des participants pour chaque mot-indice
col_names = ['cues']
for num_participant in id_participants:
    col_name_first = "p_" + str(num_participant) + "f"
    col_names.append(col_name_first)
    col_name_distant = "p_" + str(num_participant) + "d"
    col_names.append(col_name_distant)
# print("Nom des colonnes du dataframe : ", col_names)
# print("Nombre de colonnes du dataframe : ", len(col_names))
responses_by_participant = pd.DataFrame(columns=col_names)


cue_index = 0
previous_num_participant = id_participants[0]
num_participant = 0
num_response = 0
for cue in cues:
    row = [cue]
    for i, response in enumerate(all_data['responses']):
        num_participant = int(all_data['id_participant'][i])
        if previous_num_participant != num_participant and num_response < 2:
            while num_response < 2:
                row.append('no_data')
                # print(f"Participant {previous_num_participant} : Mot(s)-indice(s) manquant(s)")
                num_response += 1
        if previous_num_participant != num_participant and num_response == 2:
            num_response = 0
            previous_num_participant = int(all_data['id_participant'][i])
        if all_data['cues'][i] == cue:
            if num_response < 2:
                row.append(response)
                num_response += 1

    # print("Ligne à ajouter au dataframe : ", row)
    # print("Taille de la ligne à ajouter au dataframe : ", len(row))
    responses_by_participant.loc[cue_index] = row
    cue_index += 1

    # print(responses_by_participant)
    responses_by_participant_filename = 'data/experimental_data/responses_by_cue.csv'
    responses_by_participant.to_csv(responses_by_participant_filename, index=False, sep=',')

########################################################################################################################

########################################################################################################################
# Création de 2 dictionnaires contenant les réponses pour la condition 1 (First) d'une part
# et les réponses pour la condition 2 (Distant) d'autre part
# Puis regroupement et sauvegarde des données dans un csv nommé cues_and_responses.csv
cues_and_responses_cond1 = dict()
cues_and_responses_cond2 = dict()

for cue in cues:
    responses_cond1 = []
    for index, value in enumerate(all_data_cond1['responses']):
        # print(f"Index : {index} - value : {value}")
        if all_data_cond1['cues'].iloc[index] == cue:
            responses_cond1.append(value)
    cues_and_responses_cond1[cue] = responses_cond1

    responses_cond2 = []
    for index, value in enumerate(all_data_cond2['responses']):
        # print(f"Index : {index} - value : {value}")
        if all_data_cond2['cues'].iloc[index] == cue:
            responses_cond2.append(value)
    cues_and_responses_cond2[cue] = responses_cond2

print("###############################################################################################################")
print("Mots-indices et Réponses pour la condition 1 (First)", cues_and_responses_cond1)
print("Mots-indices et Réponses pour la condition 2 (Distant)", cues_and_responses_cond2)
print("###############################################################################################################")

header = ['cues']
for num_participant in id_participants:
    header.append("p_" + str(num_participant))
with open('data/experimental_data/responses_by_cue_2_lines.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    row_first = []
    row_distant = []
    for cue in cues:
        row_first = cues_and_responses_cond1[cue]
        row_first.insert(0, cue)
        row_distant = cues_and_responses_cond2[cue]
        row_distant.insert(0, cue)

        writer.writerow(row_first)
        writer.writerow(row_distant)


########################################################################################################################

########################################################################################################################
# Création et affichage du nombre d'occurrences pour chacune des réponses données par les participants
# en condition 1 (First) et en condition 2 (Distant)

# Initialisation des fichiers csv et création des headers
header = ['cues']
nb_max_response = len(id_participants)
for i in range(nb_max_response):
    header.append("r_" + str(i+1))

f_nb_occurrences_first = open(f'data/experimental_data/nb_occurrences_by_response_first.csv', 'w')
writer_first = csv.writer(f_nb_occurrences_first)
writer_first.writerow(header)

f_nb_occurrences_distant = open(f'data/experimental_data/nb_occurrences_by_response_distant.csv', 'w')
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
    fig_f_and_ch.suptitle(f"{cue} - Nb d'occurrences des 1ers mots et des mots créatifs choisis par les participants",
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
    ax1.barh(y=df_first_words.mots, width=df_first_words.nb_occurrences)
    ax1.set_title('First words & Nb_occurrences')
    ax1.set(xlabel='nb_occurrences')

    ax2.set_xticks(grid2_x_ticks, minor=True)
    ax2.grid(axis='x', which='major', color='grey', linestyle='-', linewidth=0.75, alpha=0.5)
    ax2.grid(axis='x', which='minor', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.barh(y=df_distant_words.mots, width=df_distant_words.nb_occurrences)
    ax2.set_title('Distant words & Nb_occurrences')
    ax2.set(xlabel='nb_occurrences')

    ####################################################################################################################
    # Sauvegarde des figures obtenues
    file_name = f"data/experimental_data/images/{cue}_first_and_distant_words.png"
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
    f_nb_occurrences_first = open(f'data/experimental_data/nb_occurrences_by_response_first.csv', 'a')
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
    f_nb_occurrences_distant = open(f'data/experimental_data/nb_occurrences_by_response_distant.csv', 'a')
    writer_distant = csv.writer(f_nb_occurrences_distant)

    writer_distant.writerow(row_distant_words)
    writer_distant.writerow(row_nb_occurrences_distant_words)

    ####################################################################################################################
    # Fermeture des 2 fichiers csv
    f_nb_occurrences_first.close()
    f_nb_occurrences_distant.close()
    ####################################################################################################################
