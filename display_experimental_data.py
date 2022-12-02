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
print("Cues : ", cues)
print("Responses : ", all_responses)
print("Nb Cues : ", len(cues))
print("Nb Responses : ", len(all_responses))
########################################################################################################################

########################################################################################################################
# Création de sous-ensembles de données (suivant la condition de passation)
# Condition 1 : First
# Condition 2 : Distant
all_data_cond1 = all_data[all_data['condition'] == 1]
all_data_cond2 = all_data[all_data['condition'] == 2]
print("Nombre de réponses en condition 1 (First) : ", len(all_data_cond1))
print("Nombre de réponses en condition 2 (Distant) : ", len(all_data_cond2))
print("Nombre de réponses au total : ", len(all_data))

all_responses_cond1 = all_data_cond1['responses'].tolist()
all_responses_cond2 = all_data_cond2['responses'].tolist()
########################################################################################################################

########################################################################################################################
# Test préalable participant 72
for num_participant in range(22, 93):
    all_data_participant = all_data[all_data['id_participant'] == num_participant]
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
        print(f"Participant {num_participant} : Problème !")

########################################################################################################################
#
# ########################################################################################################################
# # Création d'un table de données regroupant les réponses des participants pour chaque mot-indice
# col_names = ['cues']
# nb_participants = 70
# for num_participant in range(22, 93):
#     col_name_first = "p_" + str(num_participant) + "f"
#     col_names.append(col_name_first)
#     col_name_distant = "p_" + str(num_participant) + "d"
#     col_names.append(col_name_distant)
# print("Nom des colonnes du dataframe : ", col_names)
# print("Nombre de colonnes du dataframe : ", len(col_names))
# responses_by_participant = pd.DataFrame(columns=col_names)
#
# cue_index = 0
# for cue in cues:
#     row = [cue]
#     for i, response in enumerate(all_data['id_participant']):
#         if all_data['cues'][i] == cue:
#             row.append(response)
#
#     print("Ligne à ajouter au dataframe : ", row)
#     print("Taille de la ligne à ajouter au dataframe : ", len(row))
#     responses_by_participant.loc[cue_index] = row
#     cue_index += 1
# ########################################################################################################################
#
# ########################################################################################################################
# # Création de 2 dictionnaires contenant les réponses pour la condition 1 (First) d'une part
# # et les réponses pour la condition 2 (Distant) d'autre part
# cues_and_responses_cond1 = dict()
# cues_and_responses_cond2 = dict()
#
# for cue in cues:
#     responses_cond1 = []
#     for index, value in enumerate(all_data_cond1['responses']):
#         if all_data_cond1['cues'][index] == cue:
#             responses_cond1.append(value)
#     cues_and_responses_cond1[cue] = responses_cond1
#
#     responses_cond2 = []
#     for index, value in enumerate(all_data_cond2['responses']):
#         if all_data_cond2['cues'][index] == cue:
#             responses_cond2.append(value)
#     cues_and_responses_cond2[cue] = responses_cond2
#
# print("Mots-indices et Réponses pour la condition 1 (First)", cues_and_responses_cond1)
# print("Mots-indices et Réponses pour la condition 2 (Distant)", cues_and_responses_cond2)
# ########################################################################################################################
#
# ########################################################################################################################
# for cue in cues:
#     responses_cond1 = cues_and_responses_cond1[cue]
#     responses_cond2 = cues_and_responses_cond2[cue]
#
#     # Calcul du nombre d'occurrences des 1ers mots donnés par les participants (First)
#     first_words = []
#     first_words_nb_occurrences = []
#     for response in responses_cond1:
#         if response not in first_words:
#             first_words.append(response)
#             first_words_nb_occurrences.append(responses_cond1.count(response))
#
#     print(f"Mots First : {first_words}")
#     print(f"Nb_occurrences mots First : {first_words_nb_occurrences}")
#     df_first_words = pd.DataFrame({
#         'mots': first_words,
#         'nb_occurrences': first_words_nb_occurrences
#     })
#     df_first_words = df_first_words.sort_values(by=['nb_occurrences'])
#
#     # Calcul du nombre d'occurrences des mots créatifs donnés par les participants (Distant)
#     distant_words = []
#     distant_words_nb_occurrences = []
#     for response in responses_cond2:
#         if response not in distant_words:
#             distant_words.append(response)
#             distant_words_nb_occurrences.append(responses_cond2.count(response))
#     print(f"Mots First : {distant_words}")
#     print(f"Nb_occurrences mots First : {distant_words_nb_occurrences}")
#     df_distant_words = pd.DataFrame({
#         'mots': distant_words,
#         'nb_occurrences': distant_words_nb_occurrences
#     })
#     df_distant_words = df_distant_words.sort_values(by=['nb_occurrences'])
#
#     # Affichage des 1ers mots (First) et des mots créatifs (Distant) donnés par les participants
#     # avec leur nombre d'occurrences
#     height = 16
#     width = 10
#     fig_f_and_ch = plt.figure(figsize=(width, height))
#     (ax1, ax2) = fig_f_and_ch.subplots(1, 2)
#     fig_f_and_ch.suptitle(f"{cue} - Nb d'occurrences des 1ers mots et des mots créatifs choisis par les participants",
#                           color='brown', fontsize=14)
#     fig_f_and_ch.tight_layout(h_pad=4, w_pad=7)
#     plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)
#
#     ax1.barh(y=df_first_words.mots, width=df_first_words.nb_occurrences)
#     ax1.set_title('First words & Nb_occurrences')
#     ax1.set(xlabel='nb_occurrences')
#     ax2.barh(y=df_distant_words.mots, width=df_distant_words.nb_occurrences)
#     ax2.set_title('Distant words & Nb_occurrences')
#     ax2.set(xlabel='nb_occurrences')
#
#     # Sauvegarde des figures obtenues
#     file_name = f"data/experimental_data/images/{cue}_first_and_distant_words.png"
#     print(file_name)
#     plt.savefig(file_name)
#     # plt.show()
