import pandas as pd


# Chargement des mots-indices depuis le fichier csv
df_cues = pd.read_csv('data/experimental_data/cues.csv', sep=',')
print("Fichier cues.csv chargé avec succès.")


# Chargement du fichier csv contenant le nombre d'occurrences pour les réponses des participants en condition 1 (First)
experimental_responses_first = pd.read_csv('data/experimental_data/nb_occurrences_by_response_first.csv', sep=',')
print("Fichier nb_occurrences_by_response_first.csv chargé avec succès.")

# Chargement du fichier csv contenant le nombre d'occurrences pour les réponses des participants en condition 2 (Distant)
experimental_responses_distant = pd.read_csv('data/experimental_data/nb_occurrences_by_response_distant.csv', sep=',')
print("Fichier nb_occurrences_by_response_distant.csv chargé avec succès.")

# Chargement du fichier csv contenant le nombre d'occurrences des réponses générées par le modèle en condition 1 (First)
generated_responses_first = pd.read_csv('data/generated_data/nb_occurrences_by_response_first.csv', sep=',')
print("Fichier nb_occurrences_by_response_first.csv chargé avec succès.")

# Chargement du fichier csv contenant le nombre d'occurrences des réponses générées par le modèle en condition 2 (Distant)
generated_responses_distant = pd.read_csv('data/generated_data/nb_occurrences_by_response_distant.csv', sep=',')
print("Fichier nb_occurrences_by_response_distant.csv chargé avec succès.")

cues = df_cues['cues'].tolist()
# print("Mots-indice : ", cues)

exp_responses_cond1 = []
exp_responses_cond2 = []
gen_responses_cond1 = []
gen_responses_cond2 = []

nb_occurrences_exp_responses_cond1 = []
nb_occurrences_exp_responses_cond2 = []
nb_occurrences_gen_responses_cond1 = []
nb_occurrences_gen_responses_cond2 = []

common_responses_first = []
common_responses_distant = []
nb_common_responses_first = 0
nb_common_responses_distant = 0

for cue in cues:
    # Pour chaque cue, on remet à 0 les listes de mots générés par le modèle ou donnés par les participants
    exp_responses_cond1 = []
    exp_responses_cond2 = []
    gen_responses_cond1 = []
    gen_responses_cond2 = []

    nb_occurrences_exp_responses_cond1 = []
    nb_occurrences_exp_responses_cond2 = []
    nb_occurrences_gen_responses_cond1 = []
    nb_occurrences_gen_responses_cond2 = []

    common_responses_first = []
    common_responses_distant = []
    nb_common_responses_first = 0
    nb_common_responses_distant = 0

    ####################################################################################################################
    # Récupération des réponses expérimentales (données par les participants) et simulées (générées par le modèle)
    ####################################################################################################################
    for row in range(generated_responses_first.shape[0]):
        if generated_responses_first['cues'][row] == cue:
            for col in range(generated_responses_first.shape[1]):
                if col == 0:    # on ne prend pas en compte le mot-indice
                    continue
                elif generated_responses_first.loc[row][col] != generated_responses_first.loc[row][col]:    # 'nan'
                    continue
                gen_responses_cond1.append(generated_responses_first.loc[row][col])
                nb_occurrences_gen_responses_cond1.append(generated_responses_first.loc[row+1][col])
    print("###########################################################################################################")
    print(f"{cue} - Nombre de réponses générées par le modèle en condition 1 : {len(gen_responses_cond1)}")
    print("###########################################################################################################")

    for row in range(generated_responses_distant.shape[0]):
        if generated_responses_distant['cues'][row] == cue:
            for col in range(generated_responses_distant.shape[1]):
                if col == 0:    # on ne prend pas en compte le mot-indice
                    continue
                elif generated_responses_distant.loc[row][col] != generated_responses_distant.loc[row][col]:    # 'nan'
                    continue
                gen_responses_cond2.append(generated_responses_distant.loc[row][col])
                nb_occurrences_gen_responses_cond2.append(generated_responses_distant.loc[row+1][col])
    print("###########################################################################################################")
    print(f"{cue} - Nombre de réponses générées par le modèle en condition 2 : {len(gen_responses_cond2)}")
    print("###########################################################################################################")

    for row in range(experimental_responses_first.shape[0]):
        if experimental_responses_first['cues'][row] == cue:
            for col in range(experimental_responses_first.shape[1]):
                if col == 0:    # on ne prend pas en compte le mot-indice
                    continue
                elif experimental_responses_first.loc[row][col] != experimental_responses_first.loc[row][col]:    # 'nan'
                    continue
                exp_responses_cond1.append(experimental_responses_first.loc[row][col])
                nb_occurrences_exp_responses_cond1.append(experimental_responses_first.loc[row+1][col])
    print("###########################################################################################################")
    print(f"{cue} - Nombre de réponses données par les participants en condition 1 : {len(exp_responses_cond1)}")
    print("###########################################################################################################")

    for row in range(experimental_responses_distant.shape[0]):
        if experimental_responses_distant['cues'][row] == cue:
            for col in range(experimental_responses_distant.shape[1]):
                if col == 0:    # on ne prend pas en compte le mot-indice
                    continue
                elif experimental_responses_distant.loc[row][col] != experimental_responses_distant.loc[row][col]:    # 'nan'
                    continue
                exp_responses_cond2.append(experimental_responses_distant.loc[row][col])
                nb_occurrences_exp_responses_cond2.append(experimental_responses_distant.loc[row+1][col])
    print("###########################################################################################################")
    print(f"{cue} - Nombre de réponses données par les participants en condition 2 : {len(exp_responses_cond2)}")
    print("###########################################################################################################")

    ####################################################################################################################

    ####################################################################################################################
    # Comparaisons entre les données générées par le modèle et celles données par les participants
    ####################################################################################################################
    # Condition 1 (First)
    # Y a-t-il des réponses communes entre les réponses données par les participants et les réponses générées par le modèle ?
    for response_first in exp_responses_cond1:
        if response_first in gen_responses_cond1:
            if response_first == cue:       # si c'est le mot-indice, on passe (car le modèle ne génère pas le mot indice)
                continue
            common_responses_first.append(response_first)
            nb_common_responses_first += 1
    print("###########################################################################################################")
    print(f"Cond1 - Réponses communes : {common_responses_first}")
    print(f"Cond1 - Nombre de réponses communes : {nb_common_responses_first}")
    print(f"Cond1 - Pourcentage de réponses expérimentales trouvées dans les réponses du modèle : {nb_common_responses_first/len(gen_responses_cond1)}")
    print(f"Cond1 - Pourcentage de réponses générées par le modèle trouvées dans les réponses des participants : {nb_common_responses_first/len(exp_responses_cond1)}")
    print("###########################################################################################################")

    # Condition 2 (Distant)
    # Y a-t-il des réponses communes entre les réponses données par les participants et les réponses générées par le modèle ?
    for response_distant in exp_responses_cond2:
        if response_distant in gen_responses_cond2:
            if response_distant == cue:     # si c'est le mot-indice, on passe (car le modèle ne génère pas le mot indice)
                continue
            common_responses_distant.append(response_distant)
            nb_common_responses_distant += 1
    print("###########################################################################################################")
    print(f"Cond2 - Réponses communes : {common_responses_distant}")
    print(f"Cond2 - Nombre de réponses communes : {nb_common_responses_distant}")
    print(f"Cond2 : Pourcentage de réponses expérimentales trouvées dans les réponses du modèle : {nb_common_responses_distant/len(gen_responses_cond2)}")
    print(f"Cond2 : Pourcentage de réponses générées par le modèle trouvées dans les réponses des participants : {nb_common_responses_distant/len(exp_responses_cond2)}")
    print("###########################################################################################################")

    ####################################################################################################################

    print("###########################################################################################################")
    print("Réponses générées par le modèles en condition 1 : ", gen_responses_cond1)
    print("Réponses générées par le modèles en condition 2 : ", gen_responses_cond2)
    print("Nombre d'occurrences des réponses générées par le modèles en condition 1 : ", nb_occurrences_gen_responses_cond1)
    print("Nombre d'occurrences des réponses générées par le modèles en condition 2 : ", nb_occurrences_gen_responses_cond2)
    print("###########################################################################################################")

    print("###########################################################################################################")
    print("Réponses données par les participants en condition 1 : ", exp_responses_cond1)
    print("Réponses données par les participants en condition 2 : ", exp_responses_cond2)
    print("Nombre d'occurrences des réponses données par les participants en condition 1 : ", nb_occurrences_exp_responses_cond1)
    print("Nombre d'occurrences des réponses données par les participants en condition 2 : ", nb_occurrences_exp_responses_cond2)
    print("###########################################################################################################")
