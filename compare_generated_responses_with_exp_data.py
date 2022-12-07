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
generated_responses_first = pd.read_csv('data/experimental_data/nb_occurrences_by_response_first.csv', sep=',')
print("Fichier nb_occurrences_by_response_first.csv chargé avec succès.")

# Chargement du fichier csv contenant le nombre d'occurrences des réponses générées par le modèle en condition 2 (Distant)
generated_responses_distant = pd.read_csv('data/experimental_data/nb_occurrences_by_response_distant.csv', sep=',')
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

for cue in cues:
    for row in range(generated_responses_first.shape[0]):
        if generated_responses_first['cues'][row] == cue:
            for col in range(generated_responses_first.shape[1]):
                gen_responses_cond1.append(generated_responses_first.loc[row][col])
                nb_occurrences_gen_responses_cond1.append(generated_responses_first.loc[row+1][col])

    for row in range(generated_responses_distant.shape[0]):
        if generated_responses_distant['cues'][row] == cue:
            for col in range(generated_responses_distant.shape[1]):
                gen_responses_cond2.append(generated_responses_distant.loc[row][col])
                nb_occurrences_gen_responses_cond2.append(generated_responses_distant.loc[row+1][col])

    for row in range(experimental_responses_first.shape[0]):
        if experimental_responses_first['cues'][row] == cue:
            for col in range(experimental_responses_first.shape[1]):
                exp_responses_cond1.append(experimental_responses_first.loc[row][col])
                nb_occurrences_exp_responses_cond1.append(experimental_responses_first.loc[row+1][col])

    for row in range(experimental_responses_distant.shape[0]):
        if experimental_responses_distant['cues'][row] == cue:
            for col in range(experimental_responses_distant.shape[1]):
                exp_responses_cond2.append(experimental_responses_distant.loc[row][col])
                nb_occurrences_exp_responses_cond2.append(experimental_responses_distant.loc[row+1][col])

print("Réponses générées par le modèles en condition 1 : ", gen_responses_cond1)
print("Réponses générées par le modèles en condition 2 : ", gen_responses_cond2)
print("Nombre d'occurrences des réponses générées par le modèles en condition 1 : ", nb_occurrences_gen_responses_cond1)
print("Nombre d'occurrences des réponses générées par le modèles en condition 2 : ", nb_occurrences_gen_responses_cond2)

print("Réponses données par les participants en condition 1 : ", exp_responses_cond1)
print("Réponses données par les participants en condition 2 : ", exp_responses_cond2)
print("Nombre d'occurrences des réponses données par les participants en condition 1 : ", nb_occurrences_exp_responses_cond1)
print("Nombre d'occurrences des réponses données par les participants en condition 2 : ", nb_occurrences_exp_responses_cond2)
