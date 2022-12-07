import pandas as pd


# Chargement des mots-indices depuis le fichier csv
df_cues = pd.read_csv('data/experimental_data/cues.csv', sep=',')
print("Fichier cues.csv chargé avec succès.")

# Chargement du fichier csv contenant les réponses des participants
experimental_responses = pd.read_csv('data/experimental_data/responses_by_cue_2_lines.csv', sep=',')
print("Fichier responses_by_cue_2_lines.csv chargé avec succès.")

# Chargement du fichier csv contenant les réponses générées par le modèle
generated_responses = pd.DataFrame()

cues = df_cues['cues']
print("Mots-indice : ", cues)

for cue in cues:
    print(cue)
