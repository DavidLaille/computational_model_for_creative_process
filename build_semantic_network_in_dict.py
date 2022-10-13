import csv
from pathlib import Path
import pandas as pd
from gensim.models import KeyedVectors

# Dans le modèle frWac_postag_no_phrase_700_skip_cut50.bin, on a 215 020 mots
# Pour chacun, il est indiqué la catégorie grammaticale par "_d", "_p", "_v"
# pathToModel = "C:/dev/word2vec_pretrained_models/frWac_postag_no_phrase_700_skip_cut50.bin"

# Dans le modèle frWac_no_postag_no_phrase_700_skip_cut50.bin, on a 184 373 mots
# La catégorie grammaticale n'est pas indiquée
pathToModel = "C:/dev/word2vec_pretrained_models/frWac_no_postag_no_phrase_700_skip_cut50.bin"

# chargement du modèle
model = KeyedVectors.load_word2vec_format(pathToModel, binary=True, unicode_errors="ignore")

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('data/cues.csv', sep=',')
print(df.shape)
networks = dict()

# depth : the depth of the semantic network
depth = 3
# N : le nombres de mots voisins qu'on veut obtenir
N = 2
matrix_size = N ** depth  # N puissance depth
# on initialise une variable qui indiquera le nb de fois qu'il faut écrire le mot dans la matrice
# pour la cue, cette nb_iter = matrix size
# pour les mots de profondeur 1, nb_iter = matrix_size/N**1
# pour les mots de profondeur 2, nb_iter = matrix_size/N**2
nb_iter = matrix_size

for i, row in df.iterrows():
    # une liste pour stocker les mots déjà visités
    visited_words = []
    # On récupère le mot-indice
    cue = row["cues"]
    # on initialise le mot courant avec la valeur du mot-indice et une valeur de fréquence bidon,
    # on initialise une liste de tuple pour que le modèle puisse traiter des objets du même type
    current_word = [(cue, 0)]
    # on remplit une liste de taille 'matrix_size' avec le mot-indice
    cue_column = [current_word[0][0] for j in range(matrix_size)]
    # on construit la première colonne du dictionnaire 'network'
    networks[cue] = dict()
    networks[cue]["cue"] = cue_column
    for d in range(depth):
        nb_iter = int(matrix_size / (N ** (d + 1)))
        key = "depth_" + str(d + 1)  # on ajoute 1 pour avoir les profondeurs à partir de 1 (et pas 0)
        column = []
        neighbours_current_word = []
        all_neighbours = []
        for w in range(len(current_word)):
            visited_words.append(current_word[w][0])
            neighbours_current_word = model.most_similar(current_word[w][0], topn=N, restrict_vocab=10000)
            # on met en mémoire tous les mots-voisins obtenus
            all_neighbours.extend(neighbours_current_word)

            for neighbour in neighbours_current_word:
                for k in range(nb_iter):
                    # ici on ne prend que le mot voisin mais on peut aussi prendre son taux de similarité
                    # en prenant neighbour au lieu de neighbour[0]
                    column.append(neighbour[0])

        # le mot courant à prendre en compte prend la valeur des mots voisins précédemment trouvés
        current_word = all_neighbours
        # on écrit les mots dans le dictionnaire
        networks[cue][key] = column

# autres méthodes pour trouver les mots similaires :
# most_similar_cosmul() & most_similar_given_key()

# cues = ["jardin", "doigt", "vin", "avis", "pierre", "appel"]

# Quelques print() pour vérifier que tout s'est bien passé
# for key, value in networks.items():
#     print(key)
#     print(value)

# print(model.most_similar("jardin", topn=N, restrict_vocab=10000))
# print(model.most_similar("doigt", topn=N, restrict_vocab=10000))
# print(model.most_similar("vin", topn=N, restrict_vocab=10000))
# print(model.most_similar("avis", topn=N, restrict_vocab=10000))
# print(model.most_similar("pierre", topn=N, restrict_vocab=10000))
# print(model.most_similar("appel", topn=N, restrict_vocab=10000))

# une liste pour stocker les noms des colonnes
column_names = []
keys = list(networks.keys())
first_key = keys[0]
for key, value in networks[first_key].items():
    column_names.append(str(key))

# # on crée un chemin vers le csv où on veut enregistrer nos données
# filepath = Path('data/semantic_networks.csv')
# filepath.parent.mkdir(parents=True, exist_ok=True)

with open('data/semantic_networks.csv', 'w', newline='', encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(column_names)
    for key, value in networks.items():
        for i in range(matrix_size):
            row = []
            for key2, value2 in value.items():
                row.append(value2[i])
            writer.writerow(row)
