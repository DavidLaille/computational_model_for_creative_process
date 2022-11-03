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
networks = pd.DataFrame()
matrix = pd.DataFrame()
# la liste avec les noms de colonnes, qu'on remplira plus tard
col_names = ["cue"]

# depth : the depth of the semantic network
depth = 3
# N : le nombres de mots voisins qu'on veut obtenir
N = 3
matrix_size = N ** depth  # N puissance depth
# on initialise une variable qui indiquera le nb de fois qu'il faut écrire le mot dans la matrice
# pour la cue, cette nb_iter = matrix size
# pour les mots de profondeur 1, nb_iter = matrix_size/N**1
# pour les mots de profondeur 2, nb_iter = matrix_size/N**2 ...
nb_iter = matrix_size

for i, row in df.iterrows():
    # une liste pour stocker les mots déjà visités
    visited_words = []
    visited_words_one_back_memory = []
    # On récupère le mot-indice
    cue = row["cues"]
    # on initialise le mot courant avec la valeur du mot-indice et une valeur de fréquence bidon,
    # on initialise une liste de tuple pour que le modèle puisse traiter des objets du même type
    current_words = [(cue, 0)]
    # on remplit une liste de taille 'matrix_size' avec le mot-indice
    cue_column = [current_words[0][0] for j in range(matrix_size)]
    # on construit la première colonne (cue) du dictionnaire 'network'
    matrix["cue"] = cue_column
    for d in range(depth):
        print("d: ", d)
        nb_iter = int(matrix_size / (N ** (d + 1)))
        col_name = "depth_" + str(d + 1)  # on ajoute 1 pour avoir les profondeurs à partir de 1 (et pas 0)
        col_names.append(col_name)

        col = []
        neighbours_current_word = []
        all_neighbours = []
        visited_words_one_back_memory = []
        for w in range(len(current_words)):
            short_term_memory = []
            if not ({current_words[w][0]} & set(visited_words)):
                visited_words.append(current_words[w][0])
            print("visited words : ", visited_words)

            # neighbours_current_word = model.most_similar(current_words[w][0], topn=N, restrict_vocab=10000)
            neighbours_current_word = model.most_similar(visited_words, topn=N, restrict_vocab=10000)
            # si on ne veut prendre en compte que les mots de l'étape précédente
            # if len(visited_words_one_back_memory) < 1:
            #     neighbours_current_word = model.most_similar(current_words[w][0], topn=N, restrict_vocab=10000)
            # else:
            #     neighbours_current_word = model.most_similar(visited_words_one_back_memory,
            #                                                  topn=N,
            #                                                  restrict_vocab=10000)

            for neighbour in neighbours_current_word:
                # OPTION 1 : on mémorise tous les mots rencontrés
                # si le mot n'a pas déjà été visité, on l'ajoute à la liste des mots visités
                if not({neighbour[0]} & set(visited_words)):
                    visited_words.append(neighbour[0])
                    # pour simuler la capacité limitée de mémoire de travail,
                    # on pourra ne garder qu'un certain nombre de mots (les 7 ou 10 derniers par ex.)

                # OPTION 2 : on mémorise uniquement les mots rencontrés à l'étape d'avant
                # short_term_memory.append(neighbour[0])
                for k in range(nb_iter):
                    # ici on ne prend que le mot voisin mais on peut aussi prendre son taux de similarité
                    # en prenant neighbour au lieu de neighbour[0]
                    col.append(neighbour[0])

            # print("short-term memory : ", short_term_memory)
            # visited_words_one_back_memory.extend(short_term_memory)
            # on met en mémoire tous les mots-voisins obtenus
            all_neighbours.extend(neighbours_current_word)

        # print("last session visited words : ", visited_words_one_back_memory)
        # le mot courant à prendre en compte prend la valeur des mots voisins précédemment trouvés
        current_words = all_neighbours
        # on rajoute la colonne dans le dataframe
        matrix[col_name] = col

    # on regarde quels mots ont été visités
    print("visited words : ", visited_words)
    #  on ajoute la matrice au dataframe existant
    networks = pd.concat([networks, matrix], ignore_index=True)

# autres méthodes pour trouver les mots similaires :
# most_similar_cosmul() & most_similar_given_key()

# Quelques print() pour vérifier que tout s'est bien passé
# print(networks)

networks.to_csv('data/semantic_networks.csv')

# si on ne veut pas l'index et qu'on veut un espace comme séparateur (au lieu de la virgule)
# networks.to_csv('data/semantic_networks.csv', index=False, sep='\t')
