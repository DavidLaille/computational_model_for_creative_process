import functions_v1 as fct
from classes import State
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


# Dans le modèle frWac_no_postag_no_phrase_700_skip_cut50.bin, on a 184 373 mots
# La catégorie grammaticale n'est pas indiquée
pathToModel = "C:/dev/word2vec_pretrained_models/frWac_no_postag_no_phrase_700_skip_cut50.bin"
model = fct.get_model(pathToModel)

# Création du dictionnaire/répertoire de mots
complete_dico = fct.create_dico(model)
# on enlève les mots sans intérêts (genre le, de, et, un, ...)
dico_clean = fct.remove_stopwords(complete_dico)

# # on prend une sous-partie du dictionnaire entier
# sub_dico = complete_dico[:100]
# sub_dico_clean = dico_clean[:100]
# print("Nb de mots dans le dictionnaire complet : ", len(complete_dico))
# print("Nb de mots dans le dictionnaire nettoyé : ", len(dico_clean))
# print(sub_dico)
# print(sub_dico_clean)

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('data/cues.csv', sep=',')

# nb_neighbours : le nombres de mots voisins qu'on veut obtenir
nb_neighbours = 3
# on crée un dataframe dans lequel on rangera les différentes caractéristiques de chaque mot voisin
col_names = ['word', 'similarity', 'adequacy', 'originality']
neighbours_data = pd.DataFrame(columns=col_names)
# l'objectif est représenté par un seuil d'agréabilité
# ici on fixe le seuil à : likeability = 0.8
goal_value = 0.8
q_value = 0
# une variable qui nous permet de fixer le nombre d'item/mots que le modèle gardera en mémoire
# illimité : memory_size = -1
memory_size = -1
vocab_size = 10000
# une variable pour indiquer au modèle quelle méthode utiliser pour calculer la proximité sémantique
# method = 1 : cosine similarity    method = 2 : multiplication combination objective
method = 2
# une liste pour récupérer la liste des mots visités (chemin suivi à travers le réseaux sémantique)
visited_words = list()
best_word = 'best'  # initialisation factice juste pour que la variable soir déclarée dans le bon scope
# un dataframe pour stocker les chemins parcourus
paths = pd.DataFrame(columns=['num_path', 'cue', 'best_word', 'q-value', 'path'])
# nombre d'essais
nb_try = 2

for cue in df['cues']:
    if cue == df['cues'][1]:
        break

    num_path = 0
    for t in range(nb_try):
        # initialisation des variables
        current_word = cue
        current_word_likeability = 0
        nb_steps = 10
        q_value = 0
        # une variable pour représenter le mot final choisi par le modèle
        best_word = current_word
        best_word_likeability = current_word_likeability
        # une liste pour stocker les mots déjà visités, initialisé avec le mot-indice
        words_in_memory = [current_word]
        # on ajoute le mot-indice et sa valeur d'agréabilité dans la liste des mots visités
        visited_words.append([current_word, current_word_likeability])
        while current_word_likeability < goal_value and nb_steps > 0:
            # on récupère les mots voisins, leur fréquence d'association avec le mot-indice
            # puis on récupère les valeurs d'adéquation, d'originalité et d'agréabilité
            neighbours, similarities = fct.get_neighbours_and_similarities(words_in_memory, model,
                                                                           nb_neighbours, vocab_size, method)
            adequacies, originalities, likeabilities = fct.get_adequacy_originality_and_likeability(neighbours, similarities)

            # on remplit le dataframe avec les données obtenues
            neighbours_data['word'] = neighbours
            neighbours_data['similarity'] = similarities
            neighbours_data['adequacy'] = adequacies
            neighbours_data['originality'] = originalities
            neighbours_data['likeability'] = likeabilities
            print("Mots en mémoire : ", words_in_memory)
            # print(neighbours_data)

            # on met à jour la q-value
            q_value = fct.update_value(current_word_likeability, q_value, goal_value, neighbours_data)

            # on passe du mot-indice au mot-voisin avec la plus grande agréabilité/désirabilité (likeability)
            current_word, current_word_likeability = fct.select_next_word(neighbours_data)
            # on ajoute le nouveau mot et sa valeur d'agréabilité dans la liste des mots visités
            visited_words.append([current_word, current_word_likeability])

            # on compare les mots pour savoir lequel est le meilleur (ici, meilleur = haute agréabilité)
            best_word, best_word_likeability = fct.select_best_word(best_word, best_word_likeability, current_word, current_word_likeability)

            # on ajoute le mot dans la liste des mots en mémoire
            if not ({current_word} & set(words_in_memory)):
                words_in_memory.append(current_word)

            # si la capacité de mémoire est infinie, on passe cette étape
            if memory_size == -1:
                pass
            # sinon on retire un mot de la liste lorsque la capacité maximale de mémoire est atteinte
            elif len(words_in_memory) > memory_size:
                # on supprime l'élément le plus ancien
                del words_in_memory[0]

            print(f"{nb_steps} - Mot actuel : {current_word}")
            print(f"q-value : {q_value}")
            print(f"Le mot qui a été choisi est : {best_word}")
            nb_steps -= 1

        print("Mots visités : ", visited_words)
        paths.loc[num_path] = [num_path + 1, cue, best_word, q_value, visited_words]  # on démarrera l'indexation à 1

        num_path += 1

    print(paths)
    paths.to_csv('data/paths.csv', index=False, sep=',')
