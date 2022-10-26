import functions_v0 as fct
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
# print(complete_dico)

# # on prend une sous-partie du dictionnaire entier
# sub_dico = complete_dico[:100]
# print("Nb de mots dans le dictionnaire complet : ", len(complete_dico))
# print(sub_dico)

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('data/cues.csv', sep=',')
# print(df.shape)
# print(df)

# initialisation de valeurs pour le mot-indice
# on pourra l'enlever par la suite
cue = 'cue'
cue_adequacy = 0
cue_originality = 0

# à la place de cette étape, il faudra charger les données depuis un set de données pré-existant
# par ex. : un set de données/vecteurs construit avec word2vec
# par ex. : fonction load_data(current_word)
# ici, je fais comme si je chargeais le dictionnaire en entier, mais on pourra n'en charger qu'une partie
# ou morceau par morceau (si c'est trop lourd/lent et si on veut ôter quelques mots peu fréquents)
# complete_dico = ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10",
# "word11", "word12", "word13", "word14", "word15", "word16", "word17", "word18", "word19", "word20"]

# For now we take a fixed value to model the size of the semantic network we consider for each word
size_of_semantic_network = 3

# if we want some arbitrary data, for the example
# close_words = ['close_word1', 'close_word2', 'close_word3',
#                 'close_word4', 'close_word5', 'close_word6']
# close_words_adequacy = [90, 85, 80, 75, 70, 65]
# close_words_originality = [10, 15, 20, 25, 30, 35]

# on initialise l'état actuel sur le mot-indice
current_state = State(cue, cue_adequacy, cue_originality)
# at the beginning, the value is equal to 0
current_state.value = 0
print(f"Etat actuel : {current_state.word} - {current_state.adequacy} - {current_state.originality}")

# on initialise les états suivants possibles
next_possible_states = {}

# On fixe le nombre d'étapes de manière arbitraire
# on pourra le changer par la suite (en remplaçant par une fonction par ex.)
steps = 3

# for now, we fix a reward with constant values for each action
# in this example, if the chosen action is the 1, the reward will be 5
reward = [5, 10, 15]


# à modifier (car saute au mot suivant dès que la condition est respectée)
# mettre à jour les règles et implémenter la fonction softargmax
for step in range(steps):
    # on charge une sous-partie du dictionnaire complet
    dico = complete_dico[0:10]
    # à chaque étape, on recrée un tableau des fréquences d'association des mots autour du mot actuel
    dico_freq = fct.get_random_frequency(len(dico))
    print(f"Frequence des mots du dictionnaire : {dico_freq}")

    # a chaque étape, on recrée un réseau sémantique autour du mot actuel
    close_words = fct.select_close_words(dico, dico_freq, size_of_semantic_network)
    close_words_adequacy = fct.get_adequacy(size_of_semantic_network, dico_freq)
    close_words_originality = fct.get_originality(size_of_semantic_network, dico_freq)
    print(f"Mots voisins pris en compte : {close_words}")
    print(f"'Adequacy' des mots voisins pris en compte : {close_words_adequacy}")
    print(f"'Originality' des mots voisins pris en compte : {close_words_originality}")

    # on crée les états possibles suivants
    for i in range(0, len(close_words)):
        next_possible_states[i] = State(close_words[i], close_words_adequacy[i], close_words_originality[i])
        # affichage de l'état possible suivant pour vérifier
        print(f"Etat suivant possible : n°{i} - {next_possible_states[i].word}"
              f" - {next_possible_states[i].adequacy}"
              f" - {next_possible_states[i].originality}")

        # on calcule la valeur associée à chaque nouvel état possible
        next_possible_states[i].value = fct.compute_value(current_state.value, reward[i])
        print(f"Valeur de l'etat {i} : {next_possible_states[i].value}")

    # on sélectionne l'état suivant parmi l'ensemble des états possibles
    chosen_state = fct.select_state(next_possible_states)
    # on met à jour l'état actuel
    current_state = chosen_state

    print(f"Après la phase {step+1} d'exploration, on se trouve dans l'état : {current_state.word}")
