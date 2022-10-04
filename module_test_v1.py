<<<<<<< Updated upstream
import functions as fct
from classes import State

cue = 'cue'
cue_adequacy = 0
cue_originality = 0

# à la place de cette étape, il faudra charger les données depuis un set de données pré-existant
# par ex. : un set de données/vecteurs construit avec word2vec
# par ex. : fonction load_data(current_word)
close_words = ['close_word1', 'close_word2', 'close_word3',
               'close_word4', 'close_word5', 'close_word6']
close_words_adequacy = [90, 85, 80, 75, 70, 65]
close_words_originality = [10, 15, 20, 25, 30, 35]

# on initialise l'état actuel sur le mot-indice
current_state = State(cue, cue_adequacy, cue_originality)
print(f"Etat actuel : {current_state.word} - {current_state.adequacy} - {current_state.originality}")
# on intialise les états suivants possibles
next_possible_states = {}
for i in range(0, len(close_words)):
    next_possible_states[i] = State(close_words[i], close_words_adequacy[i], close_words_originality[i])
    # affichage de l'état possible suivant pour vérifier
    print(f"Etat suivant possible : {next_possible_states[i].word}"
          f" - {next_possible_states[i].adequacy}"
          f" - {next_possible_states[i].originality}")

# Pour l'exemple, on prend une règle de comportement
# ici la règle n°1 : maximisation de l'adéquation
policy = 1

for j in range(1, len(next_possible_states)):
    if fct.policy_respected(policy, current_state, next_possible_states[j]):
        fct.change_state(current_state, next_possible_states[j])
        print(current_state.word)
    else:
        print(f"{next_possible_states[j].word} ne respecte pas la règle appliquée.")

print(f"Après la première phase d'exploration, on se trouve dans l'état : {current_state.word}")
=======
import functions as fct
from classes import State

cue = 'cue'
cue_adequacy = 0
cue_originality = 0


# à la place de cette étape, il faudra charger les données depuis un set de données pré-existant
# par ex. : un set de données/vecteurs construit avec word2vec
# par ex. : fonction load_data(current_word)
dico = ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10"]
dico_freq = fct.get_frequency(len(dico))
print(dico_freq)

# For now we take a fixed value to model the size of the semantic network we consider for each word
size_of_semantic_network = 3

close_words = fct.select_close_words(dico, dico_freq, size_of_semantic_network)
close_words_adequacy = fct.get_adequacy(size_of_semantic_network)
close_words_originality = fct.get_originality(size_of_semantic_network)
print(close_words)
print(close_words_adequacy)
print(close_words_originality)

# if we want some arbitrary data, for the example
# close_words = ['close_word1', 'close_word2', 'close_word3',
#                 'close_word4', 'close_word5', 'close_word6']
# close_words_adequacy = [90, 85, 80, 75, 70, 65]
# close_words_originality = [10, 15, 20, 25, 30, 35]

# on initialise l'état actuel sur le mot-indice
current_state = State(cue, cue_adequacy, cue_originality)
print(f"Etat actuel : {current_state.word} - {current_state.adequacy} - {current_state.originality}")
# on intialise les états suivants possibles
next_possible_states = {}
for i in range(0, len(close_words)):
    next_possible_states[i] = State(close_words[i], close_words_adequacy[i], close_words_originality[i])
    # affichage de l'état possible suivant pour vérifier
    print(f"Etat suivant possible : {next_possible_states[i].word}"
          f" - {next_possible_states[i].adequacy}"
          f" - {next_possible_states[i].originality}")

# Pour l'exemple, on prend une règle de comportement
# ici la règle n°1 : maximisation de l'adéquation
policy = 1
# On fixe le nombre d'étapes de manière arbitaire
# on pourra le changer par la suite (en remplaçant par une fonction par ex.)
steps = 1

# à modifier (car saute au mot suivant dès que la condition est respectée)
# mettre à jour les règles et implémenter la fonction softargmax
for step in range(steps):
    for j in range(len(next_possible_states)):
        if fct.policy_respected(policy, current_state, next_possible_states[j]):
            fct.change_state(current_state, next_possible_states[j])
            print(current_state.word)
        else:
            print(f"{next_possible_states[j].word} ne respecte pas la règle appliquée.")

    print(f"Après la phase {step+1} d'exploration, on se trouve dans l'état : {current_state.word}")
>>>>>>> Stashed changes
