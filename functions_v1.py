import random
import numpy as np
import math
from gensim.models import KeyedVectors


'''
########################################################################################################################
# Fonctions disponibles
########################################################################################################################

########################################################################################################################
get_model(path_to_model)
   return model

get_model(string) => gensim.models.KeyedVectors
       paramètres d'entrée     : path_to_model -> chemin d'accès vers un modèle word2vec pré-entraîné
       paramètres de sortie    : model -> modèle une fois chargé

########################################################################################################################
create_dico(model)
   return complete_dico

create_dico(gensim.models.KeyedVectors) => list()
       paramètres d'entrée     : model -> un modèle word2vec pré-entraîné
       paramètres de sortie    : complete_dico -> un dictionnaire sous forme de liste

########################################################################################################################
get_neighbours_and_similarities(cue, model, nb_neighbours, vocab_size, method=1)
   return neighbours, similarities

get_neighbours_and_similarities(string, gensim.models.KeyedVectors, int, int, int) => list(), list()
       paramètres d'entrée     : cue -> le mot-indice pour lequel on souhaite trouver les mots voisins
                                 model -> un modèle word2vec pré-entraîné
                                 nb_neighbours -> le nombre de mots voisins désirés
                                 vocab_size -> la taille du lexique à considérer pour la recherche de mots voisins
                                 method -> la méthode à utiliser pour déterminer les mots voisins
                                           method=1 (par défaut)   : cosine similarity
                                           method=2                : cosmul computation
                                           method=3                : cosine similarity + sélection aléatoire
                                           method=4                : cosmul computation + sélection aléatoire
       paramètres de sortie    : neighbours -> liste des mots voisins trouvés pour le mot-indice considéré
                                 similarities -> liste des valeurs de similarités entre chaque mot voisin et le
                                                 mot-indice considéré

########################################################################################################################
get_adequacy_originality_and_likeability(neighbours, similarities, adequacy_influence)
   return adequacies, originalities, likeabilities

get_adequacy_originality_and_likeability(list(), list(), float, float, float) => list(), list(), list()
       paramètres d'entrée     : neighbours -> la liste des mots voisins à considérer
                                 similarities -> la liste des similarités (fréquences d'association) entre chaque
                                                 mot voisin et le mot-indice
                                 adequacy_influence -> la part d'influence de la valeur d'adéquation dans le calcul
                                                       de l'agréabilité (likeability)
                                                       pour obtenir la part d'originalité : 1 - adequacy_influence
       paramètres de sortie    : adequacies -> liste des valeurs d'adéquation attribuées à chaque mot voisin
                                 originalities -> liste des valeurs d'originalité attribuées à chaque mot voisin
                                 likeabilities -> liste des valeurs d'agréabilité attribuées à chaque mot voisin

########################################################################################################################
get_random_adequacy_originality_and_likeability(neighbours, adequacy_influence)
   return adequacies, originalities, likeabilities

get_random_adequacy_originality_and_likeability(list(), float) => list(), list(), list()
       paramètres d'entrée     : neighbours -> la liste des mots voisins à considérer
                                 adequacy_influence -> la part d'influence de la valeur d'adéquation dans le calcul
                                                       de l'agréabilité (likeability)
       paramètres de sortie    : adequacies -> liste des valeurs d'adéquation attribuées à chaque mot voisin
                                 originalities -> liste des valeurs d'originalité attribuées à chaque mot voisin
                                 likeabilities -> liste des valeurs d'agréabilité attribuées à chaque mot voisin

########################################################################################################################
update_q_value(current_word_likeability, current_q_value, goal_value, neighbours_data, alpha, gamma)
   return new_q_value

update_q_value(float, float, float, pd.dataframe) => float
       paramètres d'entrée     : current_word_likeability -> le degré d'agréabilité du mot actuel
                                 current_q_value -> la q-value de l'état actuel
                                 goal_value -> la valeur d'agréabilité du but à atteindre
                                 neighbours_data -> les données relatives aux mots voisins du mot actuel
                                 alpha -> taux d'apprentissage de l'algo de RL (détermine la vitesse d'apprentissage)
                                 gamma -> taux de prise en compte de la récompense future (détermine dans quelle mesure
                                          on prend en compte les états futurs lors du calcul de la valeur)
       paramètres de sortie    : new_q_value -> une q-value actualisée

########################################################################################################################
select_next_word(neighbours_data)
   return next_word, next_word_likeability, next_word_similarity

select_next_word(pd.dataframe) => string, float, float
       paramètres d'entrée     : neighbours_data -> les données relatives aux mots voisins du mot actuel
       paramètres de sortie    : next_word -> le mot correspondant à l'état suivant, c-à-d le mot qui a été choisi
                                 next_word_likeability -> la valeur d'agréabilité du nouveau mot choisi
                                 next_word_similarity -> le taux de similarité entre le mot choisi et le mot précédent

########################################################################################################################
select_best_word(word1, word1_likeability, word2, word2_likeability)
       return selected_word, selected_word_likeability

select_best_word(pd.dataframe) => string, float
       paramètres d'entrée     : word1 -> le 1er mot à comparer
                                 word1_likeability -> la valeur d'agréabilité du 1er mot
                                 word2 -> le 2e mot à comparer
                                 word2_likeability  -> la valeur d'agréabilité du 2e mot
       paramètres de sortie    : selected_word -> le mot sélectionné à l'issue de la comparaison
                                 selected_word_likeability -> la valeur d'agréabilité du mot sélectionné

########################################################################################################################
discount_goal_value(discounting_rate, goal_value)
   return new_goal_value

discount_goal_value(float, float) => float
       paramètres d'entrée     : discounting_rate -> le taux de décroissance de la valeur d'agréabilité à atteindre
                                 goal_value -> la valeur d'agréabilité actuelle à atteindre (but)
       paramètres de sortie    : new_goal_value -> la valeur d'agréabilité à atteindre (but) mise à jour

########################################################################################################################
'''


def get_model(path_to_model):
    # chargement du modèle
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=True, unicode_errors="ignore")

    return model


# Création du dictionnaire/répertoire de mots
def create_dico(model):
    complete_dico = []
    for index, word in enumerate(model.index_to_key):
        complete_dico.append(word)

    return complete_dico


def get_max_likeability(word2vec_model, cue, adequacy_influence, vocab_size):
    likeabilities = []
    neighbours = word2vec_model.similar_by_key(key=cue, topn=100, restrict_vocab=vocab_size)
    for neighbour in neighbours:
        likeability = get_likeability_to_cue(word2vec_model, cue, neighbour[0], adequacy_influence)
        likeabilities.append(likeability)
    max_likeability = max(likeabilities)

    return max_likeability


def get_neighbours_and_similarities(cue, model, nb_neighbours, vocab_size, method=1):
    most_similar_words = []
    # méthode 1 : calcule la similarité en se basant sur le calcul de la "distance" entre les vecteurs
    if method == 1:
        most_similar_words = model.most_similar(cue, topn=nb_neighbours, restrict_vocab=vocab_size)
    # méthode 2 : calcule la similarité en se basant sur le calcul de "multiplication combination" (cosmul)
    elif method == 2:
        most_similar_words = model.most_similar_cosmul(cue, topn=nb_neighbours, restrict_vocab=vocab_size)
    # méthode 3 : identique à la méthode 1 sauf qu'on prend N mots proches
    # et on en sélectionne le nombre désiré (nb_neighbours)
    elif method == 3:
        nb_options = nb_neighbours * 10
        most_similar_words = model.most_similar(cue, topn=nb_options, restrict_vocab=vocab_size)
        most_similar_words = random.choices(most_similar_words, k=nb_neighbours)
    # méthode 4 : identique à la méthode 2 sauf qu'on prend N mots proches
    # et on en sélectionne le nombre désiré (nb_neighbours)
    elif method == 4:
        nb_options = nb_neighbours * 10
        most_similar_words = model.most_similar_cosmul(cue, topn=nb_options, restrict_vocab=vocab_size)
        most_similar_words = random.choices(most_similar_words, k=nb_neighbours)

    neighbours = []
    similarities = []
    for word in most_similar_words:
        neighbours.append(word[0])
        similarities.append(word[1])

    return neighbours, similarities


def get_adequacy_originality_and_likeability(neighbours, similarities, adequacy_influence):
    adequacies = []
    originalities = []
    likeabilities = []

    for i in range(len(neighbours)):
        # calcul de l'adéquation de chaque mot voisin
        adequacy = compute_adequacy(similarities[i])
        adequacies.append(adequacy)

        # calcul de l'originalité de chaque mot voisin
        originality = compute_originality(similarities[i])
        originalities.append(originality)

        # calcul de l'agréabilité de chaque mot voisin
        likeability = compute_likeability(adequacy, originality, adequacy_influence)
        likeabilities.append(likeability)

    return adequacies, originalities, likeabilities


def get_random_adequacy_originality_and_likeability(neighbours, adequacy_influence):
    adequacies = []
    originalities = []
    likeabilities = []

    for i in range(len(neighbours)):
        # calcul de l'adéquation de chaque mot voisin
        adequacy = random.random()
        adequacies.append(adequacy)

        # calcul de l'originalité de chaque mot voisin
        originality = random.random()
        originalities.append(originality)

        # calcul de l'agréabilité de chaque mot voisin
        likeability = adequacy_influence * adequacy + (1-adequacy_influence) * originality
        likeabilities.append(likeability)

    return adequacies, originalities, likeabilities


def compute_adequacy(similarity):
    muA_3 = 1.2  # coeff multiplicateur de sim^3
    muA_2 = -1.8  # coeff multiplicateur de sim^2
    muA_1 = 1  # coeff multiplicateur de sim^1
    muA_0 = 0.7  # coeff multiplicateur de sim^0
    adequacy = muA_3 * similarity ** 3 + muA_2 * similarity ** 2 + muA_1 * similarity + muA_0
    adequacy = adequacy + float(random.randint(-100, 100)) / 1000.0  # ajout d'incertitude

    if adequacy < 0:
        adequacy = 0
    elif adequacy > 1:
        adequacy = 1

    return adequacy


def compute_originality(similarity):
    muO_3 = -0.5  # coeff multiplicateur de sim^3
    muO_2 = 1  # coeff multiplicateur de sim^2
    muO_1 = -1.3  # coeff multiplicateur de sim^1
    muO_0 = 0.8  # coeff multiplicateur de sim^0
    originality = muO_3 * similarity ** 3 + muO_2 * similarity ** 2 + muO_1 * similarity + muO_0
    originality = originality + float(random.randint(-100, 100)) / 1000.0  # ajout d'incertitude

    if originality < 0:
        originality = 0
    elif originality > 1:
        originality = 1

    return originality


def compute_likeability(adequacy, originality, adequacy_influence):
    delta = 0.8
    likeability = (adequacy_influence * (adequacy ** delta)
                   + (1-adequacy_influence) * (originality ** delta)) ** (1/delta)

    return likeability


def get_similarity_between_words(word2vec_model, word1, word2):
    similarity = word2vec_model.similarity(word1, word2)

    return similarity


def get_likeability_to_cue(word2vec_model, cue, word, adequacy_influence):
    likeability_to_cue = 0
    if cue == word:
        likeability_to_cue = 0
    else:
        similarity = get_similarity_between_words(word2vec_model, cue, word)
        adequacy_to_cue = compute_adequacy(similarity)
        originality_to_cue = compute_originality(similarity)
        likeability_to_cue = compute_likeability(adequacy_to_cue, originality_to_cue, adequacy_influence)

    return likeability_to_cue


def update_q_value(current_word_likeability, current_q_value, goal_value, neighbours_data, alpha, gamma):
    # détermination de la récompense
    # à remplacer par une fonction compute_reward() à l'avenir
    if current_word_likeability > goal_value:
        r = 1
    else:
        r = 0

    li_max = max(neighbours_data['likeability'])

    # calcul de la q-value
    new_q_value = current_q_value + alpha * (r + gamma * (li_max - goal_value))
    # new_value = (current_word_likeability - goal_value) + alpha * (r + gamma * max(neighbours_data['likeability'] - goal_value))

    return new_q_value


def select_next_word(neighbours_data):
    index_max_value = np.argmax(neighbours_data['likeability'])
    next_word = neighbours_data['neighbours'][index_max_value]
    next_word_likeability = neighbours_data['likeability'][index_max_value]
    next_word_similarity = neighbours_data['similarity'][index_max_value]

    return next_word, next_word_likeability, next_word_similarity


def select_best_word(word1, word1_likeability, word2, word2_likeability):
    selected_word = word1
    selected_word_likeability = word1_likeability
    if word1_likeability < word2_likeability:
        selected_word = word2
        selected_word_likeability = word2_likeability

    return selected_word, selected_word_likeability


def select_best_word_among_all_visited_words(neighbours_data_one_path):
    index_max_value = np.argmax(neighbours_data_one_path['likeability_to_cue'])
    best_word = neighbours_data_one_path['current_word'][index_max_value]
    while '_' in best_word:  # on supprime le tag indiquant le numéro du step
        # print("best word avant : ", best_word)
        best_word = best_word[:-1]
        # print("best word après : ", best_word)
    best_word_likeability = neighbours_data_one_path['likeability_to_cue'][index_max_value]

    return best_word, best_word_likeability


def discount_goal_value_linear(decrease_amount, goal_value):
    new_goal_value = goal_value - decrease_amount

    return new_goal_value


def discount_goal_value(discounting_rate, goal_value):
    # cette fonction est une exponentielle de base a avec : a = 1 - discounting_rate
    new_goal_value = (1 - discounting_rate) * goal_value

    return new_goal_value


def discount_goal_value_exp(discounting_rate, goal_value):
    # si le discounting_rate est élevé, la diminution de la goal_value sera lente au départ puis très rapide
    new_goal_value = (-1) * math.exp((-1) * discounting_rate * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-2) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-3) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-4) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-5) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-6) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-7) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-8) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-9) * goal_value) + 1
    # new_goal_value = (-1) * math.exp((-10) * goal_value) + 1

    return new_goal_value


def discount_goal_value_log(discounting_rate, goal_value):
    # si le discounting_rate est élevé, la diminution de la goal_value sera lente au départ puis très rapide
    new_goal_value = (math.log(goal_value)/discounting_rate) + 1
    # new_goal_value = (math.log(goal_value)/2) + 1
    # new_goal_value = (math.log(goal_value)/3) + 1
    # new_goal_value = (math.log(goal_value)/4) + 1
    # new_goal_value = (math.log(goal_value)/5) + 1
    # new_goal_value = (math.log(goal_value)/6) + 1
    # new_goal_value = (math.log(goal_value)/7) + 1
    # new_goal_value = (math.log(goal_value)/8) + 1
    # new_goal_value = (math.log(goal_value)/9) + 1
    # new_goal_value = (math.log(goal_value)/10) + 1

    return new_goal_value


def discount_goal_value_sqrt(goal_value):
    # avec la fonction "racine-carrée", la diminution de la goal_value sera lente au départ puis très rapide
    if goal_value == 1:
        new_goal_value = 0.99
        # on décrémente un peu la goal_value car sqrt(1) = 1 (on ne bougera pas si on applique le calcul)
    else:
        new_goal_value = math.sqrt(goal_value)

    return new_goal_value
