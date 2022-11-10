import random
import numpy as np
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
get_adequacy_originality_and_likeability(neighbours, similarities, s_impact_on_a, s_impact_on_o, adequacy_influence)
   return adequacies, originalities, likeabilities

get_adequacy_originality_and_likeability(list(), list(), float, float, float) => list(), list(), list()
       paramètres d'entrée     : neighbours -> la liste des mots voisins à considérer
                                 similarities -> la liste des similarités (fréquences d'association) entre chaque
                                                 mot voisin et le mot-indice
                                 s_impact_on_a -> proportionnalité entre adéquation et freq. d'association (similarité)
                                 s_impact_on_o -> proportionnalité entre originalité et freq. d'association (similarité)
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


def remove_stopwords(existing_dico):
    new_dico = existing_dico.copy()
    words_to_remove = ['</s>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                       'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'ça', 'ces', 'cette', 'cela', 'celui', 'celle',
                       'mais', 'ou', 'et', 'donc', 'or', 'ni', 'car', 'néanmoins', 'si', 'toutefois', 'sinon',
                       'ainsi', 'puis', 'dès', 'jusque', 'cependant', 'pourtant', 'comme', 'lorsque', 'enfin', 'alors',
                       'puisque', 'dont', 'depuis', 'quelque', 'encore', 'chaque',
                       'à', 'au', 'aux', 'afin', 'dans', 'par', 'parmi', 'pour', 'en', 'vers', 'avec', 'de', 'du', 'y',
                       'sans', 'sous', 'sur', 'selon', 'via', 'malgré', 'entre', 'hormis', 'hors',
                       'quel', 'quelle', 'qui', 'que', 'quoi', 'quand', 'comment', 'pourquoi', 'où',
                       'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
                       'moi', 'toi', 'lui', 'eux',
                       'me', 'te', 'ne', 'se', 'leur', 'leurs',
                       'très', 'peu', 'aussi', 'même', 'tout', 'plus', 'aucun',
                       '#', '*', '-', "'",
                       'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'dix',
                       'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
                       'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
                       'cent', 'mille', 'million', 'milliard']
    # problème des homonymes : neuf (le nb VS nouveau), son (le sien VS le bruit), ...
    for word in words_to_remove:
        if word in existing_dico:
            new_dico.remove(word)
    return new_dico


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
        N = 20
        most_similar_words = model.most_similar(cue, topn=N, restrict_vocab=vocab_size)
        most_similar_words = random.choices(most_similar_words, k=nb_neighbours)
    # méthode 4 : identique à la méthode 2 sauf qu'on prend N mots proches
    # et on en sélectionne le nombre désiré (nb_neighbours)
    elif method == 4:
        N = 20
        most_similar_words = model.most_similar_cosmul(cue, topn=N, restrict_vocab=vocab_size)
        most_similar_words = random.choices(most_similar_words, k=nb_neighbours)

    neighbours = []
    similarities = []
    for word in most_similar_words:
        neighbours.append(word[0])
        similarities.append(word[1])
    return neighbours, similarities


def get_adequacy_originality_and_likeability(neighbours, similarities, s_impact_on_a, s_impact_on_o, adequacy_influence):
    adequacies = []
    originalities = []
    likeabilities = []

    delta = 1  # valeur arbitraire

    for i in range(len(neighbours)):
        # calcul de l'adéquation de chaque mot voisin
        adequacy = compute_adequacy(similarities[i], s_impact_on_a)
        # adequacy = coeff_A * np.log10(similarities[i])
        adequacies.append(adequacy)

        # calcul de l'originalité de chaque mot voisin
        originality = compute_adequacy(similarities[i], s_impact_on_o)
        # originality = coeff_O * np.log10(similarities[i]) + coeff_O * (np.log10(similarities[i])**2)
        originalities.append(originality)

        # calcul de l'agréabilité de chaque mot voisin
        likeability = compute_likeability(adequacy, originality, adequacy_influence)
        # likeability = (adequacy_influence * (adequacy ** delta)
        #               + (1-adequacy_influence) * (originality ** delta)) ** (1/delta)
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


def compute_adequacy(similarity, s_impact_on_a):
    adequacy = s_impact_on_a * similarity
    return adequacy


def compute_originality(similarity, s_impact_on_o):
    originality = s_impact_on_o * similarity + s_impact_on_o * similarity ** 2
    return originality


def compute_likeability(adequacy, originality, adequacy_influence):
    likeability = adequacy_influence * adequacy + (1-adequacy_influence) * originality
    return likeability


def get_similarity_between_words(word2vec_model, word1, word2):
    similarity = word2vec_model.similarity(word1, word2)
    return similarity


def get_likeability_to_cue(word2vec_model, cue, word, s_impact_on_a, s_impact_on_o, adequacy_influence):
    likeability_to_cue = 0
    if cue == word:
        likeability_to_cue = 0
    else:
        similarity = get_similarity_between_words(word2vec_model, cue, word)
        adequacy_to_cue = compute_adequacy(similarity, s_impact_on_a)
        originality_to_cue = compute_originality(similarity, s_impact_on_o)
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
    # elif word1_likeability > word2_likeability:
    #     selected_word = word1
    #     selected_word_likeability = word1_likeability
    # else:
    #     if word1 == word2:
    #         print("Les mots sont identiques, on garde le premier")
    #     else:
    #         print("Les mots sont équivalents, on garde le premier")
    return selected_word, selected_word_likeability


def select_best_word_among_all_visited_words(neighbours_data_one_path):
    index_max_value = np.argmax(neighbours_data_one_path['likeability_to_cue'])
    best_word = neighbours_data_one_path['current_word'][index_max_value]
    if '_' in best_word:
        # print("best word avant : ", best_word)
        best_word = best_word[:-2]
        # print("best word après : ", best_word)
    best_word_likeability = neighbours_data_one_path['likeability_to_cue'][index_max_value]
    return best_word, best_word_likeability


def discount_goal_value(discounting_rate, goal_value):
    new_goal_value = (1 - discounting_rate) * goal_value
    return new_goal_value
