from operator import index
import random
from classes import State
import numpy as np
from gensim.models import KeyedVectors


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
    words_to_remove = ['</s>',
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
                       'a', '#',
                       'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'dix',
                       'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
                       'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
                       'cent', 'mille', 'million', 'milliard']
    # je garde neuf car il peut vouloir dire nouveau
    for word in words_to_remove:
        if word in existing_dico:
            new_dico.remove(word)
    return new_dico


# we will change the parameters of this function by using the vectors coming from word2vec
def get_neighbours_and_similarities(cue, model, nb_neighbours, vocab_size, method=1):
    # méthode 1 : calcule la similarité en se basant sur le calcul de la "distance" entre les vecteurs
    if method == 1:
        most_similar_words = model.most_similar(cue, topn=nb_neighbours, restrict_vocab=vocab_size)
    # méthode 2 : calcule la similarité en se basant sur le calcul de
    elif method == 2:
        most_similar_words = model.most_similar_cosmul(cue, topn=nb_neighbours, restrict_vocab=vocab_size)

    close_words = []
    similarities = []
    for word in most_similar_words:
        close_words.append(word[0])
        similarities.append(word[1])
    return close_words, similarities


def get_adequacy_originality_and_likeability(neighbours, similarities):
    adequacies = []
    originalities = []
    likeabilities = []

    # coeffs pour le calcul de l'adéquation (adequacy)
    coeff_A = random.random()  # valeur entre 0 et 1
    # coeffs pour le calcul de l'originalité (originality)
    l = 1;    q = 1;    coeff_O = random.random()  # valeur entre 0 et 1
    # coeffs pour le calcul de l'agréabilité (likeability)
    alpha = 0.5  # valeur arbitraire
    delta = 1  # valeur arbitraire

    for i in range(len(neighbours)):
        # calcul de l'adéquation de chaque mot voisin
        adequacy = coeff_A * similarities[i]
        # adequacy = coeff_A * np.log10(similarities[i])
        adequacies.append(adequacy)

        # calcul de l'originalité de chaque mot voisin
        originality = (coeff_O ** l) * similarities[i] + (coeff_O ** q) * similarities[i]**2
        # originality = (coeff_O ** l) * np.log10(similarities[i]) + (coeff_O ** q) * (np.log10(similarities[i])**2)
        originalities.append(originality)

        # calcul de l'agréabilité de chaque mot voisin
        likeability = alpha * originality + (1-alpha) * adequacy
        # likeability = (alpha * (originality ** delta) + (1-alpha) * (adequacy ** delta)) ** (1/delta)
        likeabilities.append(likeability)

    return adequacies, originalities, likeabilities


def get_random_adequacy_originality_and_likeability(neighbours, similarities):
    adequacies = []
    originalities = []
    likeabilities = []
    alpha = 0.5  # valeur arbitraire

    for i in range(len(neighbours)):
        # calcul de l'adéquation de chaque mot voisin
        adequacy = random.random()
        adequacies.append(adequacy)

        # calcul de l'originalité de chaque mot voisin
        originality = random.random()
        originalities.append(originality)

        # calcul de l'agréabilité de chaque mot voisin
        likeability = alpha * originality + (1-alpha) * adequacy
        likeabilities.append(likeability)

    return adequacies, originalities, likeabilities


def update_value(current_word_likeability, current_q_value, goal_value, neighbours_data):
    alpha = 0.5  # valeur arbitraire
    gamma = 0.4  # valeur arbitraire

    # détermination de la récompense
    if current_word_likeability > goal_value:
        r = 1
    else:
        r = 0

    # calcul de la q-value
    new_value = current_q_value + alpha * (r + gamma * max(neighbours_data['likeability'] - goal_value))
    # new_value = (current_word_likeability - goal_value) + alpha * (r + gamma * max(neighbours_data['likeability'] - goal_value))
    return new_value


def select_next_word(neighbours_data):
    index_max_value = np.argmax(neighbours_data['likeability'])
    next_word = neighbours_data['word'][index_max_value]
    next_word_likeability = neighbours_data['likeability'][index_max_value]
    return next_word, next_word_likeability


def select_best_word(word1, word1_likeability, word2, word2_likeability):
    if word1_likeability > word2_likeability:
        return word1, word1_likeability
    elif word1_likeability < word2_likeability:
        return word2, word2_likeability
    else:
        if word1 == word2:
            print("Les mots sont identiques, on garde le premier")
        else:
            print("Les mots sont équivalents, on garde le premier")
        return word1, word1_likeability
