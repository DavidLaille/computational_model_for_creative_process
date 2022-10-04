<<<<<<< Updated upstream
import random
from classes import State


def change_state(current_state=State('default0', 0, 0), next_possible_state=State('default1', 0, 0)):
    current_state.word = next_possible_state.word
    current_state.adequacy = next_possible_state.adequacy
    current_state.originality = next_possible_state.originality


def policy_respected(policy, current_state, next_possible_state):
    if policy == 1:  # maximisation of adequacy
        if current_state.adequacy < next_possible_state.adequacy:
            return True
    elif policy == 2:  # maximisation of originality
        if current_state.originality < next_possible_state.originality:
            return True
    elif policy == 3:  # maximisation of adequacy AND originality
        if (current_state.adequacy < next_possible_state.adequacy
                and current_state.originality < next_possible_state.originality):
            return True
    elif policy == 4:  # maximisation of adequacy OR originality
        if (current_state.adequacy < next_possible_state.adequacy
                or current_state.originality < next_possible_state.originality):
            return True
    else:
        return False


def choose_close_word(close_words, index=0):
    selected_close_word = close_words[index]
    return selected_close_word


def select_action_randomly(nb_possible_actions=2):
    num_action = random.randrange(1, nb_possible_actions, 1)
    return num_action


def select_action_by_proba(possible_actions, action_probas):
    num_action = random.choices(population=possible_actions, weights=action_probas, k=1)
    return num_action[0]  # as random.choices() return a list we select only the first element of this list


def assign_random_proba():
    p = random.randrange(0, 101, 1)  # return an integer between 0% and 100%
    return p
=======
from operator import index
import random
from classes import State
import numpy as np



def change_state(current_state=State('default0', 0, 0), next_possible_state=State('default1', 0, 0)):
    current_state.word = next_possible_state.word
    current_state.adequacy = next_possible_state.adequacy
    current_state.originality = next_possible_state.originality

# we will change the parameters of this function by using the vectors coming from word2vec
def select_close_words(dico, dico_freq, size_of_semantic_network):
    close_words = []
    for i in range(size_of_semantic_network):
        index_highest_freq = np.argmax(dico_freq)
        close_words.append(dico[index_highest_freq])
        # we delete the word corresponding to the highest associate frequency
        del(dico[index_highest_freq], dico_freq[index_highest_freq])
    
    return close_words

def get_frequency(size_list):
    frequencies = []
    for i in range(size_list):
        frequencies.append(random.randint(0, 101)) # we put a random percentage
    
    return frequencies


def get_adequacy(size_list):
    adequacies = []
    for i in range(size_list):
        adequacies.append(random.randint(0, 101)) # we put a random percentage
    
    return adequacies


def get_originality(size_list):
    originalities = []
    for i in range(size_list):
        originalities.append(random.randint(0, 101)) # we put a random percentage
    
    return originalities


def policy_respected(policy, current_state, next_possible_state):
    if policy == 1:  # maximisation of adequacy
        if current_state.adequacy < next_possible_state.adequacy:
            return True
    elif policy == 2:  # maximisation of originality
        if current_state.originality < next_possible_state.originality:
            return True
    elif policy == 3:  # maximisation of adequacy AND originality
        if (current_state.adequacy < next_possible_state.adequacy
                and current_state.originality < next_possible_state.originality):
            return True
    elif policy == 4:  # maximisation of adequacy OR originality
        if (current_state.adequacy < next_possible_state.adequacy
                or current_state.originality < next_possible_state.originality):
            return True
    else:
        return False


def choose_close_word(close_words, index=0):
    selected_close_word = close_words[index]
    return selected_close_word


def select_action_randomly(nb_possible_actions=2):
    num_action = random.randrange(1, nb_possible_actions, 1)
    return num_action


def select_action_by_proba(possible_actions, action_probas):
    num_action = random.choices(population=possible_actions, weights=action_probas, k=1)
    return num_action[0]  # as random.choices() return a list we select only the first element of this list


def assign_random_proba():
    p = random.randrange(0, 101, 1)  # return an integer between 0% and 100%
    return p
>>>>>>> Stashed changes
