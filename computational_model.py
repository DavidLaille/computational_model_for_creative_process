import random
import numpy as np
import math
import pandas as pd

"""
model_type = 0  : meilleur mot temporaire choisi parmi les N voisins actuels (dans son "champs de vision"), 
                  meilleur mot final choisi parmi les N derniers mots découverts
                  propriété : "vue locale sans mémoire"
                  surnom : "le poisson rouge"
                  
model_type = 1  : meilleur mot temporaire choisi parmi ses voisins actuels et le meilleur mot trouvé jusque-là   
                  meilleur mot final choisi parmi les N derniers mots découverts et le meilleur mot trouvé jusque-là
                  propriété : "vue locale avec mémoire" 
                  surnom : "le petit Poucet"
                  
model_type = 2  : meilleur mot temporaire choisi parmi les N voisins actuels (dans son "champs de vision"), 
                  meilleur mot final choisi parmi l'ensemble des mots visités
                  propriété : "vue globale sans mémoire permanente"
                  surnom : "le bateau naviguant dans le brouillard avant l'éclaircie"
                                    
model_type = 3  : meilleur mot temporaire choisi parmi l'ensemble des mots visités jusqu'alors
                  meilleur mot final choisi parmi l'ensemble des mots visités
                  propriété : "vue globale avec mémoire permanente"
                  surnom : "le bateau naviguant par beau temps, vue dégagée"

"""

"""
Paramètres du modèle
    Type de modèle
        word2vec_model          : le modèle word2vec pré-entraîné à charger
        model_type              : le type de modèle qu'on veut utiliser
                                  model_type = 0 : "vue locale sans mémoire"
                                  model_type = 1 : "vue locale avec mémoire" 
                                  model_type = 2 : "vue globale sans mémoire"
                                  model_type = 3 : "vue globale avec mémoire" 
    Paramètres de mapping
        adequacy_influence      : représente l'influence de l'adéquation dans le calcul d'agréabilité
                                  l'influence de l'originalité est obtenue en faisant : 1 - adequacy_influence

    Paramètres de but
        goal_value              : le seuil de "likeability" (entre 0 et 1) à partir duquel on arrête la recherche
        discounting_rate        : le taux (entre 0 et 1) avec lequel on va réduire la valeur du but à atteindre (goal_value)

    Paramètres de capacité
        memory_size             : le nombre de mots que le modèle gardera en mémoire
                                  si memory_size = -1 alors on considère une capacité de mémoire illimitée
        vocab_size              : la taille du lexique dans lequel on pioche les mots qui composeront le réseau
                                  les dictionnaires étant triés par fréquence d'occurrence,
                                  on pourra éliminer les mots rares en restreignant la taille du lexique

    Paramètres du réseau sémantique
        nb_neighbours           : le nombre de mots voisins qu'on veut obtenir (le nombre de branches issues d'un mot)
        method                  : la méthode utilisée pour déterminer les mots voisins
                                  method = 1 : most_similar() - "distance vectorielle" ou "cosine similarity"
                                  method = 2 : most_similar_cosmul() - "multiplicative combination objective"
                                                                   proposed by Omer Levy and Yoav Goldberg

    Paramètres d'apprentissage par renforcement (RL)
         q-value                : valeur associée à un état (ici un mot)
                                  et à l'action réalisée (ici la sélection du mot voisin)
         alpha                  : taux d'apprentissage de l'algo de RL (détermine la vitesse d'apprentissage)
         gamma                  : taux de prise en compte de la récompense future (détermine dans quelle mesure
                                  on prend en compte les états futurs lors du calcul de la valeur)
"""


class ComputationalModel:
    def __init__(self, word2vec_model, model_type=2,
                 adequacy_influence=0.5,
                 initial_goal_value=1, discounting_rate=0.07,
                 memory_size=7, vocab_size=10000,
                 nb_neighbours=5, method=3,
                 alpha=0.5, gamma=0.5):

        self.word2vec_model = word2vec_model
        self.model_type = model_type

        self.adequacy_influence = adequacy_influence

        self.initial_goal_value = initial_goal_value
        self.discounting_rate = discounting_rate  # (1%)

        self.memory_size = memory_size
        self.vocab_size = vocab_size

        self.nb_neighbours = nb_neighbours
        self.method = method

        self.alpha = alpha
        self.gamma = gamma

        self.nb_max_steps = 100

        # Dataframes pour récupérer les résultats
        self.paths = pd.DataFrame()
        self.neighbours_data = pd.DataFrame()
        self.all_neighbours_data = pd.DataFrame()

    def create_dataframes(self):
        """
        Stockage des données obtenues
           all_neighbours_data     : dataframe qui répertorie toutes les données relatives à tous les chemins parcourus
               Colonnes            : | num_path | num_step | cue | best_word | q-value |
                                     | current_word | neighbours | similarity | adequacy | originality | likeability |
           neighbours_data         : dataframe qui répertorie toutes les données relatives à une étape (un step)
               Colonnes            : | num_path | num_step | cue | best_word | q-value |
                                     | current_word | neighbours | similarity | adequacy | originality | likeability |
               Une boucle agrège les données de neighbours_data dans all_neighbours_data

           paths                   : dataframe qui répertorie toutes les données relatives à une étape (un step)
               Colonnes            : | num_path | best_word | q-value | cue |
                                     | step_1 | step_2 | ...
                                     | likeability_1 | likeability_2 | ...
                                     | similarity_1 | similarity_2 | ...
               Lignes              : une ligne par chemin parcouru
        """
        # Création du dataframe : neighbours_data
        col_names = ['num_path', 'num_step', 'cue', 'best_word', 'temporary_best_word', 'q-value', 'current_word', 'neighbours',
                     'similarity', 'adequacy', 'originality', 'likeability', 'likeability_to_cue', 'goal_value']
        self.neighbours_data = pd.DataFrame(columns=col_names)

        # Création du dataframe : paths
        col_names_paths = ['num_path', 'nb_steps',
                           'best_word', 'sim_best_word', 'li_best_word',
                           'final_goal_value', 'q-value', 'cue']
        col_steps = list()
        col_similarity = list()
        col_likeability = list()
        for i in range(self.nb_max_steps):
            col_steps.append("step_" + str(i + 1))  # on démarrera l'indexation à step_1
            col_similarity.append("similarity_" + str(i + 1))  # on démarrera l'indexation à similarity_1
            col_likeability.append("likeability_" + str(i + 1))  # on démarrera l'indexation à likeability_1
        col_names_paths.extend(col_steps)
        col_names_paths.extend(col_similarity)
        col_names_paths.extend(col_likeability)
        self.paths = pd.DataFrame(columns=col_names_paths)

    def launch_model(self, cue, nb_try):
        # initialisation des dataframes pour recueillir les données
        self.create_dataframes()

        num_path = 0

        for t in range(nb_try):
            # initialisation des variables
            likeability_to_cue = 0
            goal_value = self.initial_goal_value
            current_word = cue
            current_word_likeability = 0
            current_word_similarity = 0
            num_step = 0
            q_value = 0
            # une variable pour représenter le mot final choisi par le modèle
            best_word = current_word
            best_word_likeability = current_word_likeability
            temporary_best_word = current_word
            temporary_best_word_likeability = current_word_likeability
            # une liste pour stocker les mots déjà visités, initialisé avec le mot-indice
            words_in_memory = [current_word]
            # on (ré-)initialise la liste des mots visités
            # puis on y ajoute le mot-indice et sa valeur d'agréabilité
            visited_words = list()
            visited_words.append([current_word, current_word_likeability, current_word_similarity])
            neighbours_data_one_path = pd.DataFrame()
            while current_word_likeability < goal_value and num_step < self.nb_max_steps:
                # on récupère les mots voisins, leur fréquence d'association avec le mot-indice
                # puis on récupère les valeurs d'adéquation, d'originalité et d'agréabilité
                neighbours, similarities = get_neighbours_and_similarities(
                    words_in_memory, self.word2vec_model, self.nb_neighbours, self.vocab_size, self.method)
                adequacies, originalities, likeabilities = get_adequacy_originality_and_likeability(
                    neighbours, similarities, self.adequacy_influence)
                likeability_to_cue = get_likeability_to_cue(self.word2vec_model, cue, current_word,
                                                            self.adequacy_influence)

                # on remplit le dataframe avec les données obtenues
                self.neighbours_data['neighbours'] = neighbours
                self.neighbours_data['similarity'] = similarities
                self.neighbours_data['adequacy'] = adequacies
                self.neighbours_data['originality'] = originalities
                self.neighbours_data['likeability'] = likeabilities

                self.neighbours_data['num_path'] = t + 1  # le +1 sert pour démarrer à 1
                self.neighbours_data['num_step'] = num_step
                self.neighbours_data['cue'] = cue
                self.neighbours_data['best_word'] = best_word
                self.neighbours_data['temporary_best_word'] = temporary_best_word
                self.neighbours_data['q-value'] = q_value
                # self.neighbours_data['current_word'] = current_word
                self.neighbours_data['current_word'] = current_word + '_' + str(num_step)
                self.neighbours_data['likeability_to_cue'] = likeability_to_cue
                self.neighbours_data['goal_value'] = goal_value
                # print("Mots en mémoire : ", words_in_memory)
                # print(neighbours_data)

                # on met toutes les infos des mots proches dans deux dataframes globaux
                # neighbours_data_one_path  : récupère les données pour un seul chemin dans le réseau
                # all_neighbours_data       : récupère les données pour tous les chemins dans le réseau
                neighbours_data_one_path = pd.concat((neighbours_data_one_path, self.neighbours_data), ignore_index=True)
                self.all_neighbours_data = pd.concat((self.all_neighbours_data, self.neighbours_data), ignore_index=True)

                # on met à jour la q-value
                q_value = update_q_value(current_word_likeability, q_value, goal_value, self.neighbours_data,
                                         self.alpha, self.gamma)

                # on passe du mot-indice au mot-voisin avec la plus grande agréabilité/désirabilité (likeability)
                current_word, current_word_likeability, current_word_similarity = select_next_word(self.neighbours_data)
                # on ajoute le nouveau mot et sa valeur d'agréabilité dans la liste des mots visités
                visited_words.append([current_word, current_word_likeability, current_word_similarity])

                if self.model_type == 0:
                    temporary_best_word = current_word
                    temporary_best_word_likeability = current_word_likeability
                    best_word = temporary_best_word
                    best_word_likeability = temporary_best_word_likeability
                elif self.model_type == 1:
                    temporary_best_word, temporary_best_word_likeability = select_best_word(temporary_best_word,
                                                                                            temporary_best_word_likeability,
                                                                                            current_word,
                                                                                            current_word_likeability)
                    best_word = temporary_best_word
                    best_word_likeability = temporary_best_word_likeability
                elif self.model_type == 2:
                    temporary_best_word = current_word
                    temporary_best_word_likeability = current_word_likeability
                    best_word = temporary_best_word
                    best_word_likeability = temporary_best_word_likeability
                elif self.model_type == 3:
                    temporary_best_word, temporary_best_word_likeability = select_best_word_among_all_visited_words(
                        neighbours_data_one_path)
                    best_word = temporary_best_word
                    best_word_likeability = temporary_best_word_likeability

                # on ajoute le mot dans la liste des mots en mémoire
                if not ({current_word} & set(words_in_memory)):
                    words_in_memory.append(current_word)

                # si la capacité de mémoire est infinie, on passe cette étape
                if self.memory_size == -1:
                    pass
                # sinon on retire un mot de la liste lorsque la capacité maximale de mémoire est atteinte
                elif len(words_in_memory) > self.memory_size:
                    # on supprime l'élément le plus ancien
                    del words_in_memory[0]

                # print("####################################################")
                # print(f"{num_step} - Mot actuel : {current_word}")
                # print(f"q-value : {q_value}")
                # print(f"Le mot qui a été choisi est : {best_word}")

                # pour la première boucle (celle correspondant au mot-indice), on ne diminue pas la goal_value
                if num_step == 0:
                    pass
                # print("Valeur du but à atteindre avant réduction : ", goal_value)
                else:
                    goal_value = discount_goal_value(self.discounting_rate, goal_value)
                    # print("Valeur du but à atteindre après réduction : ", goal_value)

                num_step += 1

            last_likeability_to_cue = get_likeability_to_cue(self.word2vec_model, cue, current_word,
                                                             self.adequacy_influence)

            neighbours_data_one_path.loc[len(neighbours_data_one_path.axes[0])] = [t + 1, num_step, cue,
                                                                                   best_word, temporary_best_word,
                                                                                   q_value, current_word,
                                                                                   None, None, None, None, None,
                                                                                   last_likeability_to_cue,
                                                                                   goal_value]
            # print(neighbours_data_one_path)

            if self.model_type == 2:
                # à la fin, on choisit le meilleur des mots parmi tous les mots parcourus
                # meilleur mot = celui qui a la plus grande valeur d'agréabilité (likeability) parmi tous les mots parcourus
                best_word, best_word_likeability = select_best_word_among_all_visited_words(neighbours_data_one_path)
            elif self.model_type == 3:
                # à la fin, on choisit le meilleur des mots parmi tous les mots parcourus
                # meilleur mot = celui qui a la plus grande valeur d'agréabilité (likeability) parmi tous les mots parcourus
                best_word, best_word_likeability = select_best_word_among_all_visited_words(neighbours_data_one_path)

            # on rajoute une ligne dans le dataframe pour prendre en considération les dernières valeurs obtenues
            self.all_neighbours_data.loc[len(self.all_neighbours_data.axes[0])] = [t + 1, num_step, cue,
                                                                                   best_word, temporary_best_word,
                                                                                   q_value, current_word,
                                                                                   None, None, None, None, None,
                                                                                   last_likeability_to_cue,
                                                                                   goal_value]

            best_word_similarity = get_similarity_between_words(self.word2vec_model, cue, best_word)

            print("Mots visités : ", visited_words)
            print("Nombre de steps : ", num_step)
            # print("Last Goal value : ", goal_value)
            # print("Final Goal value : ", final_goal_value)
            # print("Neighbours data one path : ", neighbours_data_one_path)
            print("Best word : ", best_word)
            print("Best word likeability : ", best_word_likeability)

            print("Final q-value : ", q_value)

            row = [num_path + 1, num_step,
                   best_word, best_word_similarity, best_word_likeability,
                   goal_value, q_value, cue]
            for i in range(self.nb_max_steps):
                if i + 1 <= num_step:
                    row.append(visited_words[i + 1][0])
                else:
                    row.append("NA")
            for j in range(self.nb_max_steps):
                if j + 1 <= num_step:
                    row.append(visited_words[j + 1][2])
                else:
                    row.append("NA")
            for k in range(self.nb_max_steps):
                if k + 1 <= num_step:
                    row.append(visited_words[k + 1][1])
                else:
                    row.append("NA")
            # print(row)
            self.paths.loc[num_path] = row

            num_path += 1
            print("###################################################################################################")
        return self.paths, self.all_neighbours_data


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
        nb_options = nb_neighbours * 5
        most_similar_words = model.most_similar(cue, topn=nb_options, restrict_vocab=vocab_size)
        most_similar_words = random.choices(most_similar_words, k=nb_neighbours)
    # méthode 4 : identique à la méthode 2 sauf qu'on prend N mots proches
    # et on en sélectionne le nombre désiré (nb_neighbours)
    elif method == 4:
        nb_options = nb_neighbours * 5
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
    muA_3 = 1.2     # coefficient multiplicateur de sim^3
    muA_2 = -1.8    # coefficient multiplicateur de sim^2
    muA_1 = 1       # coefficient multiplicateur de sim^1
    muA_0 = 0.7     # coefficient multiplicateur de sim^0
    adequacy = muA_3 * similarity ** 3 + muA_2 * similarity ** 2 + muA_1 * similarity + muA_0
    adequacy = adequacy + float(random.randint(-100, 100)) / 1000.0  # ajout d'incertitude

    if adequacy < 0:
        adequacy = 0
    elif adequacy > 1:
        adequacy = 1

    return adequacy


def compute_originality(similarity):
    muO_3 = -0.5    # coefficient multiplicateur de sim^3
    muO_2 = 1       # coefficient multiplicateur de sim^2
    muO_1 = -1.3    # coefficient multiplicateur de sim^1
    muO_0 = 0.8     # coefficient multiplicateur de sim^0
    originality = muO_3 * similarity ** 3 + muO_2 * similarity ** 2 + muO_1 * similarity + muO_0
    originality = originality + float(random.randint(-100, 100)) / 1000.0  # ajout d'incertitude

    if originality < 0:
        originality = 0
    elif originality > 1:
        originality = 1

    return originality


def compute_likeability(adequacy, originality, adequacy_influence):
    delta = 0.62
    delta = delta + float(random.randint(-100, 100)) / 1000.0  # ajout d'incertitude

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
    # calcul de la valeur maximale de la likeability
    next_word_likeability = max(neighbours_data['likeability'])

    # détermination de la récompense
    reward = compute_reward(current_word_likeability, next_word_likeability)

    # calcul de la q-value
    new_q_value = current_q_value + alpha * (reward + gamma * (next_word_likeability - goal_value))

    return new_q_value


def compute_reward(current_word_likeability, next_word_likeability):
    reward = 0
    if current_word_likeability < next_word_likeability:
        reward = 1
    elif current_word_likeability > next_word_likeability:
        reward = -1

    return reward


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
