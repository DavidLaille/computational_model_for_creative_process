import functions_v1 as fct
import pandas as pd


"""
Infos du modèle word2vec pré-entraîné
    nom du modèle   : frWac_no_postag_no_phrase_700_skip_cut50.bin
    nombre de mots  : 184 373 mots
    no_postag       : la catégorie grammaticale n'est pas indiquée (pas de tag '_a', '_n' ou '_v')
    no_phrase       : le modèle ne contient que des mots (pas de phrase ou d'expressions)
    700             : les vecteurs sont de taille 700
    skip            : la méthode utilisée est la méthode skip-gram
    cut50           : seuls les mots qui apparaissaient 50 fois ou plus dans le corpus ont été conservés
"""
pathToModel = "C:/dev/word2vec_pretrained_models/frWac_no_postag_no_phrase_700_skip_cut50_modified.bin"
word2vec_model = fct.get_model(pathToModel)

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('../data/experimental_data/cues.csv', sep=',')

'''
Paramètres du modèle
   Paramètres de mapping
       s_impact_on_a       : représente la proportionnalité entre adéquation et fréquence d'association (similarité)
       s_impact_on_o       : représente la proportionnalité entre originalité et fréquence d'association (similarité)
       adequacy_influence  : représente l'influence de l'adéquation dans le calcul d'agréabilité
                             l'influence de l'originalité est obtenue en faisant : 1 - adequacy_influence

   Paramètres de but
       goal_value          : le seuil de "likeability" (entre 0 et 1) à partir duquel on arrête la recherche
       discounting_rate    : le taux (entre 0 et 1) avec lequel on va réduire la valeur du but à atteindre (goal_value)

   Paramètres de capacité
       memory_size         : le nombre de mots que le modèle gardera en mémoire
                             si memory_size = -1 alors on considère une capacité de mémoire illimitée
       vocab_size          : la taille du lexique dans lequel on pioche les mots qui composeront le réseau
                             les dictionnaires étant triés par fréquence d'occurrence,
                             on pourra éliminer les mots rares en restreignant la taille du lexique

   Paramètres du réseau sémantique
       nb_neighbours       : le nombre de mots voisins qu'on veut obtenir (le nombre de branches issues d'un mot)
       nb_max_steps        : le nombre d'itérations maximal réalisé par le modèle (la profondeur du réseau sémantique)
       method              : la méthode utilisée pour déterminer les mots voisins
                             method = 1 : most_similar() - "distance vectorielle" ou "cosine similarity"
                             method = 2 : most_similar_cosmul() - "multiplicative combination objective"
                                                                   proposed by Omer Levy and Yoav Goldberg

   Paramètres d'entraînement
       nb_try              : nombre de tentatives pour trouver le meilleur chemin
       q-value             : valeur associée à un état (ici un mot)
                             et à l'action réalisée (ici la sélection du mot voisin)
       alpha               : taux d'apprentissage de l'algo de RL (détermine la vitesse d'apprentissage)
       gamma               : taux de prise en compte de la récompense future (détermine dans quelle mesure
                             on prend en compte les états futurs lors du calcul de la valeur)
'''

'''
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
'''

########################################################################################################################
# Initialisation des paramètres du modèle computationnel
s_impact_on_a = 0.5
s_impact_on_o = 0.5
adequacy_influence = 0.5

initial_goal_value = 0.8
discounting_rate = 0.01  # (1%)

memory_size = 7
vocab_size = 10000

nb_neighbours = 5
nb_max_steps = 100
method = 1

alpha = 0.5
gamma = 0.5

nb_try = 10

########################################################################################################################
# Création du dataframe : all_neighbours_data
all_neighbours_data = pd.DataFrame()

# Création du dataframe : neighbours_data
col_names = ['num_path', 'num_step', 'cue', 'best_word', 'q-value', 'current_word', 'neighbours',
             'similarity', 'adequacy', 'originality', 'likeability']
neighbours_data = pd.DataFrame(columns=col_names)

# Création du dataframe : paths
col_names_paths = ['num_path', 'best_word', 'q-value', 'cue']
col_steps = list()
col_similarity = list()
col_likeability = list()
for i in range(nb_max_steps):
    col_steps.append("step_" + str(i+1))  # on démarrera l'indexation à step_1
    col_similarity.append("similarity_" + str(i+1))  # on démarrera l'indexation à similarity_1
    col_likeability.append("likeability_" + str(i+1))  # on démarrera l'indexation à likeability_1
col_names_paths.extend(col_steps)
col_names_paths.extend(col_similarity)
col_names_paths.extend(col_likeability)
paths = pd.DataFrame(columns=col_names_paths)

# visited_words : liste des mots déjà visités (mot-indice + mots sélectionnés)
visited_words = list()
# best_word : mot qui a été sélectionné à la fin
best_word = 'best'

for cue in df['cues']:
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    all_neighbours_data = pd.DataFrame()
    num_path = 0
    for t in range(nb_try):
        # initialisation des variables
        goal_value = 0.8
        current_word = cue
        current_word_likeability = 0
        current_word_similarity = 0
        num_step = 0
        q_value = 0
        # une variable pour représenter le mot final choisi par le modèle
        best_word = current_word
        best_word_likeability = current_word_likeability
        # une liste pour stocker les mots déjà visités, initialisé avec le mot-indice
        words_in_memory = [current_word]
        # on (ré-)initialise la liste des mots visités
        # puis on y ajoute le mot-indice et sa valeur d'agréabilité
        visited_words = list()
        visited_words.append([current_word, current_word_likeability, current_word_similarity])
        while current_word_likeability < goal_value and num_step < nb_max_steps:
            # on récupère les mots voisins, leur fréquence d'association avec le mot-indice
            # puis on récupère les valeurs d'adéquation, d'originalité et d'agréabilité
            neighbours, similarities = fct.get_neighbours_and_similarities(
                words_in_memory, word2vec_model, nb_neighbours, vocab_size, method)
            adequacies, originalities, likeabilities = fct.get_adequacy_originality_and_likeability(
                neighbours, similarities, s_impact_on_a, s_impact_on_o, adequacy_influence)

            # on remplit le dataframe avec les données obtenues
            neighbours_data['neighbours'] = neighbours
            neighbours_data['similarity'] = similarities
            neighbours_data['adequacy'] = adequacies
            neighbours_data['originality'] = originalities
            neighbours_data['likeability'] = likeabilities

            neighbours_data['num_path'] = t+1  # le +1 sert pour démarrer à 1
            neighbours_data['num_step'] = num_step+1  # le +1 sert pour démarrer à 1
            neighbours_data['cue'] = cue
            neighbours_data['best_word'] = best_word
            neighbours_data['q-value'] = q_value
            neighbours_data['current_word'] = current_word
            print("Mots en mémoire : ", words_in_memory)
            # print(neighbours_data)

            # on met toutes les infos des mots proches dans un dataframe global
            all_neighbours_data = pd.concat((all_neighbours_data, neighbours_data), ignore_index=True)

            # on met à jour la q-value
            q_value = fct.update_q_value(current_word_likeability, q_value, goal_value, neighbours_data,
                                         alpha, gamma)

            # on passe du mot-indice au mot-voisin avec la plus grande agréabilité/désirabilité (likeability)
            current_word, current_word_likeability, current_word_similarity = fct.select_next_word(neighbours_data)
            # on ajoute le nouveau mot et sa valeur d'agréabilité dans la liste des mots visités
            visited_words.append([current_word, current_word_likeability, current_word_similarity])

            # on compare les mots pour savoir lequel est le meilleur (ici, meilleur = haute agréabilité)
            best_word, best_word_likeability = fct.select_best_word(best_word, best_word_likeability,
                                                                    current_word, current_word_likeability)

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

            print("####################################################")
            print(f"{num_step} - Mot actuel : {current_word}")
            print(f"q-value : {q_value}")
            print(f"Le mot qui a été choisi est : {best_word}")

            # print("Valeur du but à atteindre avant réduction : ", goal_value)
            goal_value = fct.discount_goal_value(discounting_rate, goal_value)
            # print("Valeur du but à atteindre après réduction : ", goal_value)
            num_step += 1

        # on rajoute une ligne dans le dataframe pour prendre en considération les dernières valeurs obtenues
        all_neighbours_data.loc[len(all_neighbours_data.axes[0]) + 1] = [t + 1, num_step + 1, cue, best_word, q_value,
                                                                         current_word, None, None, None, None, None]

        print("Mots visités : ", visited_words)
        row = [num_path + 1, best_word, q_value, cue]
        for i in range(nb_max_steps):
            if i+1 <= num_step:
                row.append(visited_words[i+1][0])
            else:
                row.append("NA")
        for j in range(nb_max_steps):
            if j+1 <= num_step:
                row.append(visited_words[j+1][1])
            else:
                row.append("NA")
        for k in range(nb_max_steps):
            if k+1 <= num_step:
                row.append(visited_words[k+1][2])
            else:
                row.append("NA")
        print(row)
        paths.loc[num_path] = row

        num_path += 1
        print("#######################################################################################################")

    '''
    Sauvegarde des données dans des fichiers csv
       all_neighbours_data_{cue}.csv   : fichier csv contenant toutes les données des réseaux sémantiques créés 
                                         et parcourus pour un mot-indice donné ("cue")
       paths_{cue}.csv                 : fichier csv contenant tous les chemins parcourus pour un mot-indice donné ("cue")
    '''
    # print(paths)
    paths_filename = f'data/dataframes/paths_{cue}.csv'
    paths.to_csv(paths_filename, index=False, sep=',')

    # print(all_neighbours_data)
    all_neighbours_data_filename = f'data/dataframes/all_neighbours_data_{cue}.csv'
    all_neighbours_data.to_csv(all_neighbours_data_filename, index=False, sep=',')