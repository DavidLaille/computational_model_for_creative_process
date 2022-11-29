import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions_v1 as fct
from computational_model import ComputationalModel


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
print("Modèle word2vec chargé avec succès.")

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('C:/dev/PycharmProjects/computational_model_for_creative_process/data/experimental_data/cues.csv',
                 sep=',')
print("Fichier cues.csv chargé avec succès.")

########################################################################################################################
# Initialisation des paramètres du modèle computationnel
s_impact_on_a = 0.5
s_impacts_on_o = np.arange(start=0, stop=1, step=0.01)
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

nb_try = 1

########################################################################################################################
# Paramètre testé
s_impact_on_o = []

# Données qu'on souhaite récupérer
nb_steps = []
first_word = []
similarity_chosen_word = []
likeability_chosen_word = []
final_goal_value = []

for cue in df['cues']:
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    # Pour chaque mot-indice, on réinitialise les listes de données à récupérer
    nb_steps = []
    first_word = []
    similarity_chosen_word = []
    likeability_chosen_word = []
    final_goal_value = []

    # Sorties du modèle en fonction de s_impact_on_a
    for s_impact_on_o in s_impacts_on_o:
        # Initialisation du modèle computationnel
        model = ComputationalModel(word2vec_model=word2vec_model,
                                   s_impact_on_a=s_impact_on_a, s_impact_on_o=s_impact_on_o,
                                   adequacy_influence=adequacy_influence,
                                   initial_goal_value=initial_goal_value, discounting_rate=discounting_rate,
                                   memory_size=memory_size, vocab_size=vocab_size,
                                   nb_neighbours=nb_neighbours, nb_max_steps=nb_max_steps, method=method,
                                   alpha=alpha, gamma=gamma)

        paths, all_neighbours_data = model.launch_model(cue=cue, nb_try=nb_try)

        nb_steps.append(paths['nb_steps'][0])
        similarity_chosen_word.append(paths['sim_best_word'][0])
        likeability_chosen_word.append(paths['li_best_word'][0])
        final_goal_value.append(paths['final_goal_value'][0])
        first_word.append(paths['step_1'][0])

    print(nb_steps)
    print(similarity_chosen_word)
    print(likeability_chosen_word)
    print(final_goal_value)
    print(first_word)

    fig = plt.figure(figsize=(10, 6))
    axs = fig.subplots(2, 2)
    fig.suptitle(f'{cue} - Influence de s_impact_on_o sur les sorties du modèle',
                 color='brown', fontsize=14)
    fig.tight_layout(h_pad=4, w_pad=4)
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)

    axs[0, 0].plot(s_impact_on_o, nb_steps)
    axs[0, 0].set_title('Nb_steps X s_impact_on_o')
    axs[0, 0].set(xlabel='s_impact_on_o', ylabel='nb_steps')
    axs[0, 1].plot(s_impact_on_o, similarity_chosen_word, 'tab:orange')
    axs[0, 1].set_title('similarity_chosen_word X s_impact_on_o')
    axs[0, 1].set(xlabel='s_impact_on_o', ylabel='similarity_chosen_word')
    axs[1, 0].plot(s_impact_on_o, likeability_chosen_word, 'tab:green')
    axs[1, 0].set_title('likeability_chosen_word X s_impact_on_o')
    axs[1, 0].set(xlabel='s_impact_on_o', ylabel='likeability_chosen_word')
    axs[1, 1].plot(s_impact_on_o, final_goal_value, 'tab:red')
    axs[1, 1].set_title('final_goal_value X s_impact_on_o')
    axs[1, 1].set(xlabel='s_impact_on_o', ylabel='final_goal_value')

    file_name = f"tests/images/test_s_impact_on_o_{cue}.png"
    print(file_name)

    plt.savefig(file_name)
    plt.close(fig)
    plt.show()

