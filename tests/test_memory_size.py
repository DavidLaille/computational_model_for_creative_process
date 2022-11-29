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

# Chargement des mots-indices depuis le fichier csv
df = pd.read_csv('../data/experimental_data/cues.csv', sep=',')
print("Fichier cues.csv chargé avec succès.")

# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/"
location_word2vec_models = location_word2vec_models_mac

pathToModel = location_word2vec_models + "frWac_no_postag_no_phrase_700_skip_cut50_modified.bin"
word2vec_model = fct.get_model(pathToModel)
print("Modèle word2vec chargé avec succès.")

model_type = 2

########################################################################################################################
# Initialisation des paramètres du modèle computationnel
s_impact_on_a = 0.5
s_impact_on_o = 0.5
adequacy_influence = 0.5

initial_goal_value = 1
discounting_rate = 0.07

memory_sizes = np.arange(start=-1, stop=16, step=1)
vocab_size = 10000

nb_neighbours = 5
nb_max_steps = 100
method = 3

alpha = 0.5
gamma = 0.5

nb_try = 10

########################################################################################################################
# Paramètre testé
memory_size = []
# axe des abscisses étendu pour le traçage des graphes
memory_size_extended_axis = []

# Données qu'on souhaite récupérer
# Données numériques
nb_steps = []
similarity_chosen_word = []
likeability_chosen_word = []
final_goal_value = []

# Données textuelles
first_words = []
chosen_words = []

for cue in df['cues']:
    # # Si on veut tester seulement un certain nombre de mots-indice
    # nb_cues = 2
    # if cue == df['cues'][nb_cues]:
    #     break

    # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    memory_size_extended_axis = []

    # Pour chaque mot-indice, on réinitialise les listes de données à récupérer
    nb_steps = []
    similarity_chosen_word = []
    likeability_chosen_word = []
    final_goal_value = []

    first_words = []
    chosen_words = []

    nb_steps_means = []
    similarity_chosen_word_means = []
    likeability_chosen_word_means = []
    final_goal_value_means = []

    nb_steps_max_q_value = []
    similarity_chosen_word_max_q_value = []
    likeability_chosen_word_max_q_value = []
    final_goal_value_max_q_value = []

    first_words_max_q_value = []
    chosen_words_max_q_value = []

    # Sorties du modèle en fonction de memory_size
    for memory_size in memory_sizes:
        nb_steps_sum = 0
        similarity_sum = 0
        likeability_chosen_word_sum = 0
        final_goal_value_sum = 0

        # Initialisation du modèle computationnel
        model = ComputationalModel(word2vec_model=word2vec_model, model_type=model_type,
                                   s_impact_on_a=s_impact_on_a, s_impact_on_o=s_impact_on_o,
                                   adequacy_influence=adequacy_influence,
                                   initial_goal_value=initial_goal_value, discounting_rate=discounting_rate,
                                   memory_size=memory_size, vocab_size=vocab_size,
                                   nb_neighbours=nb_neighbours, nb_max_steps=nb_max_steps, method=method,
                                   alpha=alpha, gamma=gamma)

        paths, all_neighbours_data = model.launch_model(cue=cue, nb_try=nb_try)

        for t in range(nb_try):
            nb_steps.append(paths['nb_steps'][t])
            similarity_chosen_word.append(paths['sim_best_word'][t])
            likeability_chosen_word.append(paths['li_best_word'][t])
            final_goal_value.append(paths['final_goal_value'][t])

            first_words.append(paths['step_1'][t])
            chosen_words.append((paths['best_word'][t]))

            nb_steps_sum += float(paths['nb_steps'][t])
            similarity_sum += float(paths['sim_best_word'][t])
            likeability_chosen_word_sum += float(paths['li_best_word'][t])
            final_goal_value_sum += float(paths['final_goal_value'][t])

            memory_size_extended_axis.append(memory_size)

        nb_steps_means.append(nb_steps_sum / nb_try)
        similarity_chosen_word_means.append(similarity_sum / nb_try)
        likeability_chosen_word_means.append(likeability_chosen_word_sum / nb_try)
        final_goal_value_means.append(final_goal_value_sum / nb_try)

        # print(paths)
        paths_filename = f'dataframes/test_memory_size_{cue}_paths_{discounting_rate}.csv'
        paths.to_csv(paths_filename, index=False, sep=',')

        # print(all_neighbours_data)
        all_neighbours_data_filename = f'dataframes/test_memory_size_{cue}_all_neighbours_data_{discounting_rate}.csv'
        all_neighbours_data.to_csv(all_neighbours_data_filename, index=False, sep=',')

    print("Nb_steps : ", nb_steps)
    print("Similarité du mot choisi : ", similarity_chosen_word)
    print("Agréabilité du mot choisi : ", likeability_chosen_word)
    print("Goal_value finale : ", final_goal_value)

    print("Nb_steps moyen : ", nb_steps_means)
    print("Similarité moyenne du mot choisi : ", similarity_chosen_word_means)
    print("Agréabilité moyenne du mot choisi : ", likeability_chosen_word_means)
    print("Goal_value finale moyenne : ", final_goal_value_means)

    print("First words : ", first_words)
    print("Chosen words : ", chosen_words)

    # Calcul du nombre d'occurrences des 1ers mots sélectionnés par le modèle (First)
    f_words = []
    f_nb_occurrences = []
    for word in first_words:
        if word not in f_words:
            f_words.append(word)
            f_nb_occurrences.append(first_words.count(word))
    print(f"Mots First : {f_words}")
    print(f"Nb_occurrences mots First : {f_nb_occurrences}")
    df_first_words = pd.DataFrame({
        'mots': f_words,
        'nb_occurrences': f_nb_occurrences
    })
    df_first_words = df_first_words.sort_values(by=['nb_occurrences'])

    # Calcul du nombre d'occurrences des mots sélectionnés par le modèle (Distant)
    ch_words = []
    ch_nb_occurrences = []
    for word in chosen_words:
        if word not in ch_words:
            ch_words.append(word)
            ch_nb_occurrences.append(chosen_words.count(word))
    print(f"Mots choisis : {ch_words}")
    print(f"Nb_occurrences mots choisis : {ch_nb_occurrences}")
    df_chosen_words = pd.DataFrame({
        'mots': ch_words,
        'nb_occurrences': ch_nb_occurrences
    })
    df_chosen_words = df_chosen_words.sort_values(by=['nb_occurrences'])

    # Affichage des 1ers mots (First) et des meilleurs mots (Distant) sélectionnés par le modèle
    # avec leur nombre d'occurrences
    height = len(ch_words)/4
    width = len(ch_words)/4
    fig_f_and_ch = plt.figure(figsize=(width, height))
    (ax1, ax2) = fig_f_and_ch.subplots(1, 2)
    fig_f_and_ch.suptitle(f"{cue} - Nb d'occurrences des 1ers mots et des meilleurs mots choisis par le modèle",
                          color='brown', fontsize=14)
    fig_f_and_ch.tight_layout(h_pad=4, w_pad=7)
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)

    ax1.barh(y=df_first_words.mots, width=df_first_words.nb_occurrences)
    ax1.set_title('First words & Nb_occurrences')
    ax1.set(xlabel='nb_occurrences')
    ax2.barh(y=df_chosen_words.mots, width=df_chosen_words.nb_occurrences)
    ax2.set_title('Chosen words & Nb_occurrences')
    ax2.set(xlabel='nb_occurrences')

    # Sauvegarde des figures obtenues
    file_name = f"images/test_memory_size_{cue}_first_and_chosen_words.png"
    print(file_name)
    plt.savefig(file_name)
    # plt.show()

    """
    Affichage des graphes représentant plusieurs variables en fonction du paramètre testé, à savoir :
    - le nombre d'étapes (nb_steps)
    - la similarité du mot sélectionné (similarity_chosen_word)
    - l'agréabilité/likeability du mot sélectionné (likeability_chosen_word)
    - la valeur finale de la "valeur de but" (final_goal_value)
    """
    width = 30
    height = 30
    fig = plt.figure(figsize=(width, height))
    axs = fig.subplots(4, 2)
    fig.suptitle(f'{cue} - Influence de memory_size sur les sorties du modèle',
                 color='brown', fontsize=14)
    fig.tight_layout(h_pad=4, w_pad=4)
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)

    # Number of steps
    axs[0, 0].plot(memory_sizes, nb_steps_means, color='blue')
    axs[0, 0].set_title('Nb_steps X memory_size')
    axs[0, 0].set(xlabel='memory_size', ylabel='nb_steps_means')
    axs[1, 0].scatter(memory_size_extended_axis, nb_steps, color='blue', marker='+')
    axs[1, 0].set_title('Nb_steps X memory_size')
    axs[1, 0].set(xlabel='memory_sizes', ylabel='nb_steps')

    # Similarity between the cue and the chosen word
    axs[0, 1].plot(memory_sizes, similarity_chosen_word_means, color='orange')
    axs[0, 1].set_title('similarity_chosen_word X memory_size')
    axs[0, 1].set(xlabel='memory_size', ylabel='similarity_chosen_word')
    axs[1, 1].scatter(memory_size_extended_axis, similarity_chosen_word, color='orange', marker='+')
    axs[1, 1].set_title('similarity_chosen_word X memory_size')
    axs[1, 1].set(xlabel='memory_sizes', ylabel='similarity_chosen_word')

    # Likeability between the cue and the chosen word
    axs[2, 0].plot(memory_sizes, likeability_chosen_word_means, color='green')
    axs[2, 0].set_title('likeability_chosen_word X memory_size')
    axs[2, 0].set(xlabel='memory_size', ylabel='likeability_chosen_word_means')
    axs[3, 0].scatter(memory_size_extended_axis, likeability_chosen_word, color='green', marker='+')
    axs[3, 0].set_title('likeability_chosen_word X memory_size')
    axs[3, 0].set(xlabel='memory_sizes', ylabel='likeability_chosen_word')

    axs[2, 1].plot(memory_sizes, final_goal_value_means, color='red')
    axs[2, 1].set_title('final_goal_value X memory_size')
    axs[2, 1].set(xlabel='memory_size', ylabel='final_goal_value_means')
    axs[3, 1].scatter(memory_size_extended_axis, final_goal_value, color='red', marker='+')
    axs[3, 1].set_title('final_goal_value X memory_size')
    axs[3, 1].set(xlabel='memory_sizes', ylabel='final_goal_value')

    # Sauvegarde des figures obtenues
    file_name = f"images/test_memory_size_{cue}_graphs.png"
    print(file_name)
    plt.savefig(file_name)

    plt.close(fig_f_and_ch)
    plt.close(fig)
    # plt.show()
