import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# chargement des mots-indices depuis le fichier csv
cues = pd.read_csv('data/experimental_data/cues.csv', sep=',')

# chargement des données depuis le fichier csv
for cue in cues['cues']:
    # if cue == cues['cues'][2]:
    #     break

    paths = pd.read_csv(f'data/paths_{cue}.csv', sep=',', header=0)
    df = pd.read_csv(f'data/all_neighbours_data_{cue}.csv', sep=',', header=0)

    G = nx.Graph()

    # nb_max_steps = 5
    nb_max_steps = max(df['num_step']) - 1

    nodes_chosen_words = []
    edges_chosen_words = []
    weights_chosen_words = []
    weights = []
    previous_word = None
    for index, num_path in enumerate(paths['num_path']):
        G = nx.Graph()
        met_words = [paths['cue'][index]]
        best_word = paths['best_word'][index]
        similarities = []
        likeabilities = []

        nodes_chosen_words = []
        edges_chosen_words = []
        weights = []
        weighted_edges = []
        nodes_color = ["#e2ff00"]

        previous_word = paths['cue'][index]
        for i in range(nb_max_steps):
            col_name_step = "step_" + str(i+1)
            met_words.append(paths[col_name_step][index])
            col_name_similarity = "similarity_" + str(i+1)
            similarities.append(paths[col_name_similarity][index])
            col_name_likeability = "likeability_" + str(i+1)
            likeabilities.append(paths[col_name_likeability][index])
            if previous_word:
                edges_chosen_words.append((previous_word, paths[col_name_step][index]))

            previous_word = paths[col_name_step][index]
            if previous_word == best_word:
                nodes_color.append("red")
            else:
                nodes_color.append("#ff7728")

        for id, word in enumerate(df['current_word']):
            weights.append(df['similarity'][id])

        nodes_chosen_words = met_words
        weights_chosen_words = similarities

        print("Mots rencontrés : ", met_words)
        # print("Mots sélectionnés reliés : ", edges_chosen_words)
        # print("Poids des liens entre les mots : ", weights_chosen_words)

        for i, edge in enumerate(edges_chosen_words):
            weighted_edges.append((edges_chosen_words[i][0], edges_chosen_words[i][1], weights_chosen_words[i]))
        print("Mots reliés + Poids des liens entre les mots : ", weighted_edges)

        G.add_nodes_from(nodes_chosen_words)
        # G.add_edges_from(edges)
        G.add_weighted_edges_from(weighted_edges)
        position = nx.spring_layout(G)
        nodes_size = [1000]
        nodes_size.extend([i*3000 for i in likeabilities])

        # print("nodes size : ", nodes_size)
        # print("length nodes size : ", len(nodes_size))
        # print("nodes color : ", nodes_color)
        # print("length nodes color : ", len(nodes_color))
        print("edges : ", edges_chosen_words)
        # print("positions : ", position)

        fig = plt.figure(figsize=(10, 6))
        # nx.draw(G, position, with_labels=True, font_size=8,
        #         node_color=nodes_color, node_size=nodes_size)
        nx.draw_circular(G, with_labels=True, font_size=8,
                         node_color=nodes_color, node_size=nodes_size)

        file_name = f"data/images/paths/{paths['cue'][index]}_paths_{num_path}.png"
        print(file_name)
        print("#######################################################################################################")

        # on sauvegarde l'image du chemin parcouru
        plt.savefig(file_name)
        # plt.show()  # pour afficher ensuite
        # puis on supprime/ferme la figure créée
        plt.close(fig)
