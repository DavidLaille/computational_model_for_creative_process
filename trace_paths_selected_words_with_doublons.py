import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# chargement des mots-indices depuis le fichier csv
cues = pd.read_csv('data/experimental_data/cues.csv', sep=',')

# chargement des données depuis le fichier csv
for cue in cues['cues']:
    # if cue == cues['cues'][2]:
    #     break

    # si on veut tester un seul mot-indice
    word_to_test = "avis"
    if cue != word_to_test:
        continue

    paths = pd.read_csv(f'data/dataframes/paths_{cue}.csv', sep=',', header=0)
    df = pd.read_csv(f'data/dataframes/all_neighbours_data_{cue}.csv', sep=',', header=0)
    # print("paths : ", paths)
    # print("df : ", df)

    nodes = {}
    nodes_label = {}
    edges = {}
    edges_label = {}
    weights = []
    pos = {}

    for index, num_path in enumerate(paths['num_path']):
        G = nx.DiGraph()

        # Calculs pour récupérer le nombre de pas/sauts dans le réseau
        # nb_max_steps = 5
        nb_steps = paths['nb_steps'][index]
        print("Nombre de pas : ", nb_steps)

        # on initialise une liste de mots visités, le mot choisi et les valeurs d'agréabilité
        met_words_with_doublons = [cue]
        met_words_with_doublons_labels = dict()
        met_words_with_doublons_labels[cue] = cue

        likeabilities = [0]

        weights = []
        weighted_edges = []
        edges_met_words_with_doublons = []

        nodes = []
        nodes_size = []
        nodes_color = []

        id_doublon = 1
        for i in range(nb_steps):
            num_col = i+1
            col_name = 'step_' + str(num_col)
            if paths[col_name][index] in met_words_with_doublons:
                word = paths[col_name][index] + '_' + str(id_doublon)
                met_words_with_doublons.append(word)
                met_words_with_doublons_labels[word] = word

                col_name_sim = 'similarity_' + str(num_col)
                weights.append(paths[col_name_sim][index])
                id_doublon += 1
            else:
                met_words_with_doublons.append(paths[col_name][index])
                met_words_with_doublons_labels[paths[col_name][index]] = paths[col_name][index]
                col_name_sim = 'similarity_' + str(num_col)
                weights.append(paths[col_name_sim][index])

        best_word = paths['best_word'][index]
        # print("Best word : ", best_word)

        print("Mots rencontrés avec doublons : ", met_words_with_doublons)
        nodes = met_words_with_doublons

        for num_word, word in enumerate(met_words_with_doublons):
            if num_word < len(met_words_with_doublons)-1:
                edges_met_words_with_doublons.append((word, met_words_with_doublons[num_word+1]))
        print("Tous les liens dans le réseau : ", edges_met_words_with_doublons)
        print("Poids des liens entre les mots : ", weights)
        print("Nombre de poids : ", len(weights))
        print("Nombre de liens : ", len(edges_met_words_with_doublons))

        # si on veut fixer la position des mots rencontrés
        x_figsize = 140
        y_figsize = 80
        x_space_between_words = x_figsize/(len(met_words_with_doublons)+1)
        x_offset = x_figsize/(2*(len(met_words_with_doublons)+1))
        y_center = y_figsize/2
        initial_pos = (x_space_between_words, y_center)
        current_pos = initial_pos
        position_met_words_with_doublons = {}

        for word in met_words_with_doublons:
            if word == cue:
                position_met_words_with_doublons[word] = initial_pos
                current_pos = initial_pos
                # print(f"Position du mot indice {word} : ", current_pos)
            else:
                position_met_words_with_doublons[word] = (current_pos[0] + x_space_between_words,
                                                          y_center)
                current_pos = position_met_words_with_doublons[word]
                # print(f"Position du mot rencontré {word} : ", current_pos)

        print("Position des mots rencontrés : ", position_met_words_with_doublons)

        for node in nodes:
            # si c'est le mot-indice, on le colore en vert-pomme
            if node == paths['cue'][index]:
                nodes_size.append(4000)
                nodes_color.append("#c4ff00")
            # si c'est le mot choisi, on le colore en orange-rouge
            elif node == best_word:
                nodes_size.append(5000)
                nodes_color.append("#ff4000")
            # si le mot fait partie des mots parcourus, on le colore en orange-pale
            elif node in met_words_with_doublons:
                nodes_size.append(3000)
                nodes_color.append("#ff7728")

        # print("Taille des noeuds : ", nodes_size)
        # print("Couleur des noeuds : ", nodes_color)

        bigger_weights = [weight*10 for weight in weights]
        for i, edge in enumerate(edges_met_words_with_doublons):
            weighted_edges.append((edges_met_words_with_doublons[i][0], edges_met_words_with_doublons[i][1], round(bigger_weights[i], 3)))
        print("Mots reliés + Poids des liens entre les mots : ", weighted_edges)

        G.add_nodes_from(nodes)
        # G.add_edges_from(edges_met_words)
        G.add_weighted_edges_from(weighted_edges)

        # print("Position des nœuds : ", position_met_words)
        # print("Labels des nœuds : ", met_words_label)

        fig = plt.figure(figsize=(x_figsize, y_figsize))
        # nx.draw_circular(G, with_labels=True, font_size=8,
        #                  node_color=nodes_color, node_size=nodes_size)
        nx.draw_networkx_nodes(G, position_met_words_with_doublons,
                               node_size=nodes_size, node_color=nodes_color)
        nx.draw_networkx_edges(G, position_met_words_with_doublons,
                               edgelist=edges_met_words_with_doublons, width=bigger_weights)
        nx.draw_networkx_labels(G, position_met_words_with_doublons,
                                labels=met_words_with_doublons_labels, font_size=10)
        # si on veut ajouter les labels des arêtes
        # nx.draw_networkx_edge_labels(G, position, edges_label, font_size=6)

        # on sauvegarde l'image du chemin parcouru
        file_name = f"data/images/paths/{paths['cue'][index]}_path_with_doublons_{num_path}.png"
        print(file_name)
        print("#######################################################################################################")

        plt.savefig(file_name)
        # si on veut afficher le réseau
        # plt.show()
        # puis on supprime/ferme la figure créée
        plt.close(fig)
