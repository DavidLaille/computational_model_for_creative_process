import pandas as pd
# import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# chargement des mots-indices depuis le fichier csv
cues = pd.read_csv('../data/experimental_data/cues.csv', sep=',')

# chargement des données depuis le fichier csv
for cue in cues['cues']:
    if cue == cues['cues'][2]:
        break

    paths = pd.read_csv(f'data/paths_{cue}.csv', sep=',', header=0)
    df = pd.read_csv(f'data/all_neighbours_data_{cue}.csv', sep=',', header=0)
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
        # et le nombre de voisins pris à chaque nouveau pas/saut
        # nb_max_steps = 5
        nb_max_steps = max(df['num_step']) - 1
        print("Nombre de pas : ", nb_max_steps)
        # nb_neighbours = 4
        nb_neighbours = int(((len(df['cue']) / len(paths['num_path'])) - 1) / nb_max_steps)
        print("Nombre de voisins : ", nb_neighbours)

        # cue = df['cue'][index]
        all_network = list()
        all_network.append(cue)
        nodes_label = dict()
        nodes_label[cue] = cue

        # on initialise une liste de mots visités, le mot choisi et les valeurs d'agréabilité
        met_words = [cue]
        met_words_labels = dict()
        met_words_labels[cue] = cue
        best_word = 'best'
        likeabilities = [0]

        nodes = []
        edges = []
        weights = []
        weighted_edges = []

        nodes_size = []
        nodes_color = []
        nodes_transparency = []

        edges_style = []
        edge_colors = []
        edges_transparency = []

        for id, word in enumerate(df['current_word']):
            if df['num_path'][id] == num_path:
                if word not in met_words:
                    met_words.append(word)
                    met_words_labels[word] = word
                if df['likeability'][id] == df['likeability'][id]:  # on vérifie que ce n'est pas NaN
                    # si le mot voisin actuel n'est pas enregistré dans le réseau, on l'ajoute
                    if df['neighbours'][id] not in all_network:
                        all_network.append(df['neighbours'][id])
                        nodes_label[df['neighbours'][id]] = df['neighbours'][id]
                        likeabilities.append(df['likeability'][id])
                    # si le lien entre le mot actuel et son voisin n'est pas enregistré, on l'ajoute
                    if (df['current_word'][id], df['neighbours'][id]) not in edges:
                        edges.append((df['current_word'][id], df['neighbours'][id]))
                        edges_label[(df['current_word'][id], df['neighbours'][id])] = round(df['similarity'][id], 3)
                        weights.append(round(df['similarity'][id], 3))  # on arrondit à 3 décimales
                else:
                    best_word = df['best_word'][id]

        nodes = all_network

        print("Mots rencontrés : ", met_words)
        print("Tous les mots du réseau : ", all_network)
        print("Tous les liens dans le réseau : ", edges)
        print("Poids des liens entre les mots : ", weights)

        # si on veut fixer la position des mots rencontrés
        x_figsize = 14
        y_figsize = 8
        x_space_between_words = x_figsize/(len(met_words)+1)
        x_offset = x_figsize/(2*(len(met_words)+1))
        y_offset = y_figsize/10
        y_center = y_figsize/2
        initial_pos = (x_space_between_words, y_center)
        current_pos = initial_pos
        num_neighbour = 0
        next_met_word = edges[0 + nb_neighbours][0]
        position_met_words = {}
        position_other_words = {}
        position_all_words = {}

        for edge in edges:
            if edge[0] == next_met_word and (edges.index(edge) + nb_neighbours) < len(edges):
                print("next met word 1: ", next_met_word)
                next_met_word = edges[edges.index(edge) + nb_neighbours][0]
                print("next met word 2: ", next_met_word)
                num_neighbour = 0
            elif (edges.index(edge) + nb_neighbours) >= len(edges):
                print("next met word 1: ", next_met_word)
                next_met_word = met_words[-1]
                print("next met word 2: ", next_met_word)
                num_neighbour = 0
            if edge[1] != next_met_word and edge[1] not in met_words and edge[1] not in position_other_words.keys():
                num_neighbour += 1
                print("num_neighbour : ", num_neighbour)

            if edge[0] == cue and edge[0] not in position_met_words.keys():
                position_met_words[cue] = initial_pos
                current_pos = initial_pos
                print(f"Position du mot indice {edge[0]} : ", current_pos)
            elif edge[0] not in position_met_words.keys():
                position_met_words[edge[0]] = (current_pos[0] + x_space_between_words,
                                               y_center)
                current_pos = position_met_words[edge[0]]
                print(f"Position du mot rencontré {edge[0]} : ", current_pos)
            if edge[1] == met_words[-1] and met_words[-2] in position_met_words.keys():
                position_met_words[edge[1]] = (position_met_words[met_words[-2]][0] + x_space_between_words,
                                               y_center)
                print(f"Position du dernier mot rencontré {edge[0]} : ", current_pos)

            if edge[1] not in met_words and edge[1] not in position_other_words.keys():
                position_other_words[edge[1]] = (current_pos[0] + x_offset,
                                                 current_pos[1] + num_neighbour * y_offset)
                print(f"Position du mot voisin {edge[1]} : ", position_other_words[edge[1]])

        print("Position des mots rencontrés : ", position_met_words)
        print("Position des autres mots du réseau : ", position_other_words)

        position_all_words.update(position_met_words)
        position_all_words.update(position_other_words)
        print("Position de tous les mots du réseau : ", position_all_words)

        # mots dont la position sera fixe
        fixed_words = met_words.copy()

        for node in nodes:
            # si c'est le mot-indice, on le colore en vert-pomme
            if node == paths['cue'][index]:
                nodes_size.append(3000)
                nodes_color.append("#c4ff00")
                nodes_transparency.append(1)
            # si c'est le mot choisi, on le colore en orange-rouge
            elif node == best_word:
                nodes_size.append(4000)
                nodes_color.append("#ff4000")
                nodes_transparency.append(1)
            # si le mot fait partie des mots parcourus, on le colore en orange-pale
            elif node in met_words:
                nodes_size.append(2000)
                nodes_color.append("#ff7728")
                nodes_transparency.append(1)
            # si c'est un mot du réseau sémantique quelconque, on le colore en gris
            # et on le rend un peu transparent
            else:
                nodes_size.append(1000)
                nodes_color.append("#8f8f8f")
                nodes_transparency.append(0.5)

        print("Taille des noeuds : ", nodes_size)
        print("Couleur des noeuds : ", nodes_color)
        print("Transparence des noeuds : ", nodes_transparency)

        bigger_weights = [weight*10 for weight in weights]
        for i, edge in enumerate(edges):
            weighted_edges.append((edges[i][0], edges[i][1], round(bigger_weights[i], 3)))
        print("Mots reliés + Poids des liens entre les mots : ", weighted_edges)

        for edge in edges:
            if edge[0] in met_words and edge[1] in met_words:
                edges_style.append('solid')
                edge_colors.append('black')
                edges_transparency.append(1)
                if len(met_words) > 1:
                    met_words.pop(0)
                print(met_words)
            else:
                edges_style.append('dashed')
                edge_colors.append('#8f8f8f')
                edges_transparency.append(0.5)
        print("Style des arêtes : ", edges_style)
        print("Couleur des arêtes : ", edge_colors)
        print("Transparence des arêtes : ", edges_transparency)

        G.add_nodes_from(nodes)
        # G.add_edges_from(edges)
        G.add_weighted_edges_from(weighted_edges)

        # on attribue une position à tous les autres nœuds du réseau
        print("Mots fixés à une certaine position : ", fixed_words)
        position = nx.spring_layout(G, pos=position_met_words)
        print("Position des noeuds : ", position)
        print("Position des noeuds : ", position_all_words)
        print("Labels des noeuds : ", nodes_label)

        fig = plt.figure(figsize=(x_figsize, y_figsize))
        nx.draw_networkx_nodes(G, position_all_words, node_size=nodes_size, node_color=nodes_color,
                               alpha=nodes_transparency)
        nx.draw_networkx_edges(G, position_all_words, edge_color=edge_colors, width=bigger_weights,
                               connectionstyle='arc3, rad=-0.05',
                               style=edges_style, alpha=edges_transparency)
        nx.draw_networkx_labels(G, position_all_words, labels=nodes_label, font_size=10)
        # si on veut ajouter les labels des arêtes
        # nx.draw_networkx_edge_labels(G, position, edges_label, font_size=6)

        # on sauvegarde l'image du chemin parcouru
        file_name = f"data/images/complete_paths/{paths['cue'][index]}_path_{num_path}.png"
        print(file_name)
        plt.savefig(file_name)
        # si on veut afficher le réseau
        # plt.show()
        # puis on supprime/ferme la figure créée
        plt.close(fig)
