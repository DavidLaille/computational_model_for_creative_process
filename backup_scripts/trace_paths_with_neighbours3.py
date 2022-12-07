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
        # et le nombre de voisins pris à chaque nouveau pas/saut
        # nb_max_steps = 5
        nb_steps = paths['nb_steps'][index]
        print("Nombre de pas : ", nb_steps)
        # nb_neighbours = 4
        nb_neighbours = int(((len(df['cue']) / len(paths['num_path'])) - 1) / nb_steps)
        print("Nombre de voisins : ", nb_neighbours)

        # cue = df['cue'][index]
        all_network = list()
        all_network.append(cue)
        nodes_label = dict()
        nodes_label[cue] = cue

        # on initialise une liste de mots visités, le mot choisi et les valeurs d'agréabilité
        met_words = []
        met_words_labels = dict()
        met_words_labels[cue] = cue
        likeabilities = [0]

        nodes = []
        edges = []
        weights = []
        weighted_edges = []

        edges_met_words = []
        edges_met_words_style = []
        edge_met_words_colors = []
        edges_met_words_transparency = []

        edges_neighbours = []
        edges_neighbours_style = []
        edge_neighbours_colors = []
        edges_neighbours_transparency = []

        edges_not_selected_words = []
        edges_not_selected_words_style = []
        edge_not_selected_words_colors = []
        edges_not_selected_words_transparency = []

        nodes_size = []
        nodes_color = []
        nodes_transparency = []

        edges_style = []
        edge_colors = []
        edges_transparency = []

        best_word = paths['best_word'][index]
        # print("Best word : ", best_word)

        for id, word in enumerate(df['current_word']):
            word_without_tag = word
            while '_' in word_without_tag:
                word_without_tag = word_without_tag[:-1]
            # print(f"Mot : {word} & Mot sans le tag : {word_without_tag}")
            if df['num_path'][id] == num_path:
                if word_without_tag not in met_words:
                    met_words.append(word_without_tag)
                    met_words_labels[word_without_tag] = word_without_tag
                if df['likeability'][id] == df['likeability'][id]:  # on vérifie que ce n'est pas NaN
                    # si le mot voisin actuel n'est pas enregistré dans le réseau, on l'ajoute
                    if df['neighbours'][id] not in all_network:
                        all_network.append(df['neighbours'][id])
                        nodes_label[df['neighbours'][id]] = df['neighbours'][id]
                        likeabilities.append(df['likeability'][id])
                    # si le lien entre le mot actuel et son voisin n'est pas enregistré, on l'ajoute
                    if (word_without_tag, df['neighbours'][id]) not in edges:
                        edges.append((word_without_tag, df['neighbours'][id]))
                        edges_label[(word_without_tag, df['neighbours'][id])] = round(df['similarity'][id], 3)
                        weights.append(round(df['similarity'][id], 3))  # on arrondit à 3 décimales
                else:
                    best_word = df['best_word'][id]

        nodes = all_network

        print("Mots rencontrés : ", met_words)
        print("Tous les mots du réseau : ", all_network)
        print("Tous les liens dans le réseau : ", edges)
        # print("Poids des liens entre les mots : ", weights)

        # si on veut fixer la position des mots rencontrés
        x_figsize = nb_steps * 2
        y_figsize = nb_neighbours + 2
        x_space_between_words = x_figsize/(len(met_words)+1)
        x_offset = x_figsize/(2*(len(met_words)+1))
        y_offset = y_figsize/10
        y_center = y_figsize/2
        initial_pos = (x_space_between_words, y_center)
        current_pos = initial_pos
        position_met_words = {}
        position_other_words = {}
        position_all_words = {}

        for word in met_words:
            if word == cue:
                position_met_words[word] = initial_pos
                current_pos = initial_pos
                # print(f"Position du mot indice {word} : ", current_pos)
            else:
                position_met_words[word] = (current_pos[0] + x_space_between_words,
                                            y_center)
                current_pos = position_met_words[word]
                # print(f"Position du mot rencontré {word} : ", current_pos)
            for edge in edges:
                index_current_word = met_words.index(word)
                # si on n'est pas sur le dernier mot rencontré
                if index_current_word < len(met_words) - 1:
                    if edge[0] == word and edge[1] == met_words[index_current_word + 1]:
                        # print("edge solid : ", edge)
                        edges_style.append('solid')
                        edge_colors.append('black')
                        edges_transparency.append(1)

                        edges_met_words.append(edge)
                        edges_met_words_style.append('solid')
                        edge_met_words_colors.append('black')
                        edges_met_words_transparency.append(1)
                    elif edge[0] == word and edge[1] in met_words:
                        # print("edge dashed : ", edge)
                        edges_style.append('dashed')
                        edge_colors.append('#8f8f8f')
                        edges_transparency.append(0.5)

                        edges_not_selected_words.append(edge)
                        edges_not_selected_words_style.append('dashed')
                        edge_not_selected_words_colors.append('#8f8f8f')
                        edges_not_selected_words_transparency.append(0.5)
                    elif edge[0] == word:
                        # print("edge dashed : ", edge)
                        edges_style.append('dashed')
                        edge_colors.append('#8f8f8f')
                        edges_transparency.append(0.5)

                        edges_neighbours.append(edge)
                        edges_neighbours_style.append('dashed')
                        edge_neighbours_colors.append('#8f8f8f')
                        edges_neighbours_transparency.append(0.5)

        # print("Style des arêtes : ", edges_style)
        # print("Couleur des arêtes : ", edge_colors)
        # print("Transparence des arêtes : ", edges_transparency)

        num_neighbour = 1
        previous_word = cue
        for word in all_network:
            if word not in met_words:
                found = False
                for edge in edges:
                    if word == edge[1] and not found:
                        if previous_word != edge[0]:
                            previous_word = edge[0]
                            num_neighbour = 1
                        found = True
                        position_other_words[word] = (position_met_words[edge[0]][0] + x_offset,
                                                      position_met_words[edge[0]][1] + num_neighbour * y_offset)
                        # print(f"Position du mot voisin {word} : ", position_other_words[word])
                num_neighbour += 1

        # print("Position des mots rencontrés : ", position_met_words)
        # print("Position des autres mots du réseau : ", position_other_words)

        position_all_words.update(position_met_words)
        position_all_words.update(position_other_words)
        print("Position de tous les mots du réseau : ", position_all_words)

        for node in nodes:
            # si c'est le mot-indice, on le colore en vert-pomme
            if node == paths['cue'][index]:
                nodes_size.append(4000)
                nodes_color.append("#c4ff00")
                nodes_transparency.append(1)
            # si c'est le mot choisi, on le colore en orange-rouge
            elif node == best_word:
                nodes_size.append(5000)
                nodes_color.append("#ff4000")
                nodes_transparency.append(1)
            # si le mot fait partie des mots parcourus, on le colore en orange-pale
            elif node in met_words:
                nodes_size.append(3000)
                nodes_color.append("#ff7728")
                nodes_transparency.append(1)
            # si c'est un mot du réseau sémantique quelconque, on le colore en gris
            # et on le rend un peu transparent
            else:
                nodes_size.append(1500)
                nodes_color.append("#8f8f8f")
                nodes_transparency.append(0.5)

        # print("Taille des noeuds : ", nodes_size)
        # print("Couleur des noeuds : ", nodes_color)
        # print("Transparence des nœuds : ", nodes_transparency)

        bigger_weights = [weight*10 for weight in weights]
        medium_weights = [weight*5 for weight in weights]
        for i, edge in enumerate(edges):
            weighted_edges.append((edges[i][0], edges[i][1], round(bigger_weights[i], 3)))
        print("Mots reliés + Poids des liens entre les mots : ", weighted_edges)

        G.add_nodes_from(nodes)
        # G.add_edges_from(edges)
        G.add_weighted_edges_from(weighted_edges)

        # print("Position des nœuds : ", position_all_words)
        # print("Labels des nœuds : ", nodes_label)

        fig = plt.figure(figsize=(x_figsize, y_figsize))
        nx.draw_networkx_nodes(G, position_all_words, node_size=nodes_size, node_color=nodes_color,
                               alpha=nodes_transparency)
        # nx.draw_networkx_edges(G, position_all_words, edgelist=edges,
        #                        edge_color=edge_colors, width=bigger_weights,
        #                        connectionstyle='arc3, rad=-0.05',
        #                        style=edges_style, alpha=edges_transparency)
        nx.draw_networkx_edges(G, position_all_words, edgelist=edges_met_words,
                               edge_color=edge_met_words_colors, width=bigger_weights,
                               connectionstyle='arc3, rad=0',
                               style=edges_met_words_style, alpha=edges_met_words_transparency)
        nx.draw_networkx_edges(G, position_all_words, edgelist=edges_neighbours,
                               edge_color=edge_neighbours_colors, width=medium_weights,
                               connectionstyle='arc3, rad=-0.05',
                               style=edges_neighbours_style, alpha=edges_neighbours_transparency)
        nx.draw_networkx_edges(G, position_all_words, edgelist=edges_not_selected_words,
                               edge_color=edge_not_selected_words_colors, width=medium_weights,
                               connectionstyle='arc3, rad=0.4',
                               style=edges_not_selected_words_style, alpha=edges_not_selected_words_transparency)
        nx.draw_networkx_labels(G, position_all_words, labels=nodes_label, font_size=10)
        # si on veut ajouter les labels des arêtes
        # nx.draw_networkx_edge_labels(G, position, edges_label, font_size=6)

        # on sauvegarde l'image du chemin parcouru
        file_name = f"data/images/complete_paths/{paths['cue'][index]}_complete_path_{num_path}.png"
        print(file_name)
        print("#######################################################################################################")

        plt.savefig(file_name)
        # si on veut afficher le réseau
        # plt.show()
        # puis on supprime/ferme la figure créée
        plt.close(fig)
