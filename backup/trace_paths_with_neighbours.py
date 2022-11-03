import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# chargement des mots-indices depuis le fichier csv
cues = pd.read_csv('../data/experimental_data/cues.csv', sep=',')

# chargement des données depuis le fichier csv
for cue in cues['cues']:
    paths = pd.read_csv(f'data/paths_{cue}.csv', sep=',', header=0)
    df = pd.read_csv(f'data/all_neighbours_data_{cue}.csv', sep=',', header=0)

    G = nx.Graph()

    nodes = []
    edges = []
    weights = []

    for index, num_path in enumerate(paths['num_path']):
        all_network = list()
        all_network.append(df['cue'][index])

        # on initialise une liste de mots visités, le mot choisi et les valeurs d'agréabilité
        met_words = [paths['cue'][index]]
        best_word = 'best'
        likeabilities = [0]

        nodes = []
        edges = []
        weights = []
        weighted_edges = []
        nodes_size = []
        nodes_color = []

        for id, word in enumerate(df['current_word']):
            if word not in met_words:
                met_words.append(word)
            if df['likeability'][id] == df['likeability'][id]:  # on vérifie que ce n'est pas NaN
                # si le mot voisin actuel n'est pas enregistré dans le réseau, on l'ajoute
                if df['neighbours'][id] not in all_network:
                    all_network.append(df['neighbours'][id])
                    likeabilities.append(df['likeability'][id])
                # si le lien entre le mot actuel et son voisin n'est pas enregistré, on l'ajoute
                if (df['current_word'][id], df['neighbours'][id]) not in edges:
                    edges.append((df['current_word'][id], df['neighbours'][id]))
                    weights.append(round(df['similarity'][id], 3))   # on arrondit à 3 décimales
            else:
                best_word = df['best_word'][id]

        print("All the words in the network : ", all_network)
        nodes = all_network

        print("Mots rencontrés : ", met_words)
        print("Tous les mots du réseau : ", all_network)
        print("Tous les liens dans le réseau : ", edges)
        print("Poids des liens entre les mots : ", weights)

        for node in nodes:
            # si c'est le mot-indice, on le colore en vert-pomme
            if node == paths['cue'][index]:
                nodes_size.append(3000)
                nodes_color.append("#c4ff00")
            # si c'est le mot choisi, on le colore en orange-rouge
            elif node == best_word:
                nodes_size.append(4000)
                nodes_color.append("#ff4000")
            # si le mot fait partie des mots parcourus, on le colore en orange-pale
            elif node in met_words:
                nodes_size.append(2000)
                nodes_color.append("#ff7728")
            # si c'est un mot du réseau sémantique quelconque, on le colore en gris
            else:
                nodes_size.append(1000)
                nodes_color.append("#8f8f8f")

        bigger_weights = [weight*10 for weight in weights]
        for i, edge in enumerate(edges):
            weighted_edges.append((edges[i][0], edges[i][1], round(bigger_weights[i], 3)))
        print("Mots reliés + Poids des liens entre les mots : ", weighted_edges)

        styles = []
        edge_colors = []
        for edge in edges:
            if edge[0] in met_words and edge[1] in met_words:
                styles.append('solid')
                edge_colors.append('black')
                if len(met_words) > 1:
                    met_words.pop(0)
                print(met_words)
            else:
                styles.append('dashed')
                edge_colors.append('#8f8f8f')
        print(styles)
        print(edge_colors)

        G.add_nodes_from(nodes)
        # G.add_edges_from(edges)
        G.add_weighted_edges_from(weighted_edges)
        position = nx.spring_layout(G)

        fig = plt.figure(figsize=(10, 6))
        nx.draw_networkx(G, position, font_size=8,
                         node_size=nodes_size, node_color=nodes_color,
                         edge_color=edge_colors, width=bigger_weights, style=styles)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, position, edge_labels, font_size=6)
        # on peut aussi essayer avec :
        # nx.draw_random(G), nx.draw_circular(G), nx.draw_spectral(G)

        file_name = f"data/complete_paths/{paths['cue'][index]}_paths_{num_path}.png"
        print(file_name)

        # on sauvegarde l'image du chemin parcouru
        plt.savefig(file_name)
        plt.show()  # pour afficher ensuite
        # puis on supprime/ferme la figure créée
        plt.close(fig)
