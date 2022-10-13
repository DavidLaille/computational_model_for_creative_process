import csv
import pandas as pd
from gensim.models import KeyedVectors

# Dans le modèle frWac_postag_no_phrase_700_skip_cut50.bin, on a 215 020 mots
# Pour chacun, il est indiqué la catégorie grammaticale par "_d", "_p", "_v"
# pathToModel = "C:/dev/word2vec_pretrained_models/frWac_postag_no_phrase_700_skip_cut50.bin"

# Dans le modèle frWac_no_postag_no_phrase_700_skip_cut50.bin, on a 184 373 mots
# La catégorie grammaticale n'est pas indiquée
pathToModel = "C:/dev/word2vec_pretrained_models/frWac_no_postag_no_phrase_700_skip_cut50.bin"

# chargement du modèle
model = KeyedVectors.load_word2vec_format(pathToModel, binary=True, unicode_errors="ignore")

# chargement des mots-indices depuis le fichier csv
df = pd.read_csv('data/cues.csv', sep=',')
print(df.shape)
close_words = dict()

# N : le nombres de mots voisins qu'on veut obtenir
N = 2
for i, row in df.iterrows():
    word = row["cues"]
    neighbours_current_word = model.most_similar(word, topn=N, restrict_vocab=10000)
    close_words[word] = neighbours_current_word

# cues = ["jardin", "doigt", "vin", "avis", "pierre", "appel"]

# Quelques print() pour vérifier que tout s'est bien passé
print(close_words)

# print(model.most_similar("jardin", topn=N, restrict_vocab=10000))
# print(model.most_similar("doigt", topn=N, restrict_vocab=10000))
# print(model.most_similar("vin", topn=N, restrict_vocab=10000))
# print(model.most_similar("avis", topn=N, restrict_vocab=10000))
# print(model.most_similar("pierre", topn=N, restrict_vocab=10000))
# print(model.most_similar("appel", topn=N, restrict_vocab=10000))


# # ouverture en écriture (w, première lettre de write) d'un fichier
# with open('semantic_networks.csv', 'w', newline='', encoding='utf8') as fichier:
#
#     # on déclare un objet writer
#     writer = csv.writer(fichier)
#
#     # écrire une ligne dans le fichier:
#     writer.writerow(['mot-indice', 'mot-voisin', 'similarite'])
#     # quelques lignes :
#     for index, cue in enumerate(close_words):
#         print(index, " - ", cue)
#         for neighbour in close_words[cue]:
#             word = neighbour[0]
#             # print(word)
#             similarity = neighbour[1]
#             # print(freq)
#             print(f"cue: {cue} - word: {word} - similarity: {similarity}")
#             writer.writerow([cue, word, similarity])
#
