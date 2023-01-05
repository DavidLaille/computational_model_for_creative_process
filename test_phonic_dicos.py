import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import random

########################################################################################################################
# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/"
location_word2vec_models = location_word2vec_models_mac

# Liste des modèles word2vec (sans postag) disponibles
# Modèles lemmatisés issus des sites web français (en .fr)
word2vec_model_name1 = "frWac_no_postag_no_phrase_700_skip_cut50_modified_2.bin"
word2vec_model_name2 = "frWac_no_postag_no_phrase_500_cbow_cut100_modified_2.bin"

# Modèles non lemmatisés issus des sites web français (en .fr)
word2vec_model_name3 = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100_modified_2.bin"
word2vec_model_name4 = "frWac_non_lem_no_postag_no_phrase_500_skip_cut100_modified_2.bin"

# Modèles lemmatisés issus du Wikipédia français
word2vec_model_name5 = "frWiki_no_postag_no_phrase_500_cbow_cut10_modified_2.bin"
word2vec_model_name6 = "frWiki_no_postag_no_phrase_700_cbow_cut100_modified_2.bin"
word2vec_model_name7 = "frWiki_no_postag_no_phrase_1000_skip_cut100_modified_2.bin"

# Modèles non lemmatisés issus du Wikipédia français
word2vec_model_name8 = "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100_modified_2.bin"
word2vec_model_name9 = "frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100_modified_2.bin"

# Modèles élaborés par l'équipe DaSciM (Polytechnique Paris - X)
word2vec_model_name10 = "fr_w2v_web_w5_modified_2.bin"
word2vec_model_name11 = "fr_w2v_fl_w5_modified_2.bin"
word2vec_model_name12 = "fr_w2v_web_w20_modified_2.bin"
# word2vec_model_name13 = "originals/fr_w2v_fl_w20.bin"

pathToWord2vecModel = location_word2vec_models + word2vec_model_name10
word2vec_model = KeyedVectors.load_word2vec_format(pathToWord2vecModel, binary=True, unicode_errors="ignore")
print("Modèle word2vec chargé avec succès.")
########################################################################################################################

rimes_r6 = pd.read_csv("dicos/rimes_r6.csv", sep=',', engine='python')
rimes_r5 = pd.read_csv("dicos/rimes_r5.csv", sep=',', engine='python')
rimes_r4 = pd.read_csv("dicos/rimes_r4.csv", sep=',', engine='python')
rimes_r3 = pd.read_csv("dicos/rimes_r3.csv", sep=',', engine='python')
# rimes_r3 = pd.read_csv("dicos/rimes_r3.csv", sep=',', encoding='utf-8', engine='c')
rimes_r2 = pd.read_csv("dicos/rimes_r2.csv", sep=',', engine='python')

meme_debut_r6 = pd.read_csv("dicos/meme_debut_r6.csv", sep=',', engine='python')
meme_debut_r5 = pd.read_csv("dicos/meme_debut_r5.csv", sep=',', engine='python')
meme_debut_r4 = pd.read_csv("dicos/meme_debut_r4.csv", sep=',', engine='python')
meme_debut_r3 = pd.read_csv("dicos/meme_debut_r3.csv", sep=',', engine='python')
# meme_debut_r2 = pd.read_csv("dicos/meme_debut_r2.csv", sep=',', engine='python')

voisins_phonologiques_r1 = pd.read_csv("dicos/voisins_phonologiques_r1.csv", sep=',', engine='python')
voisins_phonologiques_r2 = pd.read_csv("dicos/voisins_phonologiques_r2.csv", sep=',', engine='python')

contenants = pd.read_csv("dicos/contenants.csv", sep=',', engine='python')
contenus = pd.read_csv("dicos/contenus.csv", sep=',', engine='python')

########################################################################################################################
nb_neighbours = 3
met_words = ["avis"]

potential_neighbours = list()
for word in met_words:
    print("Word : ", word)
    if not rimes_r6[rimes_r6['Mot'] == word].empty:
        print("rimes r6 trouvé")
        index = rimes_r6[rimes_r6['Mot'] == word].index[0]
        cols = rimes_r6.columns
        for col in cols:
            similar_word = rimes_r6[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.9))
    elif not rimes_r5[rimes_r5['Mot'] == word].empty:
        print("rimes r5 trouvé")
        index = rimes_r5[rimes_r5['Mot'] == word].index[0]
        cols = rimes_r5.columns
        for col in cols:
            similar_word = rimes_r5[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.85))
    elif not rimes_r4[rimes_r4['Mot'] == word].empty:
        print("rimes r4 trouvé")
        index = rimes_r4[rimes_r4['Mot'] == word].index[0]
        cols = rimes_r4.columns
        for col in cols:
            similar_word = rimes_r4[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.8))
    elif not rimes_r3[rimes_r3['Mot'] == word].empty:
        print("rimes r3 trouvé")
        index = rimes_r3[rimes_r3['Mot'] == word].index[0]
        cols = rimes_r3.columns
        for col in cols:
            similar_word = rimes_r3[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.75))
    elif not rimes_r2[rimes_r2['Mot'] == word].empty:
        print("rimes r2 trouvé")
        index = rimes_r2[rimes_r2['Mot'] == word].index[0]
        cols = rimes_r2.columns
        for col in cols:
            similar_word = rimes_r2[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.7))

    if not meme_debut_r6[meme_debut_r6['Mot'] == word].empty:
        print("meme début r6 trouvé")
        index = meme_debut_r6[meme_debut_r6['Mot'] == word].index[0]
        cols = meme_debut_r6.columns
        for col in cols:
            similar_word = meme_debut_r6[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.9))
    elif not meme_debut_r5[meme_debut_r5['Mot'] == word].empty:
        print("meme début r5 trouvé")
        index = meme_debut_r5[meme_debut_r5['Mot'] == word].index[0]
        cols = meme_debut_r5.columns
        for col in cols:
            similar_word = meme_debut_r5[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.85))
    elif not meme_debut_r4[meme_debut_r4['Mot'] == word].empty:
        print("meme début r4 trouvé")
        index = meme_debut_r4[meme_debut_r4['Mot'] == word].index[0]
        cols = meme_debut_r4.columns
        for col in cols:
            similar_word = meme_debut_r4[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.8))
    elif not meme_debut_r3[meme_debut_r3['Mot'] == word].empty:
        print("meme début r3 trouvé")
        index = meme_debut_r3[meme_debut_r3['Mot'] == word].index[0]
        cols = meme_debut_r3.columns
        for col in cols:
            similar_word = meme_debut_r3[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.75))

    if not voisins_phonologiques_r1[voisins_phonologiques_r1['Mot'] == word].empty:
        print("voisin phonologique r1 trouvé")
        index = voisins_phonologiques_r1[voisins_phonologiques_r1['Mot'] == word].index[0]
        cols = voisins_phonologiques_r1.columns
        for col in cols:
            similar_word = voisins_phonologiques_r1[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.9))

    if not voisins_phonologiques_r2[voisins_phonologiques_r2['Mot'] == word].empty:
        print("voisin phonologique r2 trouvé")
        index = voisins_phonologiques_r2[voisins_phonologiques_r2['Mot'] == word].index[0]
        cols = voisins_phonologiques_r2.columns
        for col in cols:
            similar_word = voisins_phonologiques_r2[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.8))

    if not contenants[contenants['Mot'] == word].empty:
        print("mot contenant trouvé")
        index = contenants[contenants['Mot'] == word].index[0]
        cols = contenants.columns
        for col in cols:
            similar_word = contenants[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.95))

    if not contenus[contenus['Mot'] == word].empty:
        print("mot contenu trouvé")
        index = contenus[contenus['Mot'] == word].index[0]
        cols = contenus.columns
        for col in cols:
            similar_word = contenus[col].loc[index]
            print("Col : ", col)
            print("Mot qui rime : ", similar_word)
            if similar_word != similar_word or similar_word is None:
                break
            elif similar_word in word2vec_model.key_to_index.keys():
                print(f"Add {similar_word} in potential neighbours")
                potential_neighbours.append((similar_word, 0.85))

print("Potential neighbours : ", potential_neighbours)
print("Potential neighbours ordered : ", sorted(potential_neighbours, key=lambda x: (x[1], x[1]), reverse=True))
potential_neighbours = sorted(potential_neighbours, key=lambda x: (x[1], x[1]), reverse=True)

if len(potential_neighbours) > 1:
    most_similar_words = random.choices(potential_neighbours[0:30], k=nb_neighbours)
else:
    most_similar_words = word2vec_model.most_similar(met_words, topn=nb_neighbours)

neighbours = []
similarities = []
for word in most_similar_words:
    neighbours.append(word[0])
    similarities.append(word[1])

print("Neighbours : ", neighbours)
print("Similarities : ", similarities)
