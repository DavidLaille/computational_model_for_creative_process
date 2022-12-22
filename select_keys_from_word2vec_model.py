from gensim.models import KeyedVectors

import numpy as np
import find_words

# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/originals/"
location_word2vec_models = location_word2vec_models_mac

# Liste des modèles word2vec (sans postag) disponibles
# Modèles lemmatisés issus des sites web français (en .fr)
word2vec_model_name1 = "frWac_no_postag_no_phrase_700_skip_cut50.bin"
word2vec_model_name2 = "frWac_no_postag_no_phrase_500_cbow_cut100.bin"

# Modèles non lemmatisés issus des sites web français (en .fr)
word2vec_model_name3 = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
word2vec_model_name4 = "frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin"

# Modèles lemmatisés issus du Wikipédia français
word2vec_model_name5 = "frWiki_no_postag_no_phrase_500_cbow_cut10.bin"
word2vec_model_name6 = "frWiki_no_postag_no_phrase_700_cbow_cut100.bin"
word2vec_model_name7 = "frWiki_no_postag_no_phrase_1000_skip_cut100.bin"

# Modèles non lemmatisés issus du Wikipédia français
word2vec_model_name8 = "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin"
word2vec_model_name9 = "frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100.bin"

# Modèles élaborés par l'équipe DaSciM (Polytechnique Paris - X)
word2vec_model_name10 = "fr_w2v_web_w5.bin"
word2vec_model_name11 = "fr_w2v_fl_w5.bin"
word2vec_model_name12 = "fr_w2v_web_w20.bin"
word2vec_model_name13 = "fr_w2v_fl_w20.bin"

# Liste des modèles word2vec (avec postag) disponibles
# Modèles lemmatisés issus des sites web français (en .fr)
word2vec_model_postag_name1 = "frWac_postag_no_phrase_700_skip_cut50.bin"

# Autres modèles lemmatisés issus du Wikipédia français
word2vec_model_postag_name2 = "frwiki-20181020.treetag.2__2019-01-24_.s500_w5_skip.word2vec.bin"
word2vec_model_postag_name3 = "frwiki-20181020.treetag.2.ngram-pass2__2019-04_.s500_w5_skip.word2vec.bin"

pathToWord2vecModel = location_word2vec_models + word2vec_model_name12
word2vec_model = KeyedVectors.load_word2vec_format(pathToWord2vecModel, binary=True, unicode_errors="ignore")
print("Modèle word2vec chargé avec succès.")

# On affiche les caractéristiques du modèle word2vec qu'on vient de charger
print("10 premiers mots du modèle avant suppression : ", word2vec_model.index_to_key[0:10])
# print("index_to_key du modèle word2vec : ", word2vec_model.index_to_key)
# print("key_to_index du modèle word2vec : ", word2vec_model.key_to_index)
# print("Vecteurs du modèle word2vec : ", word2vec_model.vectors)
# print("'expandos' du modèle word2vec : ", word2vec_model.expandos)
print("Longueur du tableau de vecteurs avant suppression des mots inutiles : ", len(word2vec_model.vectors))
print("Longueur de l'index1 avant : ", len(word2vec_model.index_to_key))
print("Longueur de l'index2 avant : ", len(word2vec_model.key_to_index))
# print("Longueur de 'expandos' : ", len(word2vec_model.expandos))
# print("Taille des vecteurs avant : ", word2vec_model.vector_size)
# if word2vec_model.norms is not None:
#     print("Longueur du tableau des normes de vecteurs avant suppression des mots inutiles : ", len(word2vec_model.norms))

print("#######################################################################################################")

mots_a_garder = find_words.get_words_to_keep(word2vec_model)

print("#######################################################################################################")
print("Début Construction")

new_index_to_key = list()
new_key_to_index = dict()
new_vectors = list()
new_norms = list()

for index, word in enumerate(mots_a_garder):
    word_to_keep = word[0]
    index_word_to_keep = word[1]
    if word_to_keep in word2vec_model.index_to_key and index_word_to_keep == word2vec_model.key_to_index[word_to_keep]:
        if word_to_keep in new_key_to_index.keys():
            print(f"{index} - Le mot : {word} existe déjà dans le lexique.")
        else:
            # print(f"index : {index_word_to_keep} - word : {word}")
            new_index_to_key.append(word_to_keep)
            new_key_to_index[word_to_keep] = index_word_to_keep
            new_vectors.append(word2vec_model.vectors[index_word_to_keep])

# for index, word in enumerate(word2vec_model.index_to_key):
#     index_word = word2vec_model.key_to_index[word]
#     # print(f"index word : {index_word} - word : {word}")
#     if word in mots_a_garder and index_word in index_mots_a_garder:
#         print(f"index : {index_word} - word : {word}")
#         new_index_to_key.append(word)
#         new_key_to_index[word] = index_word
#         new_vectors.append(word2vec_model.vectors[index_word])

# print("index_to_key du nouveau modèle word2vec : ", new_index_to_key[0:20])
# print("key_to_index du nouveau modèle word2vec : ", new_key_to_index)
# print("Vecteurs du nouveau modèle word2vec : ", new_vectors)
print("Taille de la liste new_index_to_key : ", len(new_index_to_key))
print("Taille de la liste new_key_to_index : ", len(new_key_to_index))
print("Taille de la liste des vecteurs du nouveau modèle word2vec : ", len(new_vectors))

word2vec_model.index_to_key = new_index_to_key
word2vec_model.key_to_index = new_key_to_index
word2vec_model.vectors = np.array(new_vectors)
word2vec_model.norms = new_norms

print("Fin Construction")
print("#######################################################################################################")

print("#######################################################################################################")
print("Réindexation des mots du dictionnaire")
# for value in word2vec_model.key_to_index.values():
#     if value > 10:
#         break
#     print("Index dans le dictionnaire de référence avant réindexation : ", value)
#     print("######################################################################")

new_expandos = list()
for index2, word in enumerate(word2vec_model.index_to_key):
    word2vec_model.key_to_index[word] = index2
    new_expandos.append(index2+1)
    # print(f"Reindexation - Mot {word} - index : {word2vec_model.key_to_index[word]}")

# On met dans l'ordre décroissant
new_expandos = new_expandos[::-1]
word2vec_model.expandos['count'] = np.array(new_expandos)
# print("'expandos' du modèle word2vec : ", word2vec_model.expandos)

# for value in word2vec_model.key_to_index.values():
#     if value > 10:
#         break
#     print("Index dans le dictionnaire de référence après réindexation : ", value)
print("#######################################################################################################")

# On affiche les caractéristiques du nouveau modèle word2vec
print("#######################################################################################################")
print("10 premiers mots du modèle après suppression : ", word2vec_model.index_to_key[0:10])
print("Longueur du tableau de vecteurs après suppression des mots inutiles : ", len(word2vec_model.vectors))
print("Longueur de l'index1 après : ", len(word2vec_model.index_to_key))
print("Longueur de l'index2 après : ", len(word2vec_model.key_to_index))
# print("Taille des vecteurs après : ", model.vector_size)
# if word2vec_model.norms is not None:
#     print("Longueur du tableau des normes de vecteurs après suppression des mots inutiles : ", len(word2vec_model.norms))
print("#######################################################################################################")

print("#######################################################################################################")
# Sauvegarde du nouveau modèle
# En format binaire
# newPathToWord2vecModel = pathToWord2vecModel[:-4] + "_modified.bin"
# print("Path to the new model : ", newPathToWord2vecModel)
# word2vec_model.save_word2vec_format(fname=newPathToWord2vecModel, binary=True)

# En format texte
# newPathToWord2vecModel = pathToWord2vecModel[:-4] + "_modified.txt"
# print("Path to the new model : ", newPathToWord2vecModel)
# word2vec_model.save_word2vec_format(fname=newPathToWord2vecModel, binary=False)

# En créant un fichier texte contenant le vocabulaire
newPathToWord2vecModel = pathToWord2vecModel[:-4] + "_modified.bin"
print("Path to the new model : ", newPathToWord2vecModel)
pathToVocab = pathToWord2vecModel[:-4] + "_modified_vocab.txt"
print("Path to the vocab of the new model : ", pathToVocab)
word2vec_model.save_word2vec_format(fname=newPathToWord2vecModel, fvocab=pathToVocab, binary=True)
print("Nouveau modèle sauvegardé.")
print("#######################################################################################################")
