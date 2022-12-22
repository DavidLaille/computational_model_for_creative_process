from gensim.models import KeyedVectors

import functions_v1 as fct
import numpy as np
import find_words as find_words

# Emplacement des modèles word2vec sur Windows et Mac (à modifier si nécessaire)
location_word2vec_models_windows = "C:/dev/word2vec_pretrained_models/"
location_word2vec_models_mac = "/Users/david.laille/dev/word2vec_pretrained_models/"
location_word2vec_models = location_word2vec_models_mac

# Liste des modèles word2vec (sans postag) disponibles
# Modèles lemmatisés issus des sites web français (en .fr)
word2vec_model_name1 = "frWac_no_postag_no_phrase_700_skip_cut50_modified.bin"
word2vec_model_name2 = "frWac_no_postag_no_phrase_500_cbow_cut100_modified.bin"

# Modèles non lemmatisés issus des sites web français (en .fr)
word2vec_model_name3 = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100_modified.bin"
word2vec_model_name4 = "frWac_non_lem_no_postag_no_phrase_500_skip_cut100_modified.bin"

# Modèles lemmatisés issus du Wikipédia français
word2vec_model_name5 = "frWiki_no_postag_no_phrase_500_cbow_cut10_modified.bin"
word2vec_model_name6 = "frWiki_no_postag_no_phrase_700_cbow_cut100_modified.bin"
word2vec_model_name7 = "frWiki_no_postag_no_phrase_1000_skip_cut100_modified.bin"

# Modèles non lemmatisés issus du Wikipédia français
word2vec_model_name8 = "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100_modified.bin"
word2vec_model_name9 = "frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100_modified.bin"

# Modèles élaborés par l'équipe DaSciM (Polytechnique Paris - X)
word2vec_model_name10 = "fr_w2v_web_w5_modified.bin"
word2vec_model_name11 = "fr_w2v_fl_w5_modified.bin"
word2vec_model_name12 = "fr_w2v_web_w20_modified.bin"
word2vec_model_name13 = "fr_w2v_fl_w20_modified.bin"

# Liste des modèles word2vec (avec postag) disponibles
# Modèles lemmatisés issus des sites web français (en .fr)
word2vec_model_postag_name1 = "frWac_postag_no_phrase_700_skip_cut50_modified.bin"

# Autres modèles lemmatisés issus du Wikipédia français
word2vec_model_postag_name2 = "frwiki-20181020.treetag.2__2019-01-24_.s500_w5_skip.word2vec_modified.bin"
word2vec_model_postag_name3 = "frwiki-20181020.treetag.2.ngram-pass2__2019-04_.s500_w5_skip.word2vec_modified.bin"

pathToWord2vecModel = location_word2vec_models + word2vec_model_name1
word2vec_model = KeyedVectors.load_word2vec_format(pathToWord2vecModel, binary=True, unicode_errors="ignore")
print("Modèle word2vec chargé avec succès.")

# # Pour tester quelques fonctions du modèle avant
# print("#######################################################################################################")
# print("Index du mot avis : ", word2vec_model.get_index("avis"))
# print("Mot similaires à avis : ", word2vec_model.most_similar("avis"))
# print("#######################################################################################################")

# On affiche les caractéristiques du modèle word2vec qu'on vient de charger
print("10 premiers mots du modèle avant suppression : ", word2vec_model.index_to_key[0:10])
print("Longueur du tableau de vecteurs avant suppression des mots inutiles : ", len(word2vec_model.vectors))
print("Longueur de l'index1 avant : ", len(word2vec_model.index_to_key))
print("Longueur de l'index2 avant : ", len(word2vec_model.key_to_index))
# print("Longueur de 'expandos' : ", len(word2vec_model.expandos))
# print("Taille des vecteurs avant : ", word2vec_model.vector_size)
# print("'expandos' du modèle word2vec : ", word2vec_model.expandos)
# if word2vec_model.norms is not None:
#     print("Longueur du tableau des normes de vecteurs avant suppression des mots inutiles : ", len(word2vec_model.norms))
print("#######################################################################################################")

# mots_a_exclure = find_words.get_words_to_remove()
mots_a_exclure = find_words.get_words_to_remove()

# on enlève les doublons
# print("Liste de mots à exclure avant suppression des doublons : ", mots_a_exclure)
print("Taille de la liste avant suppression des doublons : ", len(mots_a_exclure))
set_mots_a_exclure = set(mots_a_exclure)  # l'ordre sera perdu
# print("Set de mots à exclure : ", set_mots_a_exclure)
mots_a_exclure = list(set_mots_a_exclure)
# print("Liste de mots à exclure après suppression des doublons : ", mots_a_exclure)
print("Taille de la liste après suppression des doublons : ", len(mots_a_exclure))

print("#######################################################################################################")
print("Début Suppression")
index_to_delete = []
for index, word in enumerate(word2vec_model.index_to_key):
    if word in mots_a_exclure:
        index_to_delete.append(index)
        print(index)
# on inverse l'ordre de la liste des index à supprimer pour les supprimer du dernier au premier
# sinon une fois qu'on a supprimé le premier, on n'accède plus aux bons index
index_to_delete = index_to_delete[::-1]

# Suppression des mots inutiles du dictionnaire key_to_index du modèle
for word in mots_a_exclure:
    if word in word2vec_model.key_to_index.keys():
        word2vec_model.key_to_index.pop(word)
        print(word)

for i in index_to_delete:
    # print("Id de l'élément supprimé : ", i)
    print(f"{i} - word deleted : {word2vec_model.index_to_key[i]}")
    # print(f"Vecteur n°{i} - Mot {word2vec_model.index_to_key[i]} : {word2vec_model.vectors[i]}")

    word2vec_model.vectors = np.delete(word2vec_model.vectors, i, 0)
    del word2vec_model.index_to_key[i]
    if word2vec_model.norms is not None:
        word2vec_model.norms = np.delete(word2vec_model.norms, i, 0)

print("Fin Suppression")
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

# # Pour tester si la modification du modèle est effective
# print("#######################################################################################################")
# print("Index du mot avis : ", word2vec_model.get_index("avis"))
# print("Mot similaires à avis : ", word2vec_model.most_similar("avis"))
# print("#######################################################################################################")

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
newPathToWord2vecModel = pathToWord2vecModel[:-4] + "_2.bin"
print("Path to the new model : ", newPathToWord2vecModel)
pathToVocab = pathToWord2vecModel[:-4] + "_vocab.txt"
print("Path to the vocab of the new model : ", pathToVocab)
word2vec_model.save_word2vec_format(fname=newPathToWord2vecModel, fvocab=pathToVocab, binary=True)
print("Nouveau modèle sauvegardé.")
print("#######################################################################################################")
