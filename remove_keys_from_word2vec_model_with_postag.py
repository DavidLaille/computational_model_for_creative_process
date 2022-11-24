import functions_v1 as fct
import numpy as np
import get_words2remove as find_words


# Modèles frWac - corpus utilisés : tous les sites en .fr
pathToModel = "C:/dev/word2vec_pretrained_models/frWac_postag_no_phrase_700_skip_cut50_modified.bin"
# pathToModel = "C:/dev/word2vec_pretrained_models/frWac_postag_no_phrase_1000_skip_cut100.bin"

model = fct.get_model(pathToModel)

# # Pour tester quelques fonctions du modèle avant
# print("#######################################################################################################")
# print("Index du mot avis : ", model.get_index("avis"))
# print("Mot similaires à avis : ", model.most_similar("avis"))
# print("#######################################################################################################")

# On affiche les caractéristiques du modèle word2vec qu'on vient de charger
print("10 premiers mots du modèle avant suppression : ", model.index_to_key[0:10])
# complete_dico = fct.create_dico(model)
# print("Longueur du lexique du modèle avant suppression des mots inutiles : ", len(complete_dico))
print("Longueur du tableau de vecteurs avant suppression des mots inutiles : ", len(model.vectors))
print("Longueur de l'index1 avant : ", len(model.index_to_key))
print("Longueur de l'index2 avant : ", len(model.key_to_index))
# print("Taille des vecteurs avant : ", model.vector_size)
# if model.norms is not None:
#     print("Longueur du tableau des normes de vecteurs avant suppression des mots inutiles : ", len(model.norms))
print("#######################################################################################################")

mots_a_exclure = find_words.get_words_to_remove(model)

# on enlève les doublons
print("Taille de la liste avant suppression des doublons : ", len(mots_a_exclure))
print("Liste de mots à exclure avant suppression des doublons : ", mots_a_exclure)
# set_mots_a_exclure = set(mots_a_exclure)  # l'ordre sera perdu
# # print("Set de mots à exclure : ", set_mots_a_exclure)
# mots_a_exclure = list(set_mots_a_exclure)
# print("Taille de la liste après suppression des doublons : ", len(mots_a_exclure))
# print("Liste de mots à exclure après suppression des doublons : ", mots_a_exclure)

print("#######################################################################################################")
print("Début Suppression")
index_to_delete = []
for index, word in enumerate(model.index_to_key):
    if word in mots_a_exclure:
        index_to_delete.append(index)
# on inverse l'ordre de la liste des index à supprimer pour les supprimer du dernier au premier
# sinon une fois qu'on a supprimé le premier, on n'accède plus aux bons index
index_to_delete = index_to_delete[::-1]

# Suppression des mots inutiles du dictionnaire key_to_index du modèle
for word in mots_a_exclure:
    if word in model.key_to_index.keys():
        model.key_to_index.pop(word)

for i in index_to_delete:
    # print("Id de l'élément supprimé : ", i)
    # print(f"{i} - word deleted : {model.index_to_key[i]}")
    # print(f"Vecteur n°{i} - Mot {model.index_to_key[i]} : {model.vectors[i]}")

    model.vectors = np.delete(model.vectors, i, 0)
    del model.index_to_key[i]
    if model.norms is not None:
        model.norms = np.delete(model.norms, i, 0)

print("Fin Suppression")
print("#######################################################################################################")

print("#######################################################################################################")
print("Réindexation des mots du dictionnaire")
# for value in model.key_to_index.values():
#     if value > 10:
#         break
#     print("Index dans le dictionnaire de référence avant réindexation : ", value)
#     print("######################################################################")

for index2, word in enumerate(model.index_to_key):
    model.key_to_index[word] = index2

# for value in model.key_to_index.values():
#     if value > 10:
#         break
#     print("Index dans le dictionnaire de référence après réindexation : ", value)
print("#######################################################################################################")

# # Pour tester si la modification du modèle est effective
# print("#######################################################################################################")
# print("Index du mot avis : ", model.get_index("avis"))
# print("Mot similaires à avis : ", model.most_similar("avis"))
# print("#######################################################################################################")

# On affiche les caractéristiques du nouveau modèle word2vec
print("#######################################################################################################")
print("10 premiers mots du modèle après suppression : ", model.index_to_key[0:10])
# dico_without_stopwords = fct.create_dico(model)
# print("Longueur du lexique du modèle après suppression des mots inutiles : ", len(dico_without_stopwords))
print("Longueur du tableau de vecteurs après suppression des mots inutiles : ", len(model.vectors))
print("Longueur de l'index1 après : ", len(model.index_to_key))
print("Longueur de l'index2 après : ", len(model.key_to_index))
# print("Taille des vecteurs après : ", model.vector_size)
# if model.norms is not None:
#     print("Longueur du tableau des normes de vecteurs après suppression des mots inutiles : ", len(model.norms))
print("#######################################################################################################")

print("#######################################################################################################")
# Sauvegarde du nouveau modèle
# En format binaire
newPathToModel = pathToModel[:-4] + "_modified.bin"
print("Path to the new model : ", newPathToModel)
model.save_word2vec_format(fname=newPathToModel, binary=True)

# En format texte
# newPathToModel = pathToModel[:-4] + "_modified.txt"
# print("Path to the new model : ", newPathToModel)
# model.save_word2vec_format(fname=newPathToModel, binary=False)

# En créant un fichier texte contenant le vocabulaire
pathToVocab = pathToModel[:-4] + "_modified_vocab.txt"
model.save_word2vec_format(fname=newPathToModel, fvocab=pathToVocab, binary=True)
print("Nouveau modèle sauvegardé.")
print("#######################################################################################################")
