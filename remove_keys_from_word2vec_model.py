import functions_v1 as fct
import numpy as np


pathToModel = "C:/dev/word2vec_pretrained_models/frWac_no_postag_no_phrase_700_skip_cut50.bin"
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

# LISTE 1 de mots/caractères à exclure (assez complète)
lettres = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
           ]
determinants = ['le', 'la', 'les', "l\'", 'un', 'une', 'des',
                'ce', 'cet', 'cette', 'ces',
                'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'ma', 'ta', 'sa',
                'mes', 'tes', 'ses', 'nos', 'vos', 'leurs',
                'nul', 'chaque', 'quelque', "quelqu\'", 'quelques', 'plusieurs',
                'certain', 'certaine', 'certains', 'certaines', 'divers', 'diverses',
                'quel', 'quelle', 'quels', 'quelles', 'lequel', 'laquelle', 'lesquels', 'lesquelles',
                'desquels', 'desquelles', 'auquel', 'auxquels', 'auxquelles'
                ]
pronoms = ['je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
           'moi', 'toi', 'lui', 'elle', 'nous', 'vous', 'eux', 'elles',
           'me', "m\'", 'te', "t\'", 'se', "s\'", 'nous', 'vous', 'se', "s\'",
           'mien', 'tien', 'sien', 'notre', 'votre', 'leur',
           'mienne', 'tienne', 'sienne', 'notre', 'votre', 'leur',
           'miens', 'tiens', 'siens', 'nôtres', 'vôtres', 'leurs',
           'ce', 'ça', 'ceci', 'cela', 'celui', 'celle', 'ceux',
           'celui-ci', 'celui-là', 'celle-ci', 'celle-là', 'ceux-ci', 'ceux-là',
           'personne', 'chacun', 'tous', 'certains',
           'qui', 'que', 'quoi', 'dont', 'où', 'quiconque',
           'lequel', 'laquelle', 'duquel', 'auquel',
           'lesquels', 'lesquelles', 'desquels', 'desquelles', 'auxquels', 'auxquelles',
           'qui', 'à qui', 'que', "q\'", 'quoi', 'quand', 'comment', 'pourquoi', 'où',
           ]
prepositions = ['à', 'au', 'aux', 'afin', 'dans', 'par', 'parmi', 'pour', 'en', 'vers', 'avec', 'de', 'du', 'y',
                'sans', 'sous', 'sur', 'entre', 'derrière', 'chez', 'de', 'contre',
                'selon', 'via', 'malgré', 'entre', 'hormis', 'hors',
                'à cause de', 'afin de', 'à l’exception de', 'quant à', 'au milieu de',
                'jusque', "jusqu'à"
                ]
prepositions2 = ['à', 'après', 'avant', 'avec', 'chez', 'concernant', 'contre', 'dans', 'de',
                 'depuis', 'derrière', 'dès', 'devant', 'durant', 'en', 'entre', 'envers',
                 'hormis', 'hors', 'jusque', 'malgré', 'moyennant', 'nonobstant', 'outre',
                 'par', 'parmi', 'pendant', 'pour', 'près', 'sans', 'sauf', 'selon', 'sous',
                 'suivant', 'sur', 'touchant', 'vers', 'via'
                 ]
conjonctions = ['mais', 'ou', 'et', 'donc', 'or', 'ni', 'car',
                'que', 'quand', 'comme', 'quoique', 'lorsque', 'puisque', 'si',
                'bien que', 'alors que', 'avant que', 'pour que', 'à condition que',
                'néanmoins', 'toutefois', 'sinon', 'comment', 'pourquoi',
                'ainsi', 'puis', 'dès', 'jusque', 'cependant', 'pourtant', 'enfin', 'alors'
                ]
ponctuation = ['#', '*', '-', "'", '\"', '.', ',', ';', ':', '/', '_', '!', '?',
               '<', '>', '(', ')', '[', ']', '{', '}', '|', '&', '^', '`', '°'
               ]
operations = ['+', '-', '*', '/', '**', '%']
nombres = ['deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'dix',
           'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
           'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
           'cent', 'cents', 'mille', 'million', 'millions', 'milliard', 'milliards'
           ]

adverbes_maniere = ['bien', 'comme', 'mal', 'volontiers', 'à nouveau', 'à tort', 'à tue-tête', 'admirablement',
                    'ainsi', 'aussi', 'bel et bien', 'comment', 'debout', 'également', 'ensemble', 'exprès', 'mal',
                    'mieux', 'plutôt', 'pour de bon', 'presque', 'tant bien que mal', 'vite'
                    ]
adverbes_lieu = ['ici', 'ailleurs', 'alentour', 'après', 'arrière', 'autour', 'avant', 'dedans', 'dehors',
                 'derrière', 'dessous', 'devant', 'là', 'loin', 'où', 'partout', 'près', 'y'
                 ]
adverbes_temps = ['quelquefois', 'parfois', 'autrefois', 'sitôt', 'bientôt', 'aussitôt', 'tantôt', 'alors',
                  'après', 'ensuite', 'enfin', "d'abord", 'tout à coup', 'premièrement', 'soudain', "aujourd'hui",
                  'demain', 'hier', 'auparavant', 'avant', 'cependant', 'déjà', 'demain', 'depuis', 'désormais',
                  'enfin', 'ensuite', 'jadis', 'jamais', 'maintenant', 'puis', 'quand', 'souvent', 'toujours',
                  'tard', 'tôt', 'tout à coup', 'tout de suite', 'longuement'
                  ]
adverbes_quantite = ['quasi', 'davantage', 'plus', 'moins', 'ainsi', 'assez', 'aussi', 'autant', 'beaucoup',
                     'combien', 'encore', 'environ', 'fort', 'guère', 'presque', 'peu', 'si', 'tant', 'tellement',
                     'tout', 'très', 'trop', 'un peu', 'à peu près', 'plus ou moins'
                     ]
adverbes_liaison = ['ainsi', 'aussi', 'pourtant', 'néanmoins', 'toutefois', 'cependant', 'en effet', 'puis',
                    'ensuite', "c'est pourquoi", 'par ailleurs', "d'ailleurs", 'de plus', 'par conséquent'
                    ]
adverbes_affirmation = ['assurément', 'certainement', 'certes', 'oui', 'peut-être', 'précisément',
                        'probablement', 'sans doute', 'volontiers', 'vraiment'
                        ]
adverbes_negation = ['ne', 'pas', 'guère']

adverbes = []
adverbes.extend(adverbes_maniere)
adverbes.extend(adverbes_lieu)
adverbes.extend(adverbes_temps)
adverbes.extend(adverbes_quantite)
adverbes.extend(adverbes_liaison)
adverbes.extend(adverbes_affirmation)
adverbes.extend(adverbes_negation)

# voir si on prend en compte les expressions régulières
# cela nécessitera d'importer la librairie "re" et d'ajouter des fonctions pour les traiter
expressions_regulieres = [r"ne (.) pas", r"ne (.) guère", r"ne (.) plus", r"ne (.) point",
                          r"ne (.) rien", r"ne (.) jamais"
                          ]

mots_a_exclure = []
mots_a_exclure.extend(lettres)
mots_a_exclure.extend(determinants)
mots_a_exclure.extend(pronoms)
mots_a_exclure.extend(conjonctions)
mots_a_exclure.extend(prepositions)
mots_a_exclure.extend(ponctuation)
mots_a_exclure.extend(nombres)
mots_a_exclure.extend(adverbes)

# on enlève les doublons
print("Taille de la liste avant suppression des doublons : ", len(mots_a_exclure))
print("Liste de mots à exclure avant suppression des doublons : ", mots_a_exclure)
set_mots_a_exclure = set(mots_a_exclure)  # l'ordre sera perdu
# print("Set de mots à exclure : ", set_mots_a_exclure)
mots_a_exclure = list(set_mots_a_exclure)
print("Taille de la liste après suppression des doublons : ", len(mots_a_exclure))
print("Liste de mots à exclure après suppression des doublons : ", mots_a_exclure)

# LISTE 2 de mots/caractères à exclure (sans doublons)
words_to_remove = ['</s>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                   'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'ça', 'ces', 'cette', 'cela', 'celui', 'celle',
                   'mais', 'ou', 'et', 'donc', 'or', 'ni', 'car', 'néanmoins', 'si', 'toutefois', 'sinon',
                   'ainsi', 'puis', 'dès', 'jusque', 'cependant', 'pourtant', 'comme', 'lorsque', 'enfin', 'alors',
                   'puisque', 'dont', 'depuis', 'quelque', 'encore', 'chaque',
                   'à', 'au', 'aux', 'afin', 'dans', 'par', 'parmi', 'pour', 'en', 'vers', 'avec', 'de', 'du', 'y',
                   'sans', 'sous', 'sur', 'selon', 'via', 'malgré', 'entre', 'hormis', 'hors',
                   'quel', 'quelle', 'qui', 'que', 'quoi', 'quand', 'comment', 'pourquoi', 'où',
                   'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
                   'moi', 'toi', 'lui', 'eux',
                   'me', 'te', 'ne', 'se', 'leur', 'leurs',
                   'très', 'peu', 'aussi', 'même', 'tout', 'plus', 'aucun',
                   '#', '*', '-', "'",
                   'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'dix',
                   'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
                   'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
                   'cent', 'mille', 'million', 'milliard']

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
# pathToVocab = pathToModel[:-4] + "_modified_vocab.txt"
# model.save_word2vec_format(fname=newPathToModel, fvocab=pathToVocab, binary=True)
print("Nouveau modèle sauvegardé.")
print("#######################################################################################################")
