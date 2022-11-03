from gensim.models import KeyedVectors

# Dans le modèle frWac_no_postag_no_phrase_700_skip_cut50.bin, on a 184 373 mots
# La catégorie grammaticale n'est pas indiquée
pathToModel = "C:/dev/word2vec_pretrained_models/frWac_no_postag_no_phrase_700_skip_cut50.bin"

# chargement du modèle
model = KeyedVectors.load_word2vec_format(pathToModel, binary=True, unicode_errors="ignore")

# Création du dictionnaire/répertoire de mots
complete_dico = []
for index, word in enumerate(model.index_to_key):
    complete_dico.append(word)
# print(complete_dico)

# on prend une sous-partie du dictionnaire entier
sub_dico = complete_dico[:100]
print(sub_dico)
