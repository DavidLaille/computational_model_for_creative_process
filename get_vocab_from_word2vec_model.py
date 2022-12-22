from gensim.models import KeyedVectors

import functions_v1 as fct
import numpy as np
import find_words as find_words

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

pathToWord2vecModel = location_word2vec_models + word2vec_model_name11
word2vec_model = KeyedVectors.load_word2vec_format(pathToWord2vecModel, binary=True, unicode_errors="ignore")
print("Modèle word2vec chargé avec succès.")

print("#######################################################################################################")
# Sauvegarde du nouveau modèle
# En format texte
# newPathToWord2vecModel = pathToWord2vecModel[:-4] + ".txt"
# print("Path to the new model : ", newPathToWord2vecModel)
# word2vec_model.save_word2vec_format(fname=newPathToWord2vecModel, binary=False)

# En créant un fichier texte contenant le vocabulaire
pathToVocab = pathToWord2vecModel[:-4] + "_vocab.txt"
word2vec_model.save_word2vec_format(fname=pathToWord2vecModel, fvocab=pathToVocab, binary=True)
print("Nouveau modèle sauvegardé.")
print("#######################################################################################################")
