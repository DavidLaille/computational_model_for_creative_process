import find_words as find_words
import functions_v1 as fct


pathToModel = "C:/dev/word2vec_pretrained_models/frWac_postag_no_phrase_700_skip_cut50.bin"
model = fct.get_model(pathToModel)

# plurals = find_words.get_plurals(model)
# print("Nombre de mots au pluriel : ", len(plurals))
# print("Mots au pluriel : ", plurals)

# found = model.has_index_for('chiens')
# print("Trouvé ? : ", found)

words_to_remove = find_words.get_words_to_remove(model)
print("Mots à supprimer : ", words_to_remove)
print("Nombre de mots à supprimer : ", len(words_to_remove))

