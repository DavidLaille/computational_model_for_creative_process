# computational_model_for_creative_process
A repository where you can find some scripts aiming to develop a model simulating the creative process (especially the generation of ideas)

# liste des fonctions
-	get_model
	  charge le modèle à partir d’un chemin donné (librairie gensim) 
-	create_dico
	   renvoie l’ensemble du lexique du modèle pré-entraîné passé en paramètre
-	remove_stopwords
	  supprime les mots inutiles d’un dictionnaire et renvoie une copie du nouveau dictionnaire créé
-	get_neighbours_and_similarities
	  renvoie les N mots les plus proches d’un mot ou d’une liste de mots en utilisant un modèle word2vec pré-entrainé
-	get_random_adequacy_originality_and_likeability
	  pour attribuer/calculer les valeurs d’adéquation, d’originalité et d’agréabilité de manière aléatoire
-	get_adequacy_originality_and_likeability
	  pour attribuer/calculer les valeurs d’adéquation, d’originalité et d’agréabilité selon les formules de l’article d’Alizée (reste à définir les paramètres)
-	update_q_value
	  pour calculer/mettre à jour la q-value
-	select_next_word
	  renvoie prend le mot voisin avec la plus grande agréabilité
-	select_best_word
	  renvoie le mot avec la plus grande agréabilité parmi 2 mots fournis
