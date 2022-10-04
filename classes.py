<<<<<<< Updated upstream
class State:
    def __init__(self, word, adequacy, originality):
        self.word = word
        self.adequacy = adequacy
        self.originality = originality
        # un compteur pour voir à quelle étape de l'exploration on en est
        # à mettre à jour à chaque changement d'état
        self.step = 0

=======
class State:
    def __init__(self, word, adequacy, originality):
        self.word = word
        self.adequacy = adequacy
        self.originality = originality
        # self.close_words = ["close_word1", "close_word2", "close_word3"]
        # un compteur pour voir à quelle étape de l'exploration on en est
        # à mettre à jour à chaque changement d'état
        self.step = 0

>>>>>>> Stashed changes
