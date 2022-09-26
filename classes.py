class State:
    def __init__(self, word, adequacy, originality):
        self.word = word
        self.adequacy = adequacy
        self.originality = originality
        # un compteur pour voir à quelle étape de l'exploration on en est
        # à mettre à jour à chaque changement d'état
        self.step = 0

