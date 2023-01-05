import pandas as pd

def get_plurals(model):
    plural_words = []
    for index, word in enumerate(model.index_to_key):
        # plural_word = word + 's'
        # if model.has_index_for(plural_word):
        #     plural_words.append(plural_word)

        plural_word = word + 'x'
        if model.has_index_for(plural_word):
            plural_words.append(plural_word)

    return plural_words


def get_words_with_postag_to_remove(model):
    words_to_remove = []
    for index, word_with_tag in enumerate(model.index_to_key):
        if '_adv' in word_with_tag:
            words_to_remove.append(word_with_tag)
        elif '_n' in word_with_tag:
            continue
        elif '_a' in word_with_tag:
            continue
        elif '_v' in word_with_tag:
            continue
        else:
            words_to_remove.append(word_with_tag)

    return words_to_remove


def get_words_to_remove():
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
               'chacun', 'tous', 'certains',
               'qui', 'que', 'quoi', 'dont', 'où', 'quiconque',
               'lequel', 'laquelle', 'duquel', 'auquel',
               'lesquels', 'lesquelles', 'desquels', 'desquelles', 'auxquels', 'auxquelles',
               'qui', 'à qui', 'que', "q\'", 'quoi', 'quand', 'comment', 'pourquoi', 'où',
               ]
    conjonctions = ['mais', 'ou', 'et', 'donc', 'or', 'ni', 'car',
                    'que', 'quand', 'comme', 'quoique', 'lorsque', 'puisque', 'si',
                    'bien que', 'alors que', 'avant que', 'pour que', 'à condition que',
                    'néanmoins', 'toutefois', 'sinon', 'comment', 'pourquoi',
                    'ainsi', 'puis', 'dès', 'jusque', 'cependant', 'pourtant', 'enfin', 'alors'
                    ]
    ponctuation = ['</s>', ' ', '#', '+', '*', '-', "'", '\"', '.', ',', ';', ':', '/', '_', '!', '?',
                   '<', '>', '(', ')', '[', ']', '{', '}', '|', '&', '^', '`', '°', '¶',
                   '⶷', '◡', '%',
                   '@', '@@', '@@@', '@@@@', '@@@@@', '@@@@@@', '@@@@@@@', '@@@@@@@@', '@@@@@@@@@', '@@@@@@@@@@',
                   '+', '++', '+++', '++++', '+++++', '++++++', '+++++++', '++++++++', '+++++++++', '++++++++++',
                   '=', '==', '===', '====', '=====', '======', '=======', '========', '=========', '==========',
                   '=============================================',
                   '--', '---', '----', '-----', '------', '-------', '--------', '----------', '----------',
                   '-----------', '------------', '-------------', '--------------', '---------------',
                   '----------------', '-----------------', '------------------', '-------------------',
                   '--------------------', '---------------------', '----------------------', '-----------------------',
                   '------------------------', '-------------------------', '--------------------------',
                   '---------------------------', '----------------------------', '-----------------------------',
                   '------------------------------------------------',
                   '------------------------------------------------------------------------------'
                   ]
    operations = ['+', '-', '*', '/', '**', '%']
    nombres = ['un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'dix',
               'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize', 'dix-sept', 'dix-huit', 'dix-neuf',
               'vingt', 'vingt-et-un', 'vingt-deux', 'vingt-trois', 'vingt-quatre',
               'vingt-cinq', 'vingt-six', 'vingt-sept', 'vingt-huit', 'vingt-neuf',
               'trente', 'trente-et-un', 'trente-deux', 'trente-trois', 'trente-quatre',
               'trente-cinq', 'trente-six', 'trente-sept', 'trente-huit', 'trente-neuf',
               'quarante', 'quarante-et-un', 'quarante-deux', 'quarante-trois', 'quarante-quatre',
               'quarante-cinq', 'quarante-six', 'quarante-sept', 'quarante-huit', 'quarante-neuf',
               'cinquante', 'cinquante-et-un', 'cinquante-deux', 'cinquante-trois', 'cinquante-quatre',
               'cinquante-cinq', 'cinquante-six', 'cinquante-sept', 'cinquante-huit', 'cinquante-neuf',
               'soixante', 'soixante-et-un', 'soixante-deux', 'soixante-trois', 'soixante-quatre',
               'soixante-cinq', 'soixante-six', 'soixante-sept', 'soixante-huit', 'soixante-neuf',
               'soixante-dix', 'soixante-et-onze', 'soixante-douze', 'soixante-treize', 'soixante-quatorze',
               'soixante-quinze', 'soixante-seize', 'soixante-dix-sept', 'soixante-dix-huit', 'soixante-dix-neuf',
               'quatre-vingt', 'quatre-vingt-un', 'quatre-vingt-deux', 'quatre-vingt-trois', 'quatre-vingt-quatre',
               'quatre-vingt-cinq', 'quatre-vingt-six', 'quatre-vingt-sept', 'quatre-vingt-huit', 'quatre-vingt-neuf',
               'quatre-vingt-dix', 'quatre-vingt-onze', 'quatre-vingt-douze', 'quatre-vingt-treize', 'quatre-vingt-quatorze',
               'quatre-vingt-quinze', 'quatre-vingt-seize', 'quatre-vingt-dix-sept',
               'quatre-vingt-dix-huit', 'quatre-vingt-dix-neuf',
               'cent', 'cents', 'mille', 'million', 'millions', 'milliard', 'milliards'
               ]
    # Problème de double-sens : neuf
    nombres_romains = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                       'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxi',
                       'ie', 'iie', 'iiie', 'ive', 've', 'vi', 'viie', 'viiie', 'ixe', 'xe',
                       'xie', 'xiie', 'xiiie', 'xive', 'xve', 'xvie', 'xviie', 'xviiie', 'xixe', 'xxe', 'xxie',
                       'ième', 'iième', 'iiième',
                       'ivème', 'vème', 'vième', 'viième', 'viiième', 'ixème', 'xème', 'xième',
                       'xième', 'xiième', 'xiiième', 'xivème', 'xvème', 'xvième', 'xviième', 'xviiième', 'xixème', 'xxème',
                       'xxième', 'xxiième', 'xxiiième', 'xxivème', 'xxvième',
                       'xixème-xxème'
                       ]
    # Problème de double-sens : vie
    eniemes = ['ème',
               'unième', 'deuxième', 'troisième', 'quatrième', 'cinquième', 'sixième', 'septième', 'huitième', 'neuvième',
               'dixième', 'onzième', 'douzième', 'treizième', 'quatorzième', 'quinzième', 'seizième',
               'dix-septième', 'dix-huitième', 'dix-neuvième',
               'vingtième', 'vingt-et-unième', 'vingt-deuxième', 'vingt-troisième', 'vingt-quatrième',
               'vingt-cinquième', 'vingt-sixième', 'vingt-septième', 'vingt-huitième', 'vingt-neuvième',
               'trentième', 'trente-et-unième', 'trente-deuxième', 'trente-troisième', 'trente-quatrième',
               'trente-cinquième', 'trente-sixième', 'trente-septième', 'trente-huitième', 'trente-neuvième',
               'quarantième', 'quarante-et-unième', 'quarante-deuxième', 'quarante-troisième', 'quarante-quatrième',
               'quarante-cinquième', 'quarante-sixième', 'quarante-septième', 'quarante-huitième', 'quarante-neuvième',
               'cinquantième', 'cinquante-et-unième', 'cinquante-deuxième', 'cinquante-troisième', 'cinquante-quatrième',
               'cinquante-cinquième', 'cinquante-sixième', 'cinquante-septième', 'cinquante-huitième', 'cinquante-neuvième',
               'soixantième', 'soixante-et-unième', 'soixante-deuxième', 'soixante-troisième', 'soixante-quatrième',
               'soixante-cinquième', 'soixante-sixième', 'soixante-septième', 'soixante-huitième', 'soixante-neuvième',
               'soixante-dixième', 'soixante-et-onzième', 'soixante-douzième', 'soixante-treizième', 'soixante-quatorzième',
               'soixante-quinzième', 'soixante-seizième', 'soixante-dix-septième', 'soixante-dix-huitième', 'soixante-dix-neuvième',
               'quatre-vingtième', 'quatre-vingt-unième', 'quatre-vingt-deuxième', 'quatre-vingt-troisième', 'quatre-vingt-quatrième',
               'quatre-vingt-cinquième', 'quatre-vingt-sixième', 'quatre-vingt-septième', 'quatre-vingt-huitième', 'quatre-vingt-neuvième',
               'quatre-vingt-dixième', 'quatre-vingt-onzième', 'quatre-vingt-douzième', 'quatre-vingt-treizième', 'quatre-vingt-quatorzième',
               'quatre-vingt-quinzième', 'quatre-vingt-seizième', 'quatre-vingt-dix-septième', 'quatre-vingt-dix-huitième', 'quatre-vingt-dix-neuvième',
               'centième', 'millième', 'millionième', 'milliardième',
               'antépénultième', 'pénultième'
               ]
    prepositions = ['à', 'au', 'aux', 'afin', 'dans', 'par', 'parmi', 'pour', 'en', 'vers', 'avec', 'de', 'du', 'y',
                    'sans', 'sous', 'sur', 'entre', 'derrière', 'chez', 'de', 'contre',
                    'selon', 'via', 'malgré', 'entre', 'hormis', 'hors',
                    'à cause de', 'afin de', 'à l’exception de', 'quant à', 'au milieu de',
                    'jusque', "jusqu'à", "jusqu'en", "jusqu'au"
                    ]
    prepositions2 = ['à', 'après', 'avant', 'avec', 'chez', 'concernant', 'contre', 'dans', 'de',
                     'depuis', 'derrière', 'dès', 'devant', 'durant', 'en', 'entre', 'envers',
                     'hormis', 'hors', 'jusque', 'malgré', 'moyennant', 'nonobstant', 'outre',
                     'par', 'parmi', 'pendant', 'pour', 'près', 'sans', 'sauf', 'selon', 'sous',
                     'suivant', 'sur', 'touchant', 'vers', 'via'
                     ]
    adverbes_maniere = ['bien', 'comme', 'mal', 'volontiers', 'à nouveau', 'à tort', 'à tue-tête', 'admirablement',
                        'ainsi', 'aussi', 'bel et bien', 'comment', 'debout', 'également', 'ensemble', 'exprès',
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
    adverbes_numeraux = ['bis', 'ter', 'quater', 'quinquies', 'sexies', 'septies', 'octies', 'nonies', 'decies',
                         'undecies', 'duodecies', 'terdecies', 'quaterdecies', 'quindecies',
                         'sexdecies', 'septdecies', 'octodecies', 'novodecies', 'vicies',
                         'unvicies', 'duovicies', 'tervicies', 'quatervicies', 'quinvicies',
                         'sexvicies', 'septvicies', 'octovicies', 'novovicies', 'tricies',
                         'untricies', 'duotricies', 'tertricies', 'quatertricies', 'quintricies',
                         'sextricies', 'septtricies1', 'octotricies', 'novotricies', 'quadragies',
                         'unquadragies', 'duoquadragies', 'terquadragies', 'quaterquadragies', 'quinquadragies',
                         'sexquadragies', 'septquadragies', 'octoquadragies', 'novoquadragies', 'quinquagies',
                         'unquinquagies', 'duoquinquagies', 'terquinquagies', 'quaterquinquagies', 'quinquinquagies',
                         'sexquinquagies', 'septquinquagies', 'octoquinquagies', 'novoquinquagies', 'sexagies',
                         'unsexagies', 'duosexagies', 'tersexagies', 'quatersexagies', 'quinsexagies',
                         'sexsexagies', 'septsexagies', 'octosexagies', 'novosexagies3'
                         ]

    adverbes = []
    adverbes.extend(adverbes_maniere)
    adverbes.extend(adverbes_lieu)
    adverbes.extend(adverbes_temps)
    adverbes.extend(adverbes_quantite)
    adverbes.extend(adverbes_liaison)
    adverbes.extend(adverbes_affirmation)
    adverbes.extend(adverbes_negation)
    adverbes.extend(adverbes_numeraux)

    autres = ['est']
    # voir si on prend en compte les expressions régulières
    # cela nécessitera d'importer la librairie "re" et d'ajouter des fonctions pour les traiter
    expressions_regulieres = [r"ne (.) pas", r"ne (.) guère", r"ne (.) plus", r"ne (.) point",
                              r"ne (.) rien", r"ne (.) jamais"
                              ]

    # plurals = find_words.get_plurals(model)
    pluriels_et_feminins = ['chiens', 'jardins', 'faits-main', 'rayons', 'théorèmes', 'situations-problèmes',
                            'deuxièmes', 'troisièmes', 'quatrièmes', 'cinquièmes', 'sixièmes',
                            'quarantièmes',
                            'malinoise']
    conjugues = ['etirez']
    morceaux_de_mot = ["c'", "d'", "j'", "l'", "m'", "n'", "s'", "t'", "qu'",
                       '-est-à-dire', '-t-elle', 'y-a',
                       '-je', '-tu', '-on', '-ce', '-il', '-elle', '-nous', '-vous'
                       'micro-', 'frigo-', 'gauche-là',
                       'lcpc-info-']
    mots_incorrects = ['levre',
                       'xviiiième', 'deuxème',
                       '+=',
                       'a=', 'b=', 'c=', 'd=', 'e=', 'f=', 'g=', 'h=', 'i=', 'j=', 'k=', 'l=', 'm=', 'n=', 'o=', 'p=',
                       'q=', 'r=', 's=', 't=', 'v=', 'w=', 'x=', 'y=', 'z=',
                       'xb=', 'id=', 'nom=',
                       'align=', 'onclick=', 'oncontextmenu=', 'url=', 'color=', 'valign=', 'quote=', 'datetime=',
                       'hspace=', 'vspace=', 'content=', 'size=', 'body=', 'bgcolor=', 'version=', 'auteurs=',
                       'onmousedown=', 'onmouseup=', 'onmouseout=writetxt',
                       'face=', 'maxlength=', 'lang=', 'pluginpage=', 'method=', 'encoding=', 'diridd=', 'language=',
                       'galleryimg=no', 'task=view', 'align=left', 'align=left', 'body=nous', 'body=commentaire', 'body=l',
                       'sa=showpost', 'toolbar=no', 'menubar=no', 'directories=no', 'status=yes',
                       'color=red', 'color=blue', 'color=green',
                       'action=techneo', 'lang=fr', 'n=droit', 'facsimile=off', 'search=no', 'rub=informations',
                       'charset=iso-', 'language=fr', 'scrollbars=yes', 'resizable=yes',
                       '=s', '=o', '=x',
                       'ntpagecontent+=',
                       '³système']

    mots_a_exclure = []
    mots_a_exclure.extend(lettres)
    mots_a_exclure.extend(determinants)
    mots_a_exclure.extend(pronoms)
    mots_a_exclure.extend(conjonctions)
    mots_a_exclure.extend(ponctuation)
    mots_a_exclure.extend(nombres)
    mots_a_exclure.extend(nombres_romains)
    mots_a_exclure.extend(eniemes)
    mots_a_exclure.extend(prepositions)
    mots_a_exclure.extend(prepositions2)
    mots_a_exclure.extend(adverbes)
    mots_a_exclure.extend(autres)

    # mots_a_exclure.extend(pluriels_et_feminins)
    # mots_a_exclure.extend(conjugues)
    # mots_a_exclure.extend(morceaux_de_mot)
    # mots_a_exclure.extend(mots_incorrects)

    return mots_a_exclure


def get_words_to_remove2():
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
    return words_to_remove


def get_words_to_remove_from_open_lexique(word2vec_model):
    lexique = pd.read_csv('dicos/lexique_nouns_verbs_adjs_3+.csv', sep=',')
    lexique_lemmes = lexique['lemme'].to_list()

    words_to_remove = []
    for index, word in enumerate(word2vec_model.index_to_key):
        if word in lexique_lemmes:
            continue
        else:
            words_to_remove.append(word)

    return words_to_remove


def get_words_and_index_to_keep(word2vec_model):
    lexique = pd.read_csv('dicos/lexique_nouns_verbs_adjs_3+.csv', sep=',')
    lexique_lemmes = lexique['lemme'].to_list()

    words_to_keep = []
    index_words_to_keep = []
    for index, word in enumerate(lexique_lemmes):
        if word in word2vec_model.index_to_key:
            # print(f"Mot actuel (du lexique) : {word} - index : {word2vec_model.key_to_index[word]}")
            words_to_keep.append(word)
            index_words_to_keep.append(word2vec_model.key_to_index[word])

    return words_to_keep, index_words_to_keep


def get_words_to_keep(word2vec_model):
    lexique = pd.read_csv('dicos/lexique_nouns_verbs_adjs_3+.csv', sep=',')
    lexique_lemmes = lexique['lemme'].to_list()
    # print("Lexique : ", lexique_lemmes)

    words_to_keep = []
    for index, word in enumerate(lexique_lemmes):
        if word in word2vec_model.index_to_key:
            # print(f"Mot actuel (du lexique) : {word} - index : {word2vec_model.key_to_index[word]}")
            words_to_keep.append((word, word2vec_model.key_to_index[word]))
    words_to_keep = sorted(words_to_keep, key=lambda x: x[1])

    return words_to_keep
