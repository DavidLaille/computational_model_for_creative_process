
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


def get_words_to_remove(model):
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


def get_words_to_remove1():
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
    ponctuation = ['</s>', '#', '*', '-', "'", '\"', '.', ',', ';', ':', '/', '_', '!', '?',
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

    # voir si on prend en compte les expressions régulières
    # cela nécessitera d'importer la librairie "re" et d'ajouter des fonctions pour les traiter
    expressions_regulieres = [r"ne (.) pas", r"ne (.) guère", r"ne (.) plus", r"ne (.) point",
                              r"ne (.) rien", r"ne (.) jamais"
                              ]

    # plurals = find_words.get_plurals(model)

    mots_a_exclure = []
    mots_a_exclure.extend(lettres)
    mots_a_exclure.extend(determinants)
    mots_a_exclure.extend(pronoms)
    mots_a_exclure.extend(conjonctions)
    mots_a_exclure.extend(prepositions)
    mots_a_exclure.extend(ponctuation)
    mots_a_exclure.extend(nombres)
    mots_a_exclure.extend(adverbes)

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
