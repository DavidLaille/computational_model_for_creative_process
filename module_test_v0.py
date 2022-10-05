import functions as fct

cue = 'word1'
close_words = ['close_word1', 'close_word2', 'close_word3',
               'close_word4', 'close_word5', 'close_word6']
selected_close_word = ''

# action : act of choosing a word in a set of semantically close words
# ex: if action n°1 is applied, the close word n°1 will be chosen
possible_actions = []
nb_possible_actions = len(close_words)
num_selected_action = 0

# action_proba : the probability for an action to occur/be selected
action_probas = []

# One action is selected and then the corresponding word is printed
num_selected_action = fct.select_action_randomly(nb_possible_actions)
print(f"Selected action : {num_selected_action}")
selected_close_word = fct.choose_close_word(close_words, num_selected_action-1)
print(f"Selected close word : {selected_close_word}")

# assign a probability to each possible action
for i in range(0, nb_possible_actions):
    possible_actions.append(i+1)
    action_probas.append(fct.assign_random_proba())
    # complement: put the actions and the proba into a dictionnary ?

print(f"Tab of possible actions : {possible_actions}")
print(f"Tab of action probabilities : {action_probas}")
num_selected_action = fct.select_action_by_proba(possible_actions, action_probas)
print(f"Selected action : {num_selected_action}")
selected_close_word = fct.choose_close_word(close_words, num_selected_action-1)
print(selected_close_word)
