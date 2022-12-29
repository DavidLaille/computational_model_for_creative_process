import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données
paths_filename = "data/generated_data/several_participants/dataframes/all_subjects/paths_all_data_10.csv"
df_paths = pd.read_csv(paths_filename, sep=',')
print("Fichier paths_all_data_10.csv chargé avec succès.")


# On récupère les données sous forme de liste
# Données numériques
nb_steps = df_paths['nb_steps'].tolist()
similarity_chosen_word = df_paths['sim_best_word'].tolist()
likeability_chosen_word = df_paths['li_best_word'].tolist()
final_goal_value = df_paths['final_goal_value'].tolist()

########################################################################################################################
# Calcul des stats descriptives
print(df_paths.describe()['li_best_word'])

nb_steps_mean = np.mean(nb_steps)
similarity_mean = np.mean(similarity_chosen_word)
likeability_mean = np.mean(likeability_chosen_word)
final_goal_value_mean = np.mean(final_goal_value)
print("Mean nb_steps : ", nb_steps_mean)
print("Mean similarity : ", similarity_mean)
print("Mean likeability : ", likeability_mean)
print("Mean final_goal_value : ", final_goal_value_mean)

nb_steps_max = np.max(nb_steps)
# print(nb_steps_max)

nb_steps_unique = list()
nb_steps_nb_occurrences = list()
likeability_means = list()
similarity_means = list()
final_goal_value_means = list()
for i in range(nb_steps_max):
    if i in nb_steps:
        nb_steps_unique.append(i)
        nb_steps_nb_occurrences.append(nb_steps.count(i))

        subset = df_paths[df_paths['nb_steps'] == i]
        likeability_means.append(np.mean(subset['li_best_word']))
        similarity_means.append(np.mean(subset['sim_best_word']))
        final_goal_value_means.append(np.mean(subset['final_goal_value']))

print(f"nb_steps_unique : {nb_steps_unique}")
print(f"nb_steps avec nb d'occurrences : {nb_steps_nb_occurrences}")
print(f"Means likeabilty : {likeability_means}")
print(f"Means similarity : {similarity_means}")
print(f"Means final_GV : {final_goal_value_means}")

########################################################################################################################
# Affichage du graphe Likeability VS Nb_steps
height = 10
width = 10
fig_Li_vs_Nb_steps = plt.figure(figsize=(width, height))
plt.scatter(nb_steps_unique, likeability_means, color='purple', marker='o')
plt.title('likeability_chosen_word X nb_steps')
plt.xlabel('nb_steps')
plt.ylabel('likeability_chosen_word')

# Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
file_name = f"data/generated_data/several_participants/analysis/Li_vs_Nb_steps.png"
print(file_name)
plt.savefig(file_name)
# plt.show()
plt.close(fig_Li_vs_Nb_steps)
########################################################################################################################

########################################################################################################################
# Affichage du graphe Similarity VS Nb_steps
height = 10
width = 10
fig_Sim_vs_Nb_steps = plt.figure(figsize=(width, height))
plt.scatter(nb_steps_unique, similarity_means, color='blue', marker='o')
plt.title('similarity_chosen_word X nb_steps')
plt.xlabel('nb_steps')
plt.ylabel('similarity_chosen_word')

# Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
file_name = f"data/generated_data/several_participants/analysis/Sim_vs_Nb_steps.png"
print(file_name)
plt.savefig(file_name)
# plt.show()
plt.close(fig_Sim_vs_Nb_steps)
########################################################################################################################

########################################################################################################################
# Affichage du graphe Final_GV VS Nb_steps
height = 10
width = 10
fig_GV_vs_Nb_steps = plt.figure(figsize=(width, height))
plt.scatter(nb_steps_unique, final_goal_value_means, color='red', marker='o')
plt.title('final_goal_value X nb_steps')
plt.xlabel('nb_steps')
plt.ylabel('final_goal_value')

# Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
file_name = f"data/generated_data/several_participants/analysis/GV_vs_Nb_steps.png"
print(file_name)
plt.savefig(file_name)
# plt.show()
plt.close(fig_GV_vs_Nb_steps)
########################################################################################################################

########################################################################################################################
# Affichage du graphe Likeability VS Similarity
height = 10
width = 10
fig_Li_vs_Sim = plt.figure(figsize=(width, height))
plt.scatter(similarity_means, likeability_means, color='red', marker='o')
plt.title('Likeability X Similarity')
plt.xlabel('Similarity')
plt.ylabel('Likeability')

# Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
file_name = f"data/generated_data/several_participants/analysis/Li_vs_Sim.png"
print(file_name)
plt.savefig(file_name)
# plt.show()
plt.close(fig_Li_vs_Sim)
########################################################################################################################

# ########################################################################################################################
# # Affichage du graphe Likeability VS Nb_steps
# height = 10
# width = 10
# fig_Li_vs_Nb_steps = plt.figure(figsize=(width, height))
# plt.scatter(nb_steps, likeability_chosen_word, color='purple', marker='o')
# plt.title('likeability_chosen_word X nb_steps')
# plt.xlabel('nb_steps')
# plt.ylabel('likeability_chosen_word')
#
# # Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
# file_name = f"data/generated_data/several_participants/analysis/Li_vs_Nb_steps.png"
# print(file_name)
# plt.savefig(file_name)
# # plt.show()
# plt.close(fig_Li_vs_Nb_steps)
# ########################################################################################################################
#
# ########################################################################################################################
# # Affichage du graphe Similarity VS Nb_steps
# height = 10
# width = 10
# fig_Sim_vs_Nb_steps = plt.figure(figsize=(width, height))
# plt.scatter(nb_steps, similarity_chosen_word, color='blue', marker='o')
# plt.title('similarity_chosen_word X nb_steps')
# plt.xlabel('nb_steps')
# plt.ylabel('similarity_chosen_word')
#
# # Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
# file_name = f"data/generated_data/several_participants/analysis/Sim_vs_Nb_steps.png"
# print(file_name)
# plt.savefig(file_name)
# # plt.show()
# plt.close(fig_Sim_vs_Nb_steps)
# ########################################################################################################################
#
# ########################################################################################################################
# # Affichage du graphe Final_GV VS Nb_steps
# height = 10
# width = 10
# fig_GV_vs_Nb_steps = plt.figure(figsize=(width, height))
# plt.scatter(nb_steps, final_goal_value, color='red', marker='o')
# plt.title('final_goal_value X nb_steps')
# plt.xlabel('nb_steps')
# plt.ylabel('final_goal_value')
#
# # Sauvegarde de la figure obtenue (Likeability_chosen_word VS Nb_steps)
# file_name = f"data/generated_data/several_participants/analysis/GV_vs_Nb_steps.png"
# print(file_name)
# plt.savefig(file_name)
# # plt.show()
# plt.close(fig_GV_vs_Nb_steps)
# ########################################################################################################################
