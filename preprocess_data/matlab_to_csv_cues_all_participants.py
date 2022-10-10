import scipy.io as spio
import csv

# un booléen pour vérifier si le header du fichier csv a déjà été écrit
header_writen = False
# écriture des cues dans un fichier csv
with open('data_csv/all_cues.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)

    for id_participant in range(22, 93):
        data = spio.loadmat(f'data_matlab/A_{id_participant}/CreHack_A_{id_participant}.mat')
        subData = data["FGAT"]["cues"][0][0]

        cues = []
        for index, array in enumerate(subData):
            # print(subData[index][0][0])
            cues.append(str(subData[index][0][0]))

        # print(cues)

        if not header_writen:
            header = ['id_participant', 'cues']
            writer.writerow(header)
            header_writen = True

        for cue in cues:
            writer.writerow([id_participant, cue])



