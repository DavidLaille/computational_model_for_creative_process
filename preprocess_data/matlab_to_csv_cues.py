import scipy.io as spio
import csv

mat = spio.loadmat('data_matlab/A_22/CreHack_A_22.mat')
subMat = mat["FGAT"]["cues"][0][0]
# print(subMat)
# print(mat["FGAT"]["cues"][0][0][1][0][0])

cues = []
for index, array in enumerate(subMat):
    # print(subMat[index][0][0])
    cues.append(str(subMat[index][0][0]))

print(cues)

# écriture des cues dans un fichier csv
with open('data_csv/cues.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)
    header = ['cues']
    writer.writerow(header)

    for cue in cues:
        writer.writerow([cue])

    # # si on veut inclure un id pour chaque cue
    # header = ['id', 'cues']
    # writer.writerow(header)
    # for i, cue in enumerate(cues):
    #     writer.writerow([i, cue])
