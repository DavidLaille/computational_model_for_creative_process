import scipy.io as spio
import csv

# a list for each array of FGAT
cues = []
responses = []
likeability = []
originality = []
adequacy = []
frequencyCreaFlexFirst = []
frequencyCreaFlexDist = []
RT_answer = []
condition = []
unique_creaflexFirst = []
unique_creaflexDist = []
frequency_dictaverf = []
# or a dictionnary with each entry corresponding to a tab
data_csv_keys = ['cues', 'responses', 'likeability', 'originality', 'adequacy',
                 'frequencyCreaFlexFirst', 'frequencyCreaFlexDist', 'RT_answer', 'condition',
                 'unique_creaflexFirst', 'unique_creaflexDist', 'frequency_dictaverf']
data_csv_values = [[], [], [], [], [], [], [], [], [], [], [], []]
data_csv = dict(zip(data_csv_keys, data_csv_values))
print(data_csv)


# un booléen pour vérifier si le header du fichier csv a déjà été écrit
header_writen = False
# écriture des cues dans un fichier csv
with open('data_csv/all_data.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)

    for id_participant in range(22, 93):
        data_matlab = spio.loadmat(f'data_matlab/A_{id_participant}/CreHack_A_{id_participant}.mat')
        FGAT = data_matlab["FGAT"]

        cues = FGAT["cues"][0][0]
        # print(cues)
        responses = FGAT["responses"][0][0]
        # print(responses)
        likeability = FGAT["likeability"][0][0]
        # print(likeability)
        originality = FGAT["originality"][0][0]
        # print(originality)
        adequacy = FGAT["adequacy"][0][0]
        # print(adequacy)
        frequencyCreaFlexFirst = FGAT["frequencyCreaFlexFirst"][0][0]
        # print(frequencyCreaFlexFirst)
        frequencyCreaFlexDist = FGAT["frequencyCreaFlexDist"][0][0]
        # print(frequencyCreaFlexDist)
        RT_answer = FGAT["RT_answer"][0][0]
        # print(RT_answer)
        condition = FGAT["condition"][0][0]
        # print(condition)
        unique_creaflexFirst = FGAT["unique_creaflexFirst"][0][0]
        # print(unique_creaflexFirst)
        unique_creaflexDist = FGAT["unique_creaflexDist"][0][0]
        # print(unique_creaflexDist)
        frequency_dictaverf = FGAT["frequency_dictaverf"][0][0]
        # print(frequency_dictaverf)

        for index, array in enumerate(cues):
            if cues[index][0].size > 0:
                data_csv["cues"].append(str(cues[index][0][0]))
            else:
                data_csv["cues"].append('')
        for index, array in enumerate(responses):
            if responses[index][0].size > 0:
                data_csv["responses"].append(str(responses[index][0][0]))
            else:
                data_csv["responses"].append('')
        for index, array in enumerate(likeability):
            data_csv["likeability"].append(likeability[index][0])
        for index, array in enumerate(originality):
            data_csv["originality"].append(originality[index][0])
        for index, array in enumerate(adequacy):
            data_csv["adequacy"].append(adequacy[index][0])
        for index, array in enumerate(frequencyCreaFlexFirst):
            data_csv["frequencyCreaFlexFirst"].append(frequencyCreaFlexFirst[index][0])
        for index, array in enumerate(frequencyCreaFlexDist):
            data_csv["frequencyCreaFlexDist"].append(frequencyCreaFlexDist[index][0])
        for index, array in enumerate(RT_answer):
            data_csv["RT_answer"].append(RT_answer[index][0])
        for index, array in enumerate(condition):
            data_csv["condition"].append(condition[index][0])
        for index, array in enumerate(unique_creaflexFirst):
            data_csv["unique_creaflexFirst"].append(unique_creaflexFirst[index][0])
        for index, array in enumerate(unique_creaflexDist):
            data_csv["unique_creaflexDist"].append(unique_creaflexDist[index][0])
        for index, array in enumerate(frequency_dictaverf):
            data_csv["frequency_dictaverf"].append(frequency_dictaverf[index][0])

        if not header_writen:
            header = ['id_participant', 'cues', 'responses',
                      'likeability', 'originality', 'adequacy',
                      'frequencyCreaFlexFirst', 'frequencyCreaFlexDist',
                      'RT_answer', 'condition',
                      'unique_creaflexFirst', 'unique_creaflexDist',
                      'frequency_dictaverf'
                      ]
            writer.writerow(header)
            header_writen = True

        for index in range(len(cues)):
            writer.writerow([id_participant,
                             data_csv["cues"][index],
                             data_csv["responses"][index],
                             data_csv["likeability"][index],
                             data_csv["originality"][index],
                             data_csv["adequacy"][index],
                             data_csv["frequencyCreaFlexFirst"][index],
                             data_csv["frequencyCreaFlexDist"][index],
                             data_csv["RT_answer"][index],
                             data_csv["condition"][index],
                             data_csv["unique_creaflexFirst"][index],
                             data_csv["unique_creaflexDist"][index],
                             data_csv["frequency_dictaverf"][index]
                             ])

# print(data_csv)
