import sys
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
#from ortoolpy import stable_matching

def main():
    if(K <= 9): topic = '0' + args[2]
    else: topic = args[2]

    if(args[1] == '1'):
        input_file = 'D:/mutational_signature/MS_cLDA/result/data' + str(data_type) + '_o' + threshold + '_all_1' \
                   +'/result_k' + topic + '.txt'
    else:
        input_file = 'D:/mutational_signature/MS_cLDA/result/data' + str(data_type) + '_o' + threshold + '_all_1' \
                   + '/figure/' + str(K) + '/k' + topic + '.txt'

    ex = open(input_file)
    co = open('D:/mutational_signature/MS_cLDA/data/signature_probability.txt')

    data_ex = ex.readlines()
    count_ex = 0
    p_ex = np.zeros([K,96])
    for line in data_ex:
        if((count_ex > 1) and (count_ex < K + 2)):
            words = line.split()
            for signature in range(96):
                p_ex[count_ex-2,signature] = float(words[signature])
        count_ex += 1