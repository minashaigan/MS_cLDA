import sys
import re
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if(K <= 9): topic = '0' + args[2]
    else: topic = args[2]

    if(data_type == 1):
        input_file = 'D:/mutational_signature/MS_LDA-master/MS_LDA-master/result/data' + str(data_type) + '_o' + threshold \
            + '_' + cancer_type + '/result_k' + topic + '.txt'
    else:
        input_file = 'D:/mutational_signature/MS_LDA-master/MS_LDA-master/result/data' + str(data_type) + '_o' + threshold \
            + '_' + cancer_type + '/figure/' + str(K) + '/k' + topic + '.txt'

    ex = open(input_file)
    co = open('D:/mutational_signature/MS_LDA-master/MS_LDA-master/data/signature_probability.txt')

    data_ex = ex.readlines()
    count_ex = 0
    p_ex = np.zeros([K,96])
    for line in data_ex:
        if((count_ex > 1) and (count_ex < K + 2)):
            words = line.split()
            for signature in range(96):
                p_ex[count_ex-2,signature] = float(words[signature])
        count_ex += 1

    data_co = co.readlines()
    count_co = 0
    p_co = np.zeros([30,96])
    for line in data_co:
        if(count_co != 0):
            words = line.split()
            for signature in range(33):
                if(signature >= 3):
                    p_co[signature-3,count_co-1] = float(words[signature])
        count_co += 1

    ex.close()
    co.close()

    p_ex_copy = p_ex.copy()
    p_co_copy = p_co.copy()

    i_ex = 0
    while(i_ex < 96):
        j_ex = int(i_ex/4)
        count_ex = 0
        while(count_ex != 4):
            for k_ex in range(K):
                for l_ex in range(4):
                    p_ex_copy[k_ex,i_ex+4*count_ex+l_ex] = p_ex[k_ex,j_ex+l_ex]
            j_ex += 24
            count_ex += 1
        i_ex += 16

    i_co = 0
    while(i_co < 96):
        j_co = int(i_co/4)
        count_co = 0
        while(count_co != 4):
            for k_co in range(30):
                for l_co in range(4):
                    p_co_copy[k_co,i_co+4*count_co+l_co] = p_co[k_co,j_co+l_co]
            j_co += 24 
            count_co += 1
        i_co += 16

    p_ex = p_ex_copy.copy()
    p_co = p_co_copy.copy()

    matching_file = 'D:/mutational_signature/MS_LDA-master/MS_LDA-master/result/data' + str(data_type) + '_o' + threshold + '_' \
            + cancer_type + '/figure/' + str(K) + '/matching.txt'

    matching = open(matching_file,'r')
    data = matching.read()
    data = data.replace("{","")
    data = data.replace("}","")
    data = data.replace(":","")
    data = data.replace(",","")
    match = data.split()

    match_list = np.zeros([2,K])
    i = 0
    j = 0
    while(i < K*2-1):
        match_list[0,j] = int(match[i])
        match_list[1,j] = int(match[i+1])
        i += 2
        j += 1
    matching.close()
    df_file = 'D:/mutational_signature/MS_LDA-master/MS_LDA-master/result/data' + str(data_type) + '_o' + threshold + '_' \
            + cancer_type + '/figure/' + str(K) + '/df.csv'

    dataframe = open(df_file,'w')

    dataframe.write('MS,')
    for i in range(96):
        dataframe.write(str(i+1))
        dataframe.write(',')
    dataframe.write('\n')
    for i in range(K):
        dataframe.write('Predicted Signature ')
        dataframe.write(str(i+1) + ' (mutation) in ' + args[4] + ',')
        for j in range(96):
            dataframe.write(str(p_ex[i,j]))
            if(j != 95):
                dataframe.write(',')
        dataframe.write('\n')
        dataframe.write('COSMIC Known Signature ')
        dataframe.write(str(int(match_list[1,i])+1) + ',')
        for j in range(96):
            dataframe.write(str(p_co[int(match_list[1,i]),j]))
            if(j != 95):
                dataframe.write(',')
        if(i != K-1):
            dataframe.write('\n')
    dataframe.close()

    sim = pd.read_csv(df_file)
    sim = sim.drop('Unnamed: 97',axis = 1)

    labels = 96*[0]
    for i in range(96):
        first = i % 16
        if(first == 0 or first == 1 or first == 2 or first == 3):
            labels[i] = 'A'
        if(first == 4 or first == 5 or first == 6 or first == 7):
            labels[i] = 'C'
        if(first == 8 or first == 9 or first == 10 or first == 11):
            labels[i] = 'G'
        if(first == 12 or first == 13 or first == 14 or first == 15):
            labels[i] = 'T'
        for j in range(16):
            if(i == j):
                labels[i] += '(C>A)'
            if(i == j+16):
                labels[i] += '(C>G)'
            if(i == j+32):
                labels[i] += '(C>T)'
            if(i == j+48):
                labels[i] += '(T>A)'
            if(i == j+64):
                labels[i] += '(T>C)'
            if(i == j+80):
                labels[i] += '(T>G)'
        second = i % 4
        if(second == 0):
            labels[i] += 'A'
        if(second == 1):
            labels[i] += 'C'
        if(second == 2):
            labels[i] += 'G'
        if(second == 3):
            labels[i] += 'T'

    colorlist = 96*[0]
    for i in range(16):
        colorlist[i] = 'r'
    for i in range(16,32):
        colorlist[i] = 'g'
    for i in range(32,48):
        colorlist[i] = 'b'
    for i in range(48,64):
        colorlist[i] = 'c'
    for i in range(64,80):
        colorlist[i] = 'm'
    for i in range(80,96):
        colorlist[i] = 'y'

    for i in range(0,K*2,2):
        fig = plt.figure()
        left = np.arange(1,97,1)
        height_ex = sim.ix[i,1:]
        height_co = sim.ix[i+1,1:]
        title1 = str(sim.ix[i,0])
        title2 = str(sim.ix[i+1,0])
        ax1 = fig.add_subplot(211)
        ax1.bar(left,height_ex,width=1,color=colorlist,align="center")
        ax1.set_ylim(0,0.2)
        ax1.set_xticks(left)
        ax1.set_xticklabels(labels)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(90)
        ax1.tick_params(labelsize=5)
        ax1.set_ylabel('p (mutation = x)')
        ax1.set_title(title1)
        ax2 = fig.add_subplot(212)
        ax2.bar(left,height_co,width=1,color=colorlist,align="center")
        ax2.set_ylim(0,0.2)
        ax2.set_xticks(left)
        ax2.set_xticklabels(labels)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(90)
        ax2.tick_params(labelsize=5)
        ax2.set_xlabel('mutation x')
        ax2.set_ylabel('p (mutation = x)')
        ax2.set_title(title2)
        fig.tight_layout()
        name = 'D:/mutational_signature/MS_LDA-master/MS_LDA-master/result/data' + str(data_type) + '_o' + threshold + '_' \
             + cancer_type + '/figure/' + args[2] + '/' \
             + str((i+1)//2 + 1) + '_match.png'
        fig.savefig(name,dpi=300)
        plt.close(1)

if __name__ == '__main__':
    args = sys.argv
    #args[1]:data_type, [2]:number_of_topic
    #    [3]:threshold number word, [4]:cancer_type
    data_type = int(args[1])
    K = int(args[2])
    threshold = args[3]
    cancer_type = args[4]
    main()
