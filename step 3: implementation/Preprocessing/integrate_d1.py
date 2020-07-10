import sys
import pandas as pd
import numpy as np

def main():
    args = sys.argv
    """args[1]:threshold number of word [2]:division number"""
    
    threshold = int(args[1])
    Num = int(args[2])
    data = pd.read_csv('D:/mutational_signature/MS_cLDA/data/data1/Pre_data1_1.txt', delimiter = '\t')
    
    """ integrate """
    for i in range(Num-1):
        page = i+2
        input_file = 'D:/mutational_signature/MS_cLDA/data/data1/Pre_data1_' + str(page) + '.txt'
        plus_data = pd.read_csv(input_file, delimiter = '\t')
        data = pd.concat([data, plus_data])

    data.reset_index(drop = True, inplace = True)

    """ select which have  the number of words more than threshold """
    last_document = data['Sample name'][0]
    name_list = data['Sample name']
    print(name_list)
    drop_list = list()
    sum_of_words = 0
    temp_index = 0
    index_list = list()
    for name in name_list:
        if(name != last_document):
            if(sum_of_words < threshold):
                drop_list.extend(index_list)
            sum_of_words = 0
            last_document = name
            index_list = list()
        sum_of_words += 1
        index_list.append(temp_index)
        temp_index += 1
    if(sum_of_words < threshold):
        drop_list.extend(index_list)

    data.drop(drop_list, inplace = True)

    data.reset_index(drop = True, inplace = True)
    data.to_csv('D:/mutational_signature/MS_cLDA/data/data1/Pre_data1_o' + args[1] + '.txt', sep = '\t')

if __name__ == '__main__':
    main()
