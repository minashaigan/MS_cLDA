import sys
import re
import pandas as pd
import numpy as np
import linecache


def main():
    args = sys.argv
    """args[1] : threshold number of word [2]:cancer type"""

    K = 96
    threshold = int(args[1])
    pre_data_file = 'D:/mutational_signature/MS_cLDA/data/data1/Pre_data1_o' + args[1] + '.txt'
    pre_data = pd.read_csv(pre_data_file, delimiter='\t')
    cancer_type = args[2]
    pre_data = select_cancer_type(pre_data, threshold, cancer_type)
    pre_data.reset_index(drop=True, inplace=True)

    output_file = 'D:/mutational_signature/MS_cLDA/data/data1_o' + args[1] + '_' + args[2] + '.txt'
    output = open(output_file, 'w')

    name_list = pre_data['Sample name']
    last_document = 'xxxxxxxxxxxx'
    number_of_document = 0
    for name in name_list:
        if(name != str(last_document)):
            last_document = name
            number_of_document += 1

    last_document = name_list[0]
    type_list = list()
    type_list.append(pre_data['Primary site'][0])
    data_mat = np.zeros([number_of_document, K])
    index = 0
    error = 0
    document = 0
    for name in name_list:
        if(name != str(last_document)):
            last_document = name
            document += 1
            type_list.append(pre_data['Primary site'][index])
        selected = calc_word(pre_data['Mutation CDS'][index],
                             pre_data['Mutation genome position'][index],
                             pre_data['Mutation strand'][index])
        if(selected == -1):
            error += 1
        else:
            data_mat[document, selected] += 1
        index += 1
        print(index)

    """ check """
    drop_list = list()
    for i in range(number_of_document):
        sum_words = 0
        for j in range(K):
            sum_words += data_mat[i, j]
        if(sum_words == 0):
            drop_list.append(i)
    number_of_document -= len(drop_list)
    data_mat = np.delete(data_mat, drop_list, 0)
    type_list = np.delete(type_list, drop_list, 0)

    print(str(len(drop_list)))

    output.write(str(number_of_document) + ' 96\n')

    for i in range(number_of_document):
        for k in range(K):
            if(k == K-1):
                output.write(str(int(data_mat[i, k])) + '\n')
            else:
                output.write(str(int(data_mat[i, k])) + ' ')


def select_cancer_type(pre_data, threshold, cancer_type):
    pre_data = pre_data.loc[pre_data['Primary site'] == cancer_type]
    print(pre_data)
    pre_data.sort_values(by='Sample name', inplace=True)
    pre_data.reset_index(drop=True, inplace=True)
    name_list = pre_data['Sample name']
    last_document = name_list[0]
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
    pre_data.drop(drop_list, inplace=True)
    return pre_data


def calc_word(mutation, position, strand):
    before = mutation[len(mutation)-3]
    after = mutation[len(mutation)-1]
    position_list = re.split(r'[:-]', position)
    if(int(position_list[0]) == 23):
        chromosome = 'X'
    elif(int(position_list[0]) == 24):
        chromosome = 'Y'
    elif(int(position_list[0]) == 25):
        chromosome = 'M'
    else:
        chromosome = int(position_list[0])
    start = int(position_list[1])
    num = int(position_list[2]) - int(position_list[1]) + 1
    GRCh_file = 'D:/mutational_signature/MS_LDA-master/MS_LDA-master/raw_data/chr' + str(chromosome) + '.fa'
    quotient = start // 50
    surplus = start % 50

    if(surplus != 0):
        target_index = int(surplus) - 1
    else:
        quotient -= 1
        target_index = 49
    targetline = linecache.getline(GRCh_file, int(quotient)+1)

    if(((targetline[target_index] != before) and (strand == '+')) or
            ((targetline[target_index] != swap(before))and(strand == '-'))):
        print('error: ' + mutation)
        print('target: ' + targetline[target_index])
        print('strand: ' + strand)
        strand = swap(strand)
        if(((targetline[target_index] != before) and (strand == '+')) or 
                ((targetline[target_index] != swap(before))and(strand == '-'))):
            print('still error')
            return -1

    if((target_index >= 1) and (target_index <= 48)):
        pattern = 1
    elif(target_index == 0):
        pattern = 2
    elif(target_index == 49):
        pattern = 3

    if(pattern == 1):
        forward = targetline[target_index - 1]
        backward = targetline[target_index + 1]
    elif(pattern == 2):
        pre_line = linecache.getline(GRCh_file, int(quotient))
        forward = pre_line[49]
        backward = targetline[target_index + 1]
    elif(pattern == 3):
        post_line = linecache.getline(GRCh_file, int(quotient)+2)
        forward = targetline[target_index - 1]
        backward = post_line[0]

    if(((strand == '+') and (before in ['A', 'G'])) or ((strand == '-') and 
        (before in ['C', 'T']))):
        buf_f = swap(forward)
        forward = swap(backward)
        backward = buf_f
    if(before in ['A', 'G']):
        before = swap(before)
        after = swap(after)

    if(forward == 'A'):
        first = 0
    elif(forward == 'C'):
        first = 1
    elif(forward == 'G'):
        first = 2
    else:
        first = 3

    if(before == 'C'):
        if(after == 'A'):
            second = 0
        elif(after == 'G'):
            second = 1
        else:
            second = 2
    elif(before == 'T'):
        if(after == 'A'):
            second = 3
        elif(after == 'C'):
            second = 4
        else:
            second = 5

    if(backward == 'A'):
        third = 0
    elif(backward == 'C'):
        third = 1
    elif(backward == 'G'):
        third = 2
    else:
        third = 3
    answer = 24*first + 4*second + third
    return(answer)


def swap(base):
    if(base == 'A'):
        return('T')
    elif(base == 'C'):
        return('G')
    elif(base == 'G'):
        return('C')
    elif(base == 'T'):
        return('A')
    elif(base == '+'):
        return('-')
    elif(base == '-'):
        return('+')
    else:
        return(base)


if __name__ == '__main__':
    main()
