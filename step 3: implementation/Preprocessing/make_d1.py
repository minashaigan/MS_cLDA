import sys
import pandas as pd


def main():
    args = sys.argv
    """ args[1]:array_number [2]:division number"""
    index = int(args[1])
    Num = int(args[2])
    mutation_file = 'D:/CosmicMutantExport.tsv'
    complete_data = pd.read_csv(mutation_file, delimiter='\t')
    data = select_data(complete_data, index, Num)
    output_file = 'D:/mutational_signature/MS_cLDA/data/data1/Pre_data1_' + args[1] + '.txt'
    data.to_csv(output_file, sep='\t', index=None)


def select_data(complete_data, index, Num):
    """ make_data 1 and 2 """
    data = complete_data[complete_data['Mutation Description'].str.contains(
        'Substitution')]
    del complete_data

    """ select which use GRCh38 """
    data = data[data['GRCh'] == 38]

    data = data.loc[:, ['Sample name', 'Mutation CDS',
                        'Mutation genome position', 'Mutation Description',
                        'Primary site', 'Primary histology',
                        'Histology subtype 1', 'Mutation strand']]

    data = data.sort_values(by='Sample name')
    data.reset_index(drop=True, inplace=True)

    """ Pearallelization """
    CDS = data['Mutation CDS']
    all_number = len(data.index)
    num = Num - 1
    unit = all_number // num
    max_index = index * unit
    start_index = max_index - unit
    if(index <= num):
        temp_CDS = CDS[start_index:max_index]
        temp_data = data[start_index:max_index]
    else:
        temp_CDS = CDS[start_index:]
        temp_data = data[start_index:]
    del CDS
    del data

    """ select which single substitution """
    count = start_index
    drop_list = list()
    for i in temp_CDS:
        char_list = list(i)
        if((char_list[len(char_list)-4] in ['A', 'C', 'G', 'T', '>']) \
                or char_list[len(char_list)-2] != '>'):
            drop_list.append(count)
        count += 1
    temp_data.drop(drop_list, inplace=True)
    return temp_data


if __name__ == '__main__':
    main()
