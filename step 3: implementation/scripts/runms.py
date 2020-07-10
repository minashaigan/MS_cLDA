import sys
import os
from multiprocessing import Pool
import subprocess

def main():
    arguments = []
    for i in range(1,2):
        for j in range(1,15):
            arguments.append((str(j), data_type, threshold, 
                              str(i)))
    pool = Pool()
    print(pool.starmap(execute, arguments))

def execute(num_topic, data_type, threshold, experiment):
    result_path = 'D:/mutational_signature/MS_cLDA/simulated_result/data' + data_type + '_o' + threshold + '_all_' +\
                   experiment
    if(os.path.exists(result_path) == False): os.mkdir(result_path)
    cmd = 'D:/mutational_signature/MS_cLDA/bin/MS.exe ' + num_topic + ' ' + data_type + ' ' + threshold +\
          ' ' + experiment
    subprocess.call(cmd.split())

if __name__ == '__main__':
    args = sys.argv
    data_type = args[1]
    threshold = args[2]
    main()
