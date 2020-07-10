# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:22:40 2020

@author: x450c1
"""

import sys
import numpy as np
from numpy import random
from scipy.stats import dirichlet

def counting_sort(array, maxval):
    m = maxval + 1
    count = [0] * m               
    for a in array:
        count[a] += 1            
    return count

def main():
    N_K = 10
    N_V = 96
    args = sys.argv
    # args[1]:alpha, [2]:eta [3]:number_of_lesions, 
    #     [4]:number_of_samples, [5]:number_of_words
    temp_a = float(args[1]) # 0.1
    alpha = np.full(N_K, temp_a)
    eta = float(args[2]) # 1
    N_J = int(args[3])   # 4
    N_M = int(args[4])	 # 25
    N_D = int(args[5])	 # 2000

	#known lam
    co = open('D:/mutational_signature/MS_cLDA/data/signature_probability.txt')
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

    co.close()
    lam = np.zeros([N_K,N_V])
    for i in range(N_K):
        lam[i, :] = p_co[i, :]
    
    tau = np.zeros([N_J, N_K])
    rho = np.zeros([N_J, N_M, N_K])
    
    z = np.zeros([N_J, N_M, N_D])
    words = np.zeros([N_J, N_M, N_D])
    
    for j in range(N_J):
        tau[j,:] = dirichlet.rvs(alpha, size=1, random_state=1)

    #print(tau)
     
    for j in range(N_J):
    	for d in range(N_M):
    		rho[j, d, :] = dirichlet.rvs(eta*tau[j,:], size=1, random_state=1)
          
    #print(rho)
    
    for j in range(N_J):
        for d in range(N_M):
            for i in range(N_D):
                z[j, d, i] = np.argmax(random.multinomial(100, rho[j, d, :], size=1))
    
    #print(z)
          
    for j in range(N_J):
        for d in range(N_M):
            for i in range(N_D):
                k = int(z[j, d, i])
                words[j, d, i] = np.argmax(random.multinomial(100, lam[k, :], size=1))
                
    words = words.astype(int)
#    print(words)
#    print(counting_sort( words[0, 0, :], 95 ))
    
    cancer_types = np.array(['breast', 'skin', 'prostate', 'stomach'])
    
    for j in range(N_J):
        words_file = "D:/mutational_signature/MS_cLDA/generated_data/data1_o400_" \
        + cancer_types[j] + ".txt"
    
        dataframe = open(words_file, 'w')
    
        dataframe.write(str(N_M) + ' 96\n')
        for d in range(N_M):
            counts = counting_sort( words[j, d, :], 95 )
            for i in range(96):
                dataframe.write(str(counts[i]) + ' ')
            dataframe.write('\n')
        dataframe.close()
	
if __name__ == '__main__':
    main()