import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    i = 2
    while(i < K+1):
        string = 'D:/mutational_signature/MS_cLDA/simulated_result/data' + str(data_type) + '_o' + threshold + '_' \
                + 'all_' + str(iteration) + '/result_k'
        if(i <= 9):
            string += '0' + str(i) + '.txt'
            files[i-2] = string
        else:
            string += str(i) + '.txt'
            files[i-2] = string
        i += 1;

    ELBO = np.zeros([K-1])
    Perplexity = np.zeros([K-1])

    labels = (K-1)*[0]
    for i in range(K-1):
        data = open(files[i])
        elbo = data.readline()
        elbo = elbo.replace("\n","")
        ELBO[i] = float(elbo)
        perplexity = data.readline()
        perplexity = perplexity.replace("\n","")
        Perplexity[i] = float(perplexity)
        labels[i] = 'topic' + str(i+2)

    max_el = 0
    min_pe = 0
    semi_el = 0
    semi_pe = 0
    for i in range(K-1):
        if(ELBO[i] > ELBO[max_el]):
            semi_el = max_el
            max_el = i
        elif(ELBO[i] > ELBO[semi_el]):
            semi_el = i
        if(Perplexity[i] < Perplexity[min_pe]):
            semi_pe = min_pe
            min_pe = i
        elif(Perplexity[i] < Perplexity[semi_pe]):
            semi_pe = i

    el_colors = (K-1)*["b"]
    el_colors[max_el] = "r"
    el_colors[semi_el] = "g"

    pe_colors = (K-1)*["b"]
    pe_colors[min_pe] = "r"
    pe_colors[semi_pe] = "g"

    left = np.arange(1,K,1)
    height = ELBO.copy()

    plt.bar(left,height, align = "center", color = el_colors)
    plt.title("Variational lower bound of each signature number")
    plt.xlabel("signature numbers")
    plt.ylabel("Variational lower bound")
    plt.xticks(left, labels, rotation = 90, fontsize = "small")
    plt.tight_layout()
    name = 'D:/mutational_signature/MS_cLDA/simulated_result/data' + str(data_type) + '_o' + threshold + '_' \
         + 'all_' + str(iteration) + '/figure/ELBO.png'
    plt.savefig(name)
    plt.close(1)

    plt.figure()

    plt.bar(left,height, align = "center", color = el_colors)
    plt.title("Variational lower bound of each signature number")
    plt.xlabel("signature numbers")
    plt.ylabel("Variational lower bound")
    plt.xticks(left, labels, rotation = 90, fontsize = "small")
    y_max =max(height) * 0.99
    y_min =min(height) * 1.01
    plt.ylim(ymax = y_max, ymin = y_min)
    plt.tight_layout()
    name = 'D:/mutational_signature/MS_cLDA/simulated_result/data' + str(data_type) + '_o' + threshold + '_' \
         + 'all_' + str(iteration) + '/figure/ELBO_a.png'
    plt.savefig(name)
    plt.close(1)

    plt.figure()

    height = Perplexity.copy()

    plt.bar(left,height, align = "center", color = pe_colors)
    plt.title("Perplexity of each topic numbers")
    plt.xlabel("topic numbers")
    plt.ylabel("Perplexity")
    plt.xticks(left, labels, rotation = 90, fontsize = "small")
    plt.tight_layout()
    name = 'D:/mutational_signature/MS_cLDA/simulated_result/data' + str(data_type) + '_o' + threshold + '_' \
         + 'all_' + str(iteration) + '/figure/Perplexity.png'
    plt.savefig(name)
    plt.close(1)

    plt.figure()
    plt.bar(left,height, align = "center", color = pe_colors)
    plt.title("Perplexity of each topic numbers")
    plt.xlabel("topic numbers")
    plt.ylabel("Perplexity")
    plt.xticks(left, labels, rotation = 90, fontsize = "small")
    y_max = max(height) * 1.01
    y_min = min(height) * 0.99
    plt.ylim(ymax = y_max, ymin = y_min)
    plt.tight_layout()
    name = 'D:/mutational_signature/MS_cLDA/simulated_result/data' + str(data_type) + '_o' + threshold + '_' \
         + 'all_' + str(iteration) + '/figure/Perplexity_a.png'
    plt.savefig(name)
    plt.close(1)

if __name__ == '__main__':
    K = 15
    files = (K-1)*[0]
    args = sys.argv
    # args[1]:data_type, [2]:threshold number of word
    #     [3]:cancer_type, [4]:iteration
    data_type = int(args[1])
    threshold = args[2]
    #cancer_type = args[3]
    iteration = int(args[3])
    main()

