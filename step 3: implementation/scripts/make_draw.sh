#!/bin/bash
data_type=$1
number_of_topic=$2
threshold=$3

if [ ${data_type} -eq 1 ]; then
    python D:/mutational_signature/MS_cLDA/Drawing/matching.py ${data_type} ${number_of_topic} \
        ${threshold}
    python D:/mutational_signature/MS_cLDA/Drawing/data_represent_single.py ${data_type} ${number_of_topic} \
        ${threshold}
fi

if [ ${data_type} -eq 2 ]; then
    python D:/mutational_signature/MS_cLDA/Drawing/n2_to_n1.py ${threshold} ${number_of_topic} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/matching.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/data_represent.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/draw_data2.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
fi

if [ ${data_type} -eq 3 ]; then
    python D:/mutational_signature/MS_cLDA/Drawing/n3_to_n1.py ${threshold} ${number_of_topic} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/matching.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/data_represent.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/draw_data3_indel.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
fi
if [ ${data_type} -eq 4 ]; then
    python D:/mutational_signature/MS_cLDA/Drawing/n4_to_n1.py ${threshold} ${number_of_topic} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/matching.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/data_represent.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/draw_data4_indel.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
    python D:/mutational_signature/MS_cLDA/Drawing/draw_data4.py ${data_type} ${number_of_topic} \
        ${threshold} ${cancer_type}
fi

