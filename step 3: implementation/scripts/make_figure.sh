#!/bin/bash
data_type=$1
threshold=$2
cancer_type=all
#str1=matsutani@133.9.8.88:project/MS_LDA/result/data${data_type}_o${threshold}_${cancer_type}*
#scp -r ${str1} result
python D:/mutational_signature/MS_cLDA/Drawing/find_best_K.py ${data_type} ${threshold}
FILENAME=D:/mutational_signature/MS_cLDA/ref/${data_type}_${threshold}_${cancer_type}.txt
cnt=0
array=()
while read line;
do
    cnt=$(expr $cnt + 1)
    if test $cnt -eq 1; then
	num_data=$line
    fi
    if test $cnt -eq 2; then
	number_of_topic=$line
    fi
    if test $cnt -ge 3; then
	array+=($line)
    fi
done<$FILENAME
echo ${number_of_topic}
python D:/mutational_signature/MS_cLDA/Drawing/comparison_K.py ${data_type} ${threshold} 1