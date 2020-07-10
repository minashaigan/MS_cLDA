#!/bin/bash
lesions=("breast" "endometrium" "large_intestine" "liver" "lung" "oesophagus" "prostate" "skin" "soft_tissue" "stomach" "upper_aerodigestive_tract" "urinary_tract")
th=$1
for le in ${lesions[@]}; do
	echo "python Preprocessing/get_M1.py ${th} ${le}"
    python Preprocessing/get_M1.py ${th} ${le} 
done
