#!/bin/bash
division=$1
echo "make d1..."
for i in $(seq 1 ${division}); do
    python Preprocessing/make_d1.py ${i} ${division}
done
python Preprocessing/integrate_d1.py 400 ${division}
python Preprocessing/integrate_d2.py 400 ${division}