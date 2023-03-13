#!/bin/scripts

python Train.py   --arch 'HiCMD_img'\
                  --config_file "./configs/softmax_triplet_prid.yml"\
                  --dataset 'prid'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
                  --seq_len 4 \
