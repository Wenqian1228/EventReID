#!/bin/scripts

python Train.py   --arch 'PSTA'\
                  --config_file "./configs/softmax_triplet_prid128.yml"\
                  --dataset 'prid'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
