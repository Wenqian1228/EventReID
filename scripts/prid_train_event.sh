#!/bin/scripts

python Train_event.py   --arch 'PSTA'\
                  --config_file "./configs/softmax_triplet_prid_event.yml"\
                  --dataset 'prid_event'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
