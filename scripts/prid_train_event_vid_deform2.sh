#!/bin/scripts

python Train_event_vid.py   --arch 'PSTA_img_event_deform2'\
                  --config_file "./configs/softmax_triplet_prid128.yml"\
                  --dataset 'prid_event_vid'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
                  --seq_len 8 \