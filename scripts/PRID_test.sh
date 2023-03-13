#!/bin/scripts
export http_proxy=http://192.168.9.99:3128
export https_proxy=http://192.168.9.99:3128
python Test.py  --arch 'PSTA'\
                --dataset 'mars'\
                --test_sampler 'Begin_interval'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --test_path '/ghome/caocz/code/Event_Camera/Event_Re_ID/VideoReID_PSTA/pth/Mars.pth'