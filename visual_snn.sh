export http_proxy=http://192.168.16.5:3128
export https_proxy=http://192.168.16.5:3128

python Train_event_vid.py   --arch 'PSTA_SNN3'\
                  --config_file "./configs/softmax_triplet_prid128.yml"\
                  --dataset 'prid_event_vid'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
                  --seq_len 4 \