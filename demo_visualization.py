import torch
from feature_visualizer import FeatureVisualizer

## Load Data
img = torch.load('data/image.pt')   # torch.Size([4, 3,   256, 128])
f = torch.load('data/feature.pt')   # torch.Size([4, 2048,  8,   4])

## Feature visualizer
V = FeatureVisualizer(
    cmap_type='jet',
    reduce_type='mean',
)

## Visualize Feature
V.save_feature(f, save_path='person_0051/0001.png')

## Visualize Image
# V.save_image(img, save_path='demo/image.jpg', recover=False)
# V.save_image(img, save_path='person_0051/0001.png', recover=False)

## Visualize both Image and Feature
# V.save_both(img, f, 'demo/demo.jpg', recover=False)

