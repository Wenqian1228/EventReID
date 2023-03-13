import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torchvision.models as models
plt.rcParams['font.sans-serif']=['STSong']

model = models.resnet18(pretrained=True)

# #1.1.模型查看
# print(model)
# model_features = list(model.children())
# print(model_features[4][0]) #取第4层Sequential()中的第0个blk


# #1.2 模型查看
# from torchsummary import summary
# summary(model.cuda(), input_size=(3, 224, 224), batch_size=-1)


#2. 导入数据
# 以RGB格式打开图像
# Pytorch DataLoader就是使用PIL所读取的图像格式
# 建议就用这种方法读取图像，当读入灰度图像时convert('')
def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')#是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)#torch.Size([3, 224, 224])
    image_info = image_info.unsqueeze(0)#torch.Size([1, 3, 224, 224])因为model的输入要求是4维，所以变成4维
    return image_info #变成tensor数据



# 获取第k层的特征图
'''
args:
k:定义提取第几层的feature map
x:图片的tensor
model_layer：是一个Sequential()特征层
'''
def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):# model的第一个Sequential()是有多层，所以遍历
            x = layer(x)                           # torch.Size([1, 64, W, H])生成了64个通道
            if k == index:
                return x


def run_layer(model_layer, x):
    with torch.no_grad():
        x = model_layer(x)
        # [1, 64, 112, 112]

    return x



#  可视化特征图
def show_feature_map(feature_map, is_save=False, save_path='maps', cmap='gray', map_size:tuple=None, mpa_k:int=-1):
    '''
    :param feature_map: [1, dims, H, W]
    :return: None
    '''

    # 是否对其尺寸
    if map_size:
        feature_map = torch.nn.Upsample(size=map_size, mode='nearest')(feature_map)

    feature_map = feature_map.squeeze(0)         # [1, 64, 112, 112] -> [64, 112, 112]

    feature_map_num = feature_map.shape[0]       #返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    for index in range(feature_map_num):         #通过遍历的方式，将64个通道的tensor拿出
        single_dim = feature_map[index] # shape[112, 112]

        plt.subplot(row_num, row_num, index+1) # idx [1...64]
        plt.imshow(single_dim, cmap=cmap)
        # plt.imshow(single_dim, cmap='viridis')
        plt.axis('off')

        if is_save:
            imageio.imwrite( f"./{save_path}/{mpa_k}_" + str(index+1) + ".jpg", single_dim)
    plt.show()



if __name__ ==  '__main__':
    # ------------------------------
    # 定义提取第the_maps_k层的feature map
    image_dir = "111.jpg"
    the_maps_k = 0

    image_info = get_image_info(image_dir)
    # get model
    model = models.resnet18(pretrained=False)

    # @ 调试这里配合 K
    model_layers= list(model.children())

    feature_map = get_k_layer_feature_map(model_layers, the_maps_k, image_info)
    show_feature_map(feature_map, is_save=True, cmap='hot', map_size=None)


    # # [2] show muti
    # for the_maps_k in [0, 4, 5, 6, 7]:
    #     feature_map = get_k_layer_feature_map(model_layers, the_maps_k, image_info)
    #     show_feature_map(feature_map, is_save=True, cmap='hot', map_size=(200, 200), mpa_k=the_maps_k)
    #     print(f"Show  Map idx : {the_maps_k}")
