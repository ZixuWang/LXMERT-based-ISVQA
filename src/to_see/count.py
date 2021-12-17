import os

path = '/root/dataset/NuScenes/feature_output_train'
files = os.listdir(path)
num = len(files)
img_set = num / 6

print(num)
print(img_set)