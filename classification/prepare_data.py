#check for duplicate pictures
import os
import hashlib

hashes = {}

for type_folder in os.listdir('dataset'):
    for tumor_folder in os.listdir(f'dataset/{type_folder}'):
        for file in os.listdir(f'dataset/{type_folder}/{tumor_folder}'):
            filepath = f'dataset/{type_folder}/{tumor_folder}/{file}'
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in hashes:
                    hashes[file_hash] = []
                
                hashes[file_hash].append(filepath)
#delete duplicates
count = 0
for file_hash, path_list in hashes.items():
    while len(path_list) > 1: 
        to_delete = path_list.pop()
        os.remove(to_delete)
print(count)

#count dataset images
def count_images(path: str)-> str:
    result = {}
    for type_folder in os.listdir(f'{path}'):
        temp = []
        for tumor_folder in os.listdir(f'{path}/{type_folder}'):
            count = 0
            for file in os.listdir(f'{path}/{type_folder}/{tumor_folder}'):
                count+=1
            temp.append(list([tumor_folder,count]))
        result[type_folder] = temp

    for i in result.keys():
        print(f'{i}: {result[i]}')
count_images('dataset')

#with duplicates:
# Testing: [['glioma_tumor', 400], ['meningioma_tumor', 421], ['no_tumor', 510], ['pituitary_tumor', 374]]
# Training: [['glioma_tumor', 2147], ['meningioma_tumor', 2161], ['no_tumor', 1990], ['pituitary_tumor', 2284]]
# Testing count: 1705
# Training count: 8582

#merge testing and training folders (unmerged from original dataset)
import shutil
if os.path.exists(f'dataset/Testing'):
    for tumor_folder in os.listdir(f'dataset/Testing'):
        for file in os.listdir(f'dataset/Testing/{tumor_folder}'):
            shutil.move(f'dataset/Testing/{tumor_folder}/{file}', f'dataset/Training/{tumor_folder}/{file}')
    shutil.rmtree('dataset/Testing')

#split testing folder into testing,validation folders
import splitfolders

input_folder = os.path.join('dataset/Training')
output_folder = os.path.join('dataset_split')
split_ratio = (0.7,0.15,0.15)

splitfolders.ratio(
    input_folder,
    output = output_folder,
    seed = 12,
    ratio = split_ratio,
    group_prefix= None
)

count_images('dataset_split')

#balance dataset
import random
balanced_values = {
    'train' : 1140,
    'val' : 245,
    'test' : 256
}
for type_folder in os.listdir('dataset_split'):
    for tumor_folder in os.listdir(f'dataset_split/{type_folder}'):
        while len(os.listdir(f'dataset_split/{type_folder}/{tumor_folder}')) > balanced_values[type_folder]:
                files = os.listdir(f'dataset_split/{type_folder}/{tumor_folder}')
                os.remove(f'dataset_split/{type_folder}/{tumor_folder}/{random.choice(files)}')

#todo pridat ze cisla balanced_values neni hardcoded ale vytahne to nejnizsi cislo souboru ve slozce + celkove vsechny path nebudou tak hardcoded napr dataset_split/

count_images('dataset_split')


