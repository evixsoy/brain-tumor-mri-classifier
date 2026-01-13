# ungroup 1. dataset containing images & mask into separate folders
import os
import shutil
import hashlib
import random


categories = ['glioma', 'meningioma', 'pituitary']
folder_types = ['masks', 'images']
for folder in folder_types:
     os.makedirs(f'cleaned_dataset/{folder}', exist_ok=True)
     for tumor in categories:
          os.makedirs(f'cleaned_dataset/{folder}/{tumor}', exist_ok=True)


used_ids = set()
folderpath = 'dataset-segmentation/Segmentation-masks&images'
for tumor_folder in os.listdir(folderpath):
     tumor_folder_path = os.path.join(folderpath, tumor_folder)
     for file in os.listdir(tumor_folder_path):
          file_path = os.path.join(tumor_folder_path, file)
          id = ""
          check = "0"
        
          for i in range(4, 9):
            if i < len(file) and file[i].isdigit():
                id += file[i]
                
          if id in used_ids:
              continue
          else:
               used_ids.add(id)

          if 'mask' in file.lower():
               image_file = f"enh_{id}.png"
               mask_file = file
          else:
               image_file = file
               mask_file = f"enh_{id}_mask.png"

          mask_path = os.path.join(tumor_folder_path, mask_file)
          image_path = os.path.join(tumor_folder_path, image_file)

          #get tumor type
          if not tumor_folder.lower() in categories:
               tumor_type = tumor_folder.split(" ")[0].lower()
          else:
               tumor_type = tumor_folder.lower()

          shutil.move(image_path, f'cleaned_dataset/images/{tumor_type}/{image_file}')
          shutil.move(mask_path, f'cleaned_dataset/masks/{tumor_type}/{mask_file}')
     

corrupted_paths = ['dataset-segmentation/image/0','dataset-segmentation/mask/0', 'dataset-segmentation/mask/1/Tr-gl_0899_m.jpg', 'dataset-segmentation/mask/2/Tr-me_0540.jpg']
for path in corrupted_paths:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    except FileNotFoundError:
        pass

folders = ['image', 'mask']
tumor_types = {1: 'glioma', 2: 'meningioma', 3:'pituitary'}

for tumor_id, tumor_name in tumor_types.items():
    source_image = f'dataset-segmentation/image/{tumor_id}'
    source_mask = f'dataset-segmentation/mask/{tumor_id}'

    final_image = f'cleaned_dataset/images/{tumor_name}'
    final_mask  = f'cleaned_dataset/masks/{tumor_name}'

    for file in os.listdir(source_image):
        shutil.move(os.path.join(source_image, file), os.path.join(final_image, file))
    for file in os.listdir(source_mask):
        shutil.move(os.path.join(source_mask, file), os.path.join(final_mask, file))

shutil.rmtree('dataset-segmentation')


#sjednoceni jmen files
for tumor_type in os.listdir(os.path.join('cleaned_dataset', 'images')):
    images_path = f'cleaned_dataset/images/{tumor_type}'
    mask_path = f'cleaned_dataset/masks/{tumor_type}'
    
    def rename(path, folder_type):
        files = sorted(os.listdir(path))
        for num, file in enumerate(files):
            id = str(num).zfill(4)

            curr_filepath = os.path.join('cleaned_dataset', folder_type, tumor_type, file)
            new_filepath = os.path.join('cleaned_dataset', folder_type, tumor_type, f"{id}_{tumor_type}.jpg")
            os.rename(curr_filepath, new_filepath)

    rename(images_path, "images")
    rename(mask_path, "masks")
            
#check for duplicate images
hashes = set()
count = 0
for folder_type in os.listdir("cleaned_dataset"):
    for tumor_type in os.listdir(f"cleaned_dataset/{folder_type}"):
        for file in os.listdir(f"cleaned_dataset/{folder_type}/{tumor_type}"):
            filename = f"cleaned_dataset/{folder_type}/{tumor_type}/{file}"
            with open(filename, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in hashes:
                    hashes.add(file_hash)
                else:
                    count+=1 
                    os.remove(filename)
                    if folder_type == "masks":
                        os.remove(f"cleaned_dataset/images/{tumor_type}/{file}")
                    else:
                        os.remove(f"cleaned_dataset/masks/{tumor_type}/{file}")

for tumor_type in os.listdir(os.path.join('cleaned_dataset', 'images')):
    images_path = f'cleaned_dataset/images/{tumor_type}'
    mask_path = f'cleaned_dataset/masks/{tumor_type}'

    rename(images_path, "images")
    rename(mask_path, "masks")


def count_images(path: str)-> str:
    result = {}
    for type_folder in os.listdir(path):
        temp = []
        for tumor_folder in os.listdir(f'{path}/{type_folder}'):
            count = 0
            for file in os.listdir(f'{path}/{type_folder}/{tumor_folder}'):
                count+=1
            temp.append(list([tumor_folder,count]))
        result[type_folder] = temp

    for i in result.keys():
        print(f'{i}: {result[i]}')
        
count_images('cleaned_dataset')


#crop data to 1203
crop_num = 1202
tumors_to_crop = ['pituitary', 'meningioma']
for tumor in tumors_to_crop:
    for file in os.listdir(f'cleaned_dataset/images/{tumor}/'):
        id = file[:4]
        if int(id) > crop_num:
            file_name = f"{id}_{tumor}.jpg"
            os.remove(f'cleaned_dataset/images/{tumor}/{file_name}')
            os.remove(f'cleaned_dataset/masks/{tumor}/{file_name}')


#split data into train,test,val 70/15/15
random.seed(12)

for tumor_type in os.listdir(os.path.join('cleaned_dataset', 'images')):
    images_path = f'cleaned_dataset/images/{tumor_type}'
    mask_path = f'cleaned_dataset/masks/{tumor_type}'

    files = [x for x in os.listdir(images_path)]
    random.shuffle(files)

    length = len(files)
    train = files[:int(0.7*length)]
    test = files[int(0.7*length):int(0.85*length)]
    val = files[int(0.85*length):]

    split_folders = zip(["train", "test", "val"], [train,test,val])
    for split_folder,file_list in list(split_folders):
        os.makedirs(f"dataset_split_segmentation/{split_folder}/images/{tumor_type}", exist_ok=True)
        os.makedirs(f"dataset_split_segmentation/{split_folder}/masks/{tumor_type}", exist_ok=True)
        'dataset_split_segmentation/train/images/glioma/0877_glioma.jpg'
        for filename in file_list:
            shutil.move(os.path.join("cleaned_dataset", "images", tumor_type, filename),os.path.join("dataset_split_segmentation",split_folder,"images",tumor_type,filename))
            shutil.move(os.path.join("cleaned_dataset", "masks", tumor_type, filename),os.path.join("dataset_split_segmentation",split_folder,"masks",tumor_type,filename))

shutil.rmtree("cleaned_dataset")


print(f"train images count: {len(os.listdir("dataset_split_segmentation/train/images/meningioma")) + len(os.listdir("dataset_split_segmentation/train/images/glioma")) + len(os.listdir("dataset_split_segmentation/train/images/pituitary")) }")
print(f"train masks count: {len(os.listdir("dataset_split_segmentation/train/masks/meningioma")) + len(os.listdir("dataset_split_segmentation/train/masks/glioma")) + len(os.listdir("dataset_split_segmentation/train/masks/pituitary")) }")
print(f"test images count: {len(os.listdir("dataset_split_segmentation/test/images/meningioma")) + len(os.listdir("dataset_split_segmentation/test/images/glioma")) + len(os.listdir("dataset_split_segmentation/test/images/pituitary")) }")
print(f"test masks count: {len(os.listdir("dataset_split_segmentation/test/masks/meningioma")) + len(os.listdir("dataset_split_segmentation/test/masks/glioma")) + len(os.listdir("dataset_split_segmentation/test/masks/pituitary")) }")
print(f"val images count: {len(os.listdir("dataset_split_segmentation/val/images/meningioma")) + len(os.listdir("dataset_split_segmentation/val/images/glioma")) + len(os.listdir("dataset_split_segmentation/val/images/pituitary")) }")
print(f"val masks count: {len(os.listdir("dataset_split_segmentation/val/masks/meningioma")) + len(os.listdir("dataset_split_segmentation/val/masks/glioma")) + len(os.listdir("dataset_split_segmentation/val/masks/pituitary")) }")
