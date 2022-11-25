import glob



def photos_file_name(path='carseg_data/carseg_raw_data/train/photo/*.jpg'):
    return glob.glob(path)



def get_photo_names(photos_file_names):
    clean_data_filenames = []
    for file_name in photos_file_names:
        image_file = file_name.split('\\')[1]
        image_name = image_file.split('.')[0]
        path = glob.glob("carseg_data/clean_data/"+image_name+".npy")
        clean_data_filenames.append(path[0])
    return clean_data_filenames

print(get_photo_names(photos_file_name()))
