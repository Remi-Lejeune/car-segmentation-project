import random
import torchvision.transforms.functional as TF


def DataAugmentation(data, pflip=0.2, pcrop=0.2, min_crop_sz=200, prot=0.2, max_rot_ang=30):
    # Randomly flip images horizontally
    im_num = len(data)
    for i in range(im_num):
        if random.random() < pflip:
            trans_im = TF.hflip(data[i])
            data.append(trans_im)

    # Crop images at random
    im_num = len(data)
    for i in range(im_num):
        if random.random() < pcrop:
            crop_sz = random.uniform(a=min_crop_sz, b=data.shape[2])
            cropper = TF.RandomResizedCrop(size=(crop_sz, crop_sz))
            trans_im = cropper(data[i])
            data.append(trans_im)

    # Rotate images at random
    im_num = len(data)
    for i in range(im_num):
        if random.random() < prot:
            angle = random.uniform(a=-max_rot_ang, b=max_rot_ang)
            trans_im = TF.rotate(data[i], angle=angle)
            data.append(trans_im)

    return data
