import numpy as np
import PIL 
import os
import cv2 as cv
from torch.utils.data import Dataset


class SizeRespectingDataset(Dataset):
    def __init__(self, image_folder, masks_folder, batch_size, filenames, transforms):
        self.masks_folder = masks_folder
        self.image_folder = image_folder
        self.transforms = transforms

        self.folders = [os.listdir(f"{image_folder}horizontal/"),
                       os.listdir(f"{image_folder}vertical/"),
                       os.listdir(f"{image_folder}square/") ]
        
        images = {0: [], 1: [], 2: []}
        self.map = {0 : 'horizontal', 1: 'vertical', 2: 'square'}
        
        for idx, folder in enumerate(self.folders):
            for image in folder:
                if image[:-4] in filenames:
                    images[idx].append(image[:-4])

            images[idx] = np.array(sorted(images[idx])[:-(len(images[idx]) % batch_size)]).reshape(-1, batch_size)

        self.image_list = []
        self.types = []

        total_length = images[0].shape[0] + images[1].shape[0] + images[2].shape[0]

        idxes = {0: 0, 1: 0, 2: 0}

        for i in range(total_length):
            cur_type = np.random.choice(3)
            while idxes[cur_type] >= images[cur_type].shape[0]:
                cur_type = np.random.choice(3)
            self.image_list.append(images[cur_type][idxes[cur_type]])
            self.types += [cur_type] * batch_size
            idxes[cur_type] += 1

        self.image_list = np.concatenate(self.image_list)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):

        image_path = os.path.join(self.image_folder, self.map[self.types[index]], self.image_list[index])
        mask_path = os.path.join(self.masks_folder, self.map[self.types[index]], self.image_list[index])
        image_path = image_path + ".jpg"
        mask_path = mask_path + ".png"
        
        image = cv.imread(image_path)[..., ::-1] / 255.0
        image = image.astype(np.float32)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        res= self.transforms[self.types[index]](image=image, mask=mask)
        
        return res