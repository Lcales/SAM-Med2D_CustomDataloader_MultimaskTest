import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, test_transforms, get_boxes_from_mask, init_point_sampling
import json
import random


class TestingDataset(Dataset):
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes the dataset.
        Args:
            data_path (str): The path to the data folder.
            image_size (int, optional): The image size for resizing. Defaults to 256.
            mode (str, optional): Mode of the dataset. Can be 'test', 'train', etc. Defaults to 'test'.
            requires_name (bool, optional): Whether to include image names in the output. Defaults to True.
            point_num (int, optional): Number of points to retrieve for the segmentation prompt. Defaults to 1.
            return_ori_mask (bool, optional): Whether to return the original mask in the output. Defaults to True.
            prompt_path (str, optional): Path to a file containing prompt information. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.requires_name = requires_name
        self.point_num = point_num
        self.prompt_path = prompt_path
        
        # Carica il JSON con i prompt se fornito
        if prompt_path:
            with open(prompt_path, "r") as f:
                self.prompt_list = json.load(f)
        else:
            self.prompt_list = {}

        # Carica il JSON che associa le immagini a più maschere
        json_file_path = os.path.join(data_path, f'image2label_{mode}.json')
        with open(json_file_path, "r") as json_file:
            self.dataset = json.load(json_file)

        # Estrai i percorsi delle immagini e delle maschere
        self.image_paths = list(self.dataset.keys())
        self.mask_paths = list(self.dataset.values())

        # Parametri di normalizzazione
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

    def __getitem__(self, index):
        image_input = {}

        # Carica e normalizza l'immagine
        image = cv2.imread(self.image_paths[index])
        image = (image - self.pixel_mean) / self.pixel_std

        # Carica tutte le maschere associate all'immagine
        mask_paths = self.mask_paths[index]
        masks = [cv2.imread(mp, 0) for mp in mask_paths]
        masks = [m / 255 if m.max() == 255 else m for m in masks]

        # Salvataggio maschere originali
        ori_masks = [mask.copy() for mask in masks]
        ori_masks = [torch.tensor(mask, dtype=torch.int64) for mask in ori_masks]

        # Controllo che ogni maschera sia binaria
        for i, m in enumerate(masks):
            assert np.array_equal(m, m.astype(bool)), f"Mask {mask_paths[i]} contains non-binary values!"
        
        # Applica le trasformazioni
        h, w = masks[0].shape
        transforms = test_transforms(self.image_size, h, w)
        augments = transforms(image=image, masks=masks)
        image = augments['image']
        masks = torch.stack([mask.clone().detach().to(dtype=torch.int64) for mask in augments['masks']])

        # Calcola i box e i punti per ogni maschera
        boxes = []
        point_coords = []
        point_labels = []
        for mask in masks:
            boxes.append(get_boxes_from_mask(mask, max_pixel=0))
            coords, labels = init_point_sampling(mask, self.point_num)
            point_coords.append(coords)
            point_labels.append(labels)

        # Organizza l'output
        image_input["image"] = image
        image_input["label"] = masks
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["original_size"] = (h, w)
        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask

        # Aggiungi il nome dell'immagine se richiesto
        if self.requires_name:
            image_input["name"] = self.image_paths[index].split('/')[-1]

        return image_input

    def __len__(self):
        return len(self.image_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, mask_num=None, point_num=1):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            point_num (int, optional): Number of points to sample. Defaults to 1.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.mask_num = mask_num

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            # Carica l'immagine e normalizzala
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except Exception as e:
            print(f"Errore durante il caricamento dell'immagine {self.image_paths[index]}: {e}")

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        masks_name_list = []
        point_coords_list, point_labels_list = [], []

        # Carica tutte le maschere associate all'immagine corrente
        for m in self.label_paths[index]:
            try:
                pre_mask = cv2.imread(m, 0)
                if pre_mask.max() == 255:
                    pre_mask = pre_mask / 255

                # Aggiungi il nome della maschera
                mask_name = m.split('/')[-1] 
                masks_name_list.append(mask_name)
                masks_list.append(pre_mask)
            except Exception as e:
                print(f"Errore durante il caricamento della maschera {m}: {e}")

        # Applica le trasformazioni alle immagini e a tutte le maschere
        augments = transforms(image=image, masks=masks_list)
        image_tensor = augments['image']
        masks_tensor = torch.stack([mask.clone().detach().to(dtype=torch.int64) for mask in augments['masks']])

        # Calcola bounding boxes e punti per ogni maschera
        for mask_tensor in masks_tensor:
            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        # Converte in tensori
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)  # Aggiungi dimensione batch
        image_input["label"] = masks_tensor.unsqueeze(1)  # Aggiungi dimensione per canale
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["label_name"] = masks_name_list

        # Aggiungi il nome dell'immagine, se richiesto
        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name

        return image_input

    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)
