import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            point_num (int, optional): Number of points to sample. Defaults to 1.
            mask_num (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        # Carica il dataset
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
            # Carica e normalizza l'immagine
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(f"Errore nel caricamento dell'immagine: {self.image_paths[index]}")

        # Prendi il percorso della maschera
        mask_path = self.label_paths[index]
        pre_mask = cv2.imread(mask_path, 0)

        # Normalizza la maschera
        if pre_mask.max() == 255:
            pre_mask = pre_mask / 255

        assert np.array_equal(pre_mask, pre_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {mask_path}"

        # Trasformazioni
        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)

        # Crea le liste per maschere, box e punti
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_names_list = []  # Lista per i nomi delle maschere

        # Estrazione delle maschere
        for _ in range(self.mask_num):
            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

            # Aggiungi il nome della maschera alla lista
            mask_name = mask_path.split('/')[-1]
            mask_names_list.append(mask_name)

        # Creazione dei tensori per maschere, box e punti
        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        # Costruzione del dizionario di output
        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["mask_names"] = mask_names_list  # Aggiungi i nomi delle maschere al dizionario

        # Aggiungi il nome dell'immagine se richiesto
        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name

        return image_input

    def __len__(self):
        """
        Restituisce il numero totale di immagini nel dataset.
        """
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    """
    Funzione per impacchettare il batch in un dizionario.
    """
    out_dict = {}
    for k, v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(train_batch_sampler):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)
