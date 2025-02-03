from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset, stack_dict_batched
from metrics import SegMetrics, calculate_metrics_per_structure
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice', 'precision'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")
    parser.add_argument('--epoch_model_dir', type=str, default='workdir/epoch_models', help="Directory containing the epoch models (default: 'workdir/epoch_models')")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    model = sam_model_registry[args.model_type](args).to(args.device) 

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    iou = []
    dice = []
    precision = []
    metrics_per_structure = {
    'artery': {'iou': [], 'dice': [], 'precision': []},
    'liver': {'iou': [], 'dice': [], 'precision': []},
    'stomach': {'iou': [], 'dice': [], 'precision': []},
    'vein': {'iou': [], 'dice': [], 'precision': []}
    }
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}
    # Inizializza un dizionario per accumulare le metriche per struttura
    structure_metrics_accum = {name: {metric: 0.0 for metric in args.metrics} for name in ['artery', 'liver', 'stomach', 'vein']}
    structure_counts = {name: 0 for name in structure_metrics_accum.keys()}  # Conta quante volte ogni struttura compare

    for i, batched_input in enumerate(test_pbar):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        mask_names = batched_input['label_name']
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
     
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        structure_names = [name[0].split('_')[-1].split('.')[0] for name in mask_names]
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

        if args.save_pred:
            save_masks(masks, save_path, mask_names, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]
        iou.append(test_batch_metrics[0])
        dice.append(test_batch_metrics[1])
        precision.append(test_batch_metrics[2])

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]

        # Calcola le metriche per struttura
        structure_metrics = calculate_metrics_per_structure(masks, ori_labels, structure_names, args.metrics)
        for structure, metrics in structure_metrics.items():
          metrics_per_structure[structure]['iou'].append(metrics['iou'])
          metrics_per_structure[structure]['dice'].append(metrics['dice'])
          metrics_per_structure[structure]['precision'].append(metrics['precision'])
        # Accumula le metriche per struttura
        # Aggiorna le metriche per struttura e i contatori
        for structure_name in structure_metrics_accum.keys():
          if structure_name in structure_metrics:
            # Accumula le metriche solo se presenti
            for metric, value in structure_metrics[structure_name].items():
              structure_metrics_accum[structure_name][metric] += value
            structure_counts[structure_name] += 1
          else:
            # Aggiungi un log per verificare quali strutture mancano
            print(f"Structure '{structure_name}' not found in batch {i}")


    # Calcola la media delle metriche per struttura
    average_structure_metrics = {
      name: {metric: (value / structure_counts[name]) if structure_counts[name] > 0 else 0.0
           for metric, value in metrics.items()}
      for name, metrics in structure_metrics_accum.items()
    }

    print(f"Final test metrics: {test_metrics}")
    print(f"Final metrics per structure: {average_structure_metrics}")

    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    import pickle

    metrics_dict = {
      'iou': iou,
      'dice': dice,
      'precision': precision
    }

    with open('metrics_per_image_.pkl', 'wb') as f:
      pickle.dump(metrics_dict, f)
    with open('metrics_per_structure_.pkl', 'wb') as f:
      pickle.dump(metrics_per_structure, f)

    average_loss = np.mean(test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir,f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")
    print("Metrics per structure:")
    for structure, metrics in average_structure_metrics.items():
      print(f"{structure}: {metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
