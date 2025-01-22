import torch
import numpy as np
import cv2

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()


def precision(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    true_positives = torch.sum(gt_ * pr_, dim=[1, 2, 3])  
    false_positives = torch.sum((1 - gt_) * pr_, dim=[1, 2, 3])  
    precision_value = (true_positives + eps) / (true_positives + false_positives + eps)
    return precision_value.cpu().numpy()



def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric

def calculate_metrics_per_structure(pred, label, structure_names, metrics):
    assert pred.shape == label.shape, "Predizioni e label devono avere la stessa forma."
    num_structures = len(structure_names)

    results = {name: {metric: 0.0 for metric in metrics} for name in structure_names}

    # Debug: stampa le dimensioni delle maschere
    print(f"Predicted masks shape: {pred.shape}")
    print(f"Original labels shape: {label.shape}")

    for i, structure_name in enumerate(structure_names):
        structure_pred = pred[i:i + 1]  # Seleziona la maschera della struttura
        structure_label = label[i:i + 1]


        for metric in metrics:
            if metric == 'iou':
                # Calcolo dell'IoU
                iou_value = float(iou(structure_pred, structure_label).mean())
                results[structure_name][metric] = iou_value

            elif metric == 'dice':
                # Calcolo del Dice
                dice_value = float(dice(structure_pred, structure_label).mean())
                results[structure_name][metric] = dice_value

            elif metric == 'precision':
                # Calcolo della Precision
                precision_value = float(precision(structure_pred, structure_label).mean())
                results[structure_name][metric] = precision_value

            else:
                raise ValueError(f"Metric '{metric}' non riconosciuta.")
  
    return results
