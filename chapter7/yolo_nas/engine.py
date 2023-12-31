from coco_eval import CocoEvaluator
import torch
import torchvision.transforms as transforms
import numpy as np
import utils
import os
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from PIL import Image
import torchvision.transforms as T
from io import BytesIO

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, coco_api ,device, output_dir, use_pantoptic_eval=False):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = ['bbox'] # tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    #coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [.5:.05:.95]

    panoptic_evaluator = None
    if use_pantoptic_eval:
        if 'panoptic' in postprocessors.keys():
            panoptic_evaluator = PanopticEvaluator(
                data_loader.dataset.ann_file,
                data_loader.dataset.ann_folder,
                output_dir=os.path.join(output_dir, "panoptic_eval"),
            )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # This is a hack to load the image as a PIL image
        print("targets")
        print(targets)

        img_id = targets[0]["image_id"].item()
        print("image_id: ", img_id)
        path = coco_api.loadImgs(img_id)[0]['file_name']
        print("imagePath: ", path)
        file_path = '../../DATASET/SDU-GAMODv4-old/' + 'scaled_dataset/' + 'val/' + 'droneview/' + path
        print("file_path ", file_path)
        img = Image.open(file_path).convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize([640, 640]),
            transforms.PILToTensor()
        ])
        transformed_image = preprocess(img).float().unsqueeze(0)

        # transformed_image = preprocess(samples).float().unsqueeze(0) # this should happen in coco.py
        raw_predictions = model(transformed_image) # transformed image
        
        # Callback uses NMS, confidence threshold
        predictions = YoloPostPredictionCallback(conf=0.1, iou=0.4)(raw_predictions)[0].cpu().numpy()
        
        print("## Predictions ##")
        print(predictions)
        
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
