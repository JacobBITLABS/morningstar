# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import PIL

# from .torchvision_datasets import CocoDetection as TvCocoDetection
from .torchvision_datasets import CocoGamodDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(CocoGamodDetection):
    def __init__(self, img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, device, image_set, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, device, image_set,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.image_set = image_set
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        mixup_img, lam, target_ground, target_aerial = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx] 

        target_ground = {'image_id': image_id, 'annotations': target_ground}
        target_aerial = {'image_id': image_id, 'annotations': target_aerial}

        mixup_img, target_ground = self.prepare(mixup_img, target_ground)
        _ , target_aerial = self.prepare(mixup_img, target_aerial)

        # After here the image is a PIL Image
        if self._transforms is not None:
            # mixup_img, target_ground = self._transforms(mixup_img, target_ground)
            mixup_img, target_aerial = self._transforms(mixup_img, target_aerial)
        
        return mixup_img, lam, target_ground, target_aerial

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep] 
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target

# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         rles = coco_mask.frPyObjects(polygons, height, width)
#         mask = coco_mask.decode(rles)
#         if len(mask.shape) < 3:
#             mask = mask[..., None]
#         mask = torch.as_tensor(mask, dtype=torch.uint8)
#         mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    
    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
          "train": (root / 'scaled_dataset' / 'train' / 'groundview', root / 'scaled_dataset' / 'train' / 'droneview', root / 'supervised_annotations' / 'ground' / 'aligned_ids' / 'ground_train_aligned_ids_w_indicator.json',  root / 'supervised_annotations' / 'aerial' / 'aligned_ids' / 'aerial_train_aligned_ids_w_indicator.json'),

        "val": (root / 'scaled_dataset' / 'val' / 'groundview', root / 'scaled_dataset' / 'val' / 'droneview', root / 'supervised_annotations' / 'ground' / 'aligned_ids' / 'ground_valid_aligned_ids_w_indicator.json', root / 'supervised_annotations' / 'aerial' / 'aligned_ids' / 'aerial_valid_aligned_ids_w_indicator.json'),
       
       "test": (root / 'scaled_dataset' / 'test' / 'groundview', root / 'scaled_dataset' / 'test' / 'droneview', root / 'supervised_annotations' / 'ground' / 'aligned_ids' / 'ground_test_aligned_ids.json', root / 'supervised_annotations' / 'aerial' / 'aligned_ids' / 'aerial_test_aligned_ids.json'),
    }

    #img_folder, ann_file = PATHS[image_set]
    img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial = PATHS[image_set] # (ground_folder, aerial, folder, anno_ground, anno_aerial)
    
    dataset = CocoDetection(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, device=args.device, image_set = image_set, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


