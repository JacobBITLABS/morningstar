# # ------------------------------------------------------------------------
# # Deformable DETR
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modified from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # ------------------------------------------------------------------------

# import torch.utils.data
# from .torchvision_datasets import CocoDetection
# from .torchvision_datasets import VanillaCocoDetection
# from util.ga import View

# from .coco import build as build_coco
# from .vanilla_coco import build as vanilla_build_coco

# def get_coco_api_from_dataset(dataset, view: View = None):
#     for _ in range(10):
#         # if isinstance(dataset, torchvision.datasets.CocoDetection):
#         #     break
#         if isinstance(dataset, torch.utils.data.Subset):
#             dataset = dataset.dataset
#     # if isinstance(dataset, CocoDetection):
#     #     return dataset.coco
#     if isinstance(dataset, VanillaCocoDetection): # used in eval datasets
#         return dataset.coco
#     if isinstance(dataset, CocoDetection):
#         if view == view.GROUND:
#             return dataset.g_coco
#         elif view == view.AERIAL:
#             return dataset.a_coco


# def build_dataset(image_set, args):
#     if args.dataset_file == 'coco':
#         return build_coco(image_set, args)
#     if args.dataset_file == 'visDrone':
#         return build_coco(image_set, args)
#     if args.dataset_file == 'gamod' or args.dataset_file == 'dvd':
#         return build_coco(image_set, args)
#     if args.dataset_file == 'coco_panoptic':
#         # to avoid making panopticapi required for coco
#         from .coco_panoptic import build as build_coco_panoptic
#         return build_coco_panoptic(image_set, args)
#     raise ValueError(f'dataset {args.dataset_file} not supported')


# ###
# # Vanilla
# ###
# def build_test_dataset(image_set, args, view: View):
#      if args.dataset_file == 'gamod' or args.dataset_file == 'dvd':
#         return vanilla_build_coco(image_set, args, view)
     

# def get_coco_api_from_vanilla_dataset(dataset, view: View = None):
#     for _ in range(10):
#         # if isinstance(dataset, torchvision.datasets.CocoDetection):
#         #     break
#         if isinstance(dataset, torch.utils.data.Subset):
#             dataset = dataset.dataset

#     if isinstance(dataset, VanillaCocoDetection): # used in eval datasets
#         return dataset.coco


# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'visDrone':
        return build_coco(image_set, args)
    if args.dataset_file == 'gamod':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
