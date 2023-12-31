# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import inspect
from pathlib import Path
import pdb
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
# from .torchvision_datasets import CocoDetection as TvCocoDetection
# from .torchvision_datasets import CocoDetection_semi as TvCocoDetection_semi
from .torchvision_datasets import CocoGamodDetection
from .torchvision_datasets import CocoGamodDetection_semi
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import datasets.transforms_with_record as Tr


# COCO dataset class    Only used for the Supervised/"Burn-in" stage of training
class CocoDetection(CocoGamodDetection):
    def __init__(self, img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, device, image_set, transforms, return_masks=False, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, device, image_set,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.image_set = image_set
        # self.ground_imgs = img_folder_ground
        # self.ground_coco = COCO(ann_file_ground)

    def __getitem__(self, idx):
        img_ground, img_aerial, target_ground, target_aerial = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
       
        if self.image_set is not "val":
            img, target = mixup_coco(image_id, img_aerial, img_ground, target_aerial, target_ground, (10, 2))
            # End of mixup stuff
        else: 
            # validation
            img = img_aerial
            target = target_aerial
            target = {'image_id': image_id, 'annotations': target_aerial}

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
       
        return img, target
        


# COCO dataset class    Used for semi-supervised and unsupervised training
class CocoDetection_semi(CocoGamodDetection_semi):
    def __init__(self,img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, image_set, transforms_strong, transforms_weak,      return_masks=False, cache_mode=False, local_rank=0, local_size=1):
        # img_folder: The path to the folder containing the images.
        # ann_file: The path to the COCO annotations file.
        # transforms_strong: A list of strong transformations or a callable object that applies strong augmentations to the images.
        # transforms_weak: A list of weak transformations or a callable object that applies weak augmentations to the images.
        # return_masks: A boolean indicating whether to return the mask information.
        # cache_mode: An optional boolean specifying whether to cache the dataset in memory. Defaults to False.
        # local_rank: An optional integer specifying the local rank of the process. Defaults to 0.
        # local_size: An optional integer specifying the total number of processes. Defaults to 1.
        # ground_img_folder: File path to the images that will be used for mixup augmentation
        # ground_ann_file: File path to the json file containing the COCO formatted annotations for the images that will be used for mixup augmentation

        # call the initializer of the CocoDetection_semi function from /torchvision_datasets/coco.py
        super(CocoDetection_semi, self).__init__(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, image_set, transform=None, target_transform=None, transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)

        self._transforms_strong = transforms_strong
        self._transforms_weak = transforms_weak
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.image_set = image_set

    def __getitem__(self, idx):
        img_ground, img_aerial, target_ground, target_aerial, indicator, labeltype = super(CocoDetection_semi, self).__getitem__(idx)  # Get the image, its associated annotations, indicator, and label type (can be either "fully" or "Unsup" from what I can tell) info
        image_id = self.ids[idx]
        # target = {'image_id': image_id, 'annotations': target}  # Before being processed, target is a 2 field dictionary, containing an image's id and its associated annotations

        if self.image_set is not "val":
            img, target = mixup_coco(image_id, img_aerial, img_ground, target_aerial, target_ground, (10, 2))
            # End of mixup stuff
        else: 
            # validation
            img = img_aerial
            target = target_aerial
            target = {'image_id': image_id, 'annotations': target_aerial}

        # # target_ground = coco.loadAnns(ann_ids)  # target_ground is a list of all the annotation dictionary objects as defined by ann_ids
        # img, target = mixup_coco(image_id, img_aerial, img_ground, target_aerial, target_ground, (10, 2))
        # End of mixup stuff
        img, target = self.prepare(img, target)  # Run the image and its associated annotations through the __call__ function of the ConvertCocoPolysToMask class object contained in self.prepare to get a processed target object

        if self._transforms_strong is not None:  # Apply any associated transforms to the fetched image
            record_q = {}
            record_q['OriginalImageSize'] = [img.height, img.width]
            img_q, target_q, record_q = self._transforms_strong(img, target, record_q)
        if self._transforms_weak is not None:
            record_k = {}
            record_k['OriginalImageSize'] = [img.height, img.width]
            img_k, target_k, record_k = self._transforms_weak(img, target, record_k)

        print_dictionary_info()
        return img_q, target_q, record_q, img_k, target_k, record_k, indicator, labeltype


# Only used if the final overall goal of the model involves segmentation
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


# This is the class (it's a function really) that creates the target objects/annotation dictionaries of images, given the image and its associated annotations
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        # formatted_target = json.dumps(target, indent=2)
        # print("Number of Annotations for IMAGE", target["image_id"], "\t:", len(target["annotations"]))

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # formatted_dict = json.dumps(anno[0:3], indent=2)
        # print("ANNO:", formatted_dict)

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        points = [obj["point"] for obj in anno]
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)
        # print("POINTS SHAPE:", points.shape)

        # # only in train annotation file. Fix instead of almost identical Dataset Class
        # if self.image_set == 'train':
        perspectives = [obj["perspective"] for obj in anno]  # New
        perspectives = torch.as_tensor(perspectives, dtype=torch.float32)  # float32)  # .reshape(-1, 2)  # New
        # print("PERSPECTIVES SHAPE:", perspectives.shape)

        lam = target["lambda"]
        perspective_mask = target["perspective_mask"]
        perspective_mask = perspective_mask.float()

        print("LAM: ", lam)
        print("ps_mask before:", perspective_mask)
        
        # go from binary representation to lam or 1-lam representation
        perspective_mask =  perspective_mask * lam + (1 - perspective_mask) * (1 - lam)

        print("ps_mask after: ", perspective_mask)

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
        if boxes.shape[0] != keep.shape[0]:
            print("BOXES SHAPE:", boxes.shape)
            print("KEEP SHAPE:", keep.shape)
            print("PERSPECTIVES SHAPE:", perspectives.shape)
            print("PERSPECTIVES:", perspectives[keep])
        # print("KEEP VALUE:", keep)

        boxes = boxes[keep]
        classes = classes[keep]
        points = points[keep]
        perspectives = perspectives[keep]  # New
        perspective_mask = perspective_mask[keep]  # New
        # print("PERSPECTIVES:", perspectives)
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["points"] = points
        target["perspective_mask"] = perspective_mask
        # target["perspectives"] = perspectives
        # target["lambda"] = lam
         
        # for lam, perspective in zip(lam, perspectives):
        #     lam = lam if perspective == 1 else 1-lam
        #     target["lambda"].append(lam)
        
        # print("Lambda:", target["lambda"])
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

        # for a small of number samples on objects365, the instance number > 300, so we simply only pick up the first 300 annotaitons. 300 is the query number
        if target["boxes"].shape[0] > 300:
            fields = ["boxes", "labels", "area", "points", "iscrowd"]
            for field in fields:
                target[field] = target[field][:300]
                # TODO: probably change this to have either an even distribution of aerial and ground annotations, or set the split to be equal to the lamba value used in the mixup \
                #   Or you might not have to if its only for the usecase of running this code with the objects365 dataset

        return image, target


def mixup_coco(img_id, image_aerial, image_ground, targets_aerial, targets_ground, alpha=(0.75, 0.75)): #alpha = (1,1)
    """
    Perform mixup on two images from a COCO formatted dataset and combine their associated information.

    Args:
        img_id (int): The integer id of the two images that are being mixed
        images (tuple): A tuple containing two PIL.Image objects.
        targets (tuple): A tuple containing two target annotation lists.
        alpha (tuple): Mixup parameter. The two coefficients that control the interpolation strength between images and annotations. Defaults to 0.75. (10,2) should give you a distribution spread that's focused between 0.75 and 1

    Returns:
        mixed_image (PIL.Image): The mixed image obtained by combining the input images.
        mixed_target (dictionary): The combined target annotations, along with perspective filtering mask and lambda value used in the mixup.
    """

    # Generate a random mixup ratio
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    lam = beta.sample()
    lam = torch.max(lam, 1. - lam)
    # mixup_ratio = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    # Mix the images using the mixup_ratio
    # image1, image2 = images
    image1_tensor = torch.from_numpy(np.array(image_aerial)).float()
    image2_tensor = torch.from_numpy(np.array(image_ground)).float()
    mixed_image_tensor = lam * image1_tensor + (1 - lam) * image2_tensor
    mixed_image = Image.fromarray(mixed_image_tensor.numpy().astype(np.uint8))

    # Mix the target annotations
    # target1, target2 = targets  # target is a 2 item dictionary, containing an image's id and list of associated annotations
    # if target1['image_id'] != target2['image_id']:
    #     raise ValueError("The image_id values are not the same, cannot combine dictionaries.")
    # combined_annotations = target1['annotations'] + target2['annotations']
    combined_annotations = targets_aerial + targets_ground
    perspective_mask = torch.cat((torch.ones(len(targets_aerial), dtype=torch.int), torch.zeros(len(targets_ground), dtype=torch.int)))
    mixed_target = {'image_id': img_id, 'annotations': combined_annotations, 'perspective_mask': perspective_mask, 'lambda': lam}
    # perspective mask: a tensor of ones and zeroes that's to be used to quickly filter the list of annotations to just the aerial or just the ground annotations through matrix operations
    # lambda: the lambda value used in the mixup. Passed along with the annotations for the purpose of the weighted loss implementation
    return mixed_image, mixed_target  # , mixed_indicator, mixed_label_type


# def get_image_filename(image_id, images):
#     for image in images:
#         if image['id'] == image_id:
#             return image['file_name']
#     return None
#
#
# def get_annotations_for_image(image_id, annotations):
#     image_annotations = []
#     for annotation in annotations:
#         if annotation['image_id'] == image_id:
#             image_annotations.append(annotation)
#     return image_annotations


def print_dictionary_info(input_dict):
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if id(value) == id(input_dict):
            print(name, "Information")
    temp = {}
    for key in input_dict:
        temp[key] = str(type(input_dict[key]))
    json_str = json.dumps(temp, indent=4)
    print(json_str)


# Only Used by build (for supervised learning/burn in stage)
def make_coco_transforms600(image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(i * 600 / 800) for i in scales]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1000),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1000),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# 800 pixels    Only Used by build (for supervised learning/burn in stage)
def make_coco_transforms(image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

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

    raise ValueError(f'unknown {image_set}')


# 800 pixels
def make_coco_strong_transforms_with_record(image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            Tr.RandomColorJiter(),
            Tr.RandomGrayScale(),
            Tr.RandomGaussianBlur(),
            # Tr.RandomContrast(),
            # Tr.RandomAdjustSharpness(),
            # Tr.RandomPosterize(),
            # Tr.RandomSolarize(),
            Tr.RandomSelect(
                Tr.RandomResize(scales, max_size=1333),
                Tr.Compose([
                    Tr.RandomResize([400, 500, 600]),
                    Tr.RandomSizeCrop(384, 600),
                    Tr.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
            Tr.RandomErasing1(),
            Tr.RandomErasing2(),
            Tr.RandomErasing3(),
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# 800 pixels    Only used
def make_coco_weak_transforms_with_record(image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_strong_transforms_with_record600(image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(i * 600 / 800) for i in scales]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            Tr.RandomColorJiter(),
            Tr.RandomGrayScale(),
            Tr.RandomGaussianBlur(),
            # Tr.RandomContrast(),
            # Tr.RandomAdjustSharpness(),
            # Tr.RandomPosterize(),
            # Tr.RandomSolarize(),
            Tr.RandomSelect(
                Tr.RandomResize(scales, max_size=1000),
                Tr.Compose([
                    Tr.RandomResize([400, 500, 600]),
                    Tr.RandomSizeCrop(384, 600),
                    Tr.RandomResize(scales, max_size=1000),
                ])
            ),
            normalize,
            Tr.RandomErasing1(),
            Tr.RandomErasing2(),
            Tr.RandomErasing3(),
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_weak_transforms_with_record600(image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if  args.dataset_file == "gamod" or args.dataset_file == "dvd":
        PATHS = {
            "train": (root / "scaled_dataset/train/groundview", root / "scaled_dataset/train/droneview", root / "supervised_annotations" / "ground/aligned_ids" / args.annotation_json_ground_label, root / "supervised_annotations" / "aerial/aligned_ids" / args.annotation_json_aerial_label),
            "val": (root / "scaled_dataset/val/groundview", root / "scaled_dataset/val/droneview", root / "supervised_annotations" / "ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json", root / "supervised_annotations" / "aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / 'instances_w_indicator_val2017.json'),
        }

    # img_folder, ann_file = PATHS[image_set]
    # (ground_folder, aerial, folder, anno_ground, anno_aerial)
    img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial = PATHS[image_set]

    if args.pixels == 600:
        dataset = CocoDetection(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, device=args.device, image_set=image_set, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    elif args.pixels == 800:
        # dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
        #                         cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
        raise Exception("Not implemented args.pixels == 800, coco.py")
    return dataset


def build_semi_label(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    if args.dataset_file == "gamod" or args.dataset_file == "dvd":
        PATHS = {
            "train": (root / "scaled_dataset/train/groundview", root / "scaled_dataset/train/droneview", root / "supervised_annotations" / "ground/aligned_ids" / args.annotation_json_ground_label, root / "supervised_annotations" / "aerial/aligned_ids" / args.annotation_json_aerial_label),
            "val": (root / "scaled_dataset/val/groundview", root / "scaled_dataset/val/groundview", root / "supervised_annotations" / "ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json", root / "supervised_annotations" / "aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }

    # img_folder, ann_file = PATHS[image_set]
    img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial = PATHS[image_set]
    if args.pixels == 600:
        dataset = CocoDetection_semi(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, image_set=image_set, transforms_strong=make_coco_strong_transforms_with_record600(image_set), transforms_weak=make_coco_weak_transforms_with_record600(image_set), return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    elif args.pixels == 800:
        raise Exception("args.pixels == 800 not implemented")
        # dataset = CocoDetection_semi(img_folder, ann_file, transforms_strong=make_coco_strong_transforms_with_record(image_set), transforms_weak=make_coco_weak_transforms_with_record(image_set), return_masks=args.masks,
        #                              cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    return dataset


def build_semi_unlabel(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if args.dataset_file == "gamod" or args.dataset_file == "dvd":
        PATHS = {
            # TODO: Change back to unlabel!!!!!
            # "train": (root / "unsupervised_annotations/ground", root / "unsupervised_annotations/aerial", root / "unsupervised_annotations" / args.annotation_json_ground_unlabel, root / "unsupervised_annotations" / args.annotation_json_aerial_unlabel),
            # "val": (root / "scaled_dataset/all_ground_images", root / "scaled_dataset/all_aerial_images", root / "supervised_annotations" / "ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json", root / "supervised_annotations" / "aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"),
            "train": (root / "scaled_dataset/train/groundview", root / "scaled_dataset/train/droneview", root / "supervised_annotations" / "ground/aligned_ids" / args.annotation_json_ground_label, root / "supervised_annotations" / "aerial/aligned_ids" / args.annotation_json_aerial_label),
            "val": (root / "scaled_dataset/val/groundview", root / "scaled_dataset/val/droneview", root / "supervised_annotations" / "ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json", root / "supervised_annotations" / "aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }

    # img_folder, ann_file = PATHS[image_set]
    img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial = PATHS[
        image_set]

    if args.pixels == 600:
        dataset = CocoDetection_semi(img_folder_ground, image_folder_aerial, ann_file_ground, ann_file_aerial, image_set, transforms_strong=make_coco_strong_transforms_with_record600(
            image_set), transforms_weak=make_coco_weak_transforms_with_record600(
            image_set), return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    elif args.pixels == 800:
        raise Exception("args.pixels == 800 not implemented")
        # dataset = CocoDetection_semi(img_folder, ann_file, transforms_strong=make_coco_strong_transforms_with_record(image_set), transforms_weak=make_coco_weak_transforms_with_record(image_set), return_masks=args.masks,
        #                              cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    return dataset