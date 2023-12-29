#
# Jacob Nielsen
#
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from enum import Enum
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
import random


class Modal(Enum):
    GROUND = 1
    AERIAL = 2


"""
MAYBE ViSIOn DaTASET NEEDS TO Be IMPLEMENTED SPECIALLY FOR THIS TO HAVE BOTH root_ground and root_aerial
"""

class CocoGamodDetection(VisionDataset):
    """
    CocoDetection but for multi-modal views, pairs of images.
    """

    def __init__(self, root_ground, root_aerial, annFile_ground, annFile_aerial, image_set, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoGamodDetection, self).__init__( None, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.image_set = image_set
        self.root_ground = root_ground
        self.root_aerial = root_aerial
        self.coco_ground = COCO(annFile_ground)
        self.coco_aerial = COCO(annFile_aerial)
        self.ids_ground = list(sorted(self.coco_ground.imgs.keys()))
        self.ids_aerial = list(sorted(self.coco_aerial.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        self.transform = transform
        self.transforms = transforms
        self.img_output_id = 0
        self.ids = None
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        if self.ids == None:
            self.ids = self.coco_ground.getImgIds()
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            for coco in [self.coco_ground, self.coco_aerial]:
                path = coco.loadImgs(img_id)[0]['file_name']
                
                # find name in corresponding view
                if "GroundView01" in path:
                    root = self.root_ground
                elif "groundView" in path:
                    root = self.root_ground
                elif "ground" in path:
                    root = self.root_ground
                elif "GroundView" in path:
                    root = self.root_ground
                elif "DroneView_01" in path:
                    root = self.root_aerial
                elif "drone" in path:
                    root = self.root_aerial
                elif "DroneView" in path: 
                    root = self.root_aerial
                elif "droneView" in path:
                    root = self.root_aerial
                else:
                    raise Exception("cannot cache filename: ", path )

                with open(os.path.join(root, path), 'rb') as f:
                    self.cache[path] = f.read()

    def get_image(self, path, modal: Modal):
        if modal == Modal.GROUND:
            root = self.root_ground
        else:
            # Modal.AERIAL
            root = self.root_aerial

        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco_ground = self.coco_ground
        coco_aerial = self.coco_aerial
        if self.ids == None:
            self.ids = coco_ground.getImgIds()

        FOUND_ANNOS = False
        while(not FOUND_ANNOS):
            img_id_ground = self.ids_ground[index]
            img_id_aerial = self.ids_aerial[index]
            ann_ids_ground = coco_ground.getAnnIds(imgIds=img_id_ground)
            ann_ids_aerial = coco_aerial.getAnnIds(imgIds=img_id_aerial)
            if len(ann_ids_ground) != 0 or len(ann_ids_aerial) != 0:
                FOUND_ANNOS = True
            else:
                index += 1

        target_ground = coco_ground.loadAnns(ann_ids_ground)
        target_aerial = coco_aerial.loadAnns(ann_ids_aerial)

        path_ground = coco_ground.loadImgs(img_id_ground)[0]['file_name']
        path_aerial = coco_aerial.loadImgs(img_id_aerial)[0]['file_name']

        img_ground = self.get_image(path_ground, Modal.GROUND)
        img_aerial = self.get_image(path_aerial, Modal.AERIAL)

        # mixup the two images into one
        # mixup_img, lam = self.mixup(img_aerial, img_ground)

        # apply augmentation(s)
        if self.transforms is not None:
            img_ground, target = self.transforms(img_ground, target_ground)

        if self.image_set == 'val':
            # only return aerial image, with no mixup
            return img_ground, img_aerial, target_ground, target_aerial
        else:
            return img_ground, img_aerial, target_ground, target_aerial

    def __len__(self):
        # Same number of frames in both sets
        assert(len(self.ids_ground) == len(self.ids_aerial))
        return len(self.ids_ground)

 

class CocoGamodDetection_semi(VisionDataset):
    """
    CocoDetection but for multi-modal views, pairs of images.
    """

    def __init__(self, root_ground, root_aerial, annFile_ground, annFile_aerial, image_set, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoGamodDetection_semi, self).__init__(
            None, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.image_set = image_set
        self.root_ground = root_ground
        self.root_aerial = root_aerial
        self.annFile_ground = annFile_ground
        self.annFile_aerial = annFile_aerial
        self.coco_ground = COCO(annFile_ground)
        self.coco_aerial = COCO(annFile_aerial)
        self.ids_ground = list(sorted(self.coco_ground.imgs.keys()))
        self.ids_aerial = list(sorted(self.coco_aerial.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        self.transform = transform
        self.transforms = transforms
        self.img_output_id = 0
        self.ids = list(sorted(self.coco_ground.imgs.keys()))
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        if self.ids == None:
            self.ids = self.coco_ground.getImgIds()
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            for coco in [self.coco_ground, self.coco_aerial]:
                path = coco.loadImgs(img_id)[0]['file_name']
                
                # find name in corresponding view
                if "GroundView01" in path:
                    root = self.root_ground
                elif "groundView" in path:
                    root = self.root_ground
                elif "ground" in path:
                    root = self.root_ground
                elif "GroundView" in path:
                    root = self.root_ground
                elif "DroneView_01" in path:
                    root = self.root_aerial
                elif "drone" in path:
                    root = self.root_aerial
                elif "DroneView" in path: 
                    root = self.root_aerial
                elif "droneView" in path:
                    root = self.root_aerial
                else:
                    raise Exception("cannot cache filename: ", path )

                with open(os.path.join(root, path), 'rb') as f:
                    self.cache[path] = f.read()

    def get_image(self, path, modal: Modal):
        if modal == Modal.GROUND:
            root = self.root_ground
        else:
            # Modal.AERIAL
            root = self.root_aerial
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco_ground = self.coco_ground
        coco_aerial = self.coco_aerial
        if self.ids == None:
            self.ids = coco_ground.getImgIds()

        img_id_ground = self.ids_ground[index]
        img_id_aerial = self.ids_aerial[index]
        # anno ids
        ann_ids_ground = coco_ground.getAnnIds(imgIds=img_id_ground)
        ann_ids_aerial = coco_aerial.getAnnIds(imgIds=img_id_aerial)

        # targets
        target_ground = coco_ground.loadAnns(ann_ids_ground)
        target_aerial = coco_aerial.loadAnns(ann_ids_aerial)

        # ground
        path_ground = coco_ground.loadImgs(img_id_ground)[0]['file_name']
        indicator_ground = coco_ground.loadImgs(img_id_ground)[0]['indicator']
        # labeltype_ground = coco_ground.loadImgs(img_id_ground)[0]['label_type']

        # aerial
        path_aerial = coco_aerial.loadImgs(img_id_aerial)[0]['file_name']
        indicator = coco_aerial.loadImgs(img_id_aerial)[0]['indicator']
        labeltype = coco_aerial.loadImgs(img_id_aerial)[0]['label_type']

        assert(indicator_ground == indicator)
    
        # aerial
        img_ground = self.get_image(path_ground, Modal.GROUND)
        img_aerial = self.get_image(path_aerial, Modal.AERIAL)

        # # apply augmentation(s)
        # if self.transforms is not None:
        #     img_ground, target_ground = self.transforms(
        #         img_ground, target_ground)
        #     img_aerial, target_aerial = self.transforms(
        #         img_aerial, target_aerial)

       
        if self.image_set == 'val':
            # only return aerial image, with no mixup
            return img_ground, img_aerial, target_ground, target_aerial, indicator, labeltype
        else:
            return img_ground, img_aerial, target_ground, target_aerial, indicator, labeltype

    def __len__(self):
        # Same number of frames in both sets
        assert(len(self.ids_ground) == len(self.ids_aerial))
        return len(self.ids_ground)
