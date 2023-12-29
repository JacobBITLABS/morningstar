# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, image_set, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        self.image_set = image_set
        if cache_mode:
            self.cache = {}
            self.cache_images()


    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            root = '../../SDU-GAMODv3/'
            if "droneView" in path:
                joined_path = root + "/scaled_dataset/" + "all_aerial_images/" + path
            elif "drone" in path:
                joined_path = root + "/scaled_dataset/" + "all_aerial_images/" + path
            elif "groundView" in path:
                joined_path = root + "/scaled_dataset/" + "all_ground_images/" + path
            elif "ground" in path:
                joined_path = root + "/scaled_dataset/" + "all_ground_images/" + path

            # with open(os.path.join(self.root, path), 'rb') as f:
            #     self.cache[path] = f.read()
            with open(joined_path, 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path, override_path=False):
        # os.path.join(self.root, path)
        if override_path:
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(path, 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(path).convert('RGB')

        else:
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        #print("SELF:IDS", len(self.ids))

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        
        # root = '../../SDU-GAMODv4/'
        # if "droneView" in path:
        #     joined_path = root + "/scaled_dataset/" + "train/droneview/" + path
        # elif "drone" in path:
        #     joined_path = root + "/scaled_dataset/" + "train/droneview/" + path
        # elif "groundView" in path:
        #     joined_path = root + "/scaled_dataset/" + "train/groundview/" + path
        # elif "ground" in path:
        #     joined_path = root + "/scaled_dataset/" + "train/groundview/" + path

        # if self.imageset == 'train':
        #     root = '../../SDU-GAMODv4-old'
        #     if "droneView" in path:
        #         joined_path = root + "/scaled_dataset/" + "train/droneview/" + path
        #     elif "drone" in path:
        #         joined_path = root + "/scaled_dataset/" + "train/droneview/" + path
        #     elif "groundView" in path:
        #         joined_path = root + "/scaled_dataset/" + "train/groundview/" + path
        #     elif "ground" in path:
        #         joined_path = root + "/scaled_dataset/" + "train/groundview/" + path

        #     # print("joined_path: ", joined_path)
        #     img = self.get_image(joined_path, override_path=True)
        # else:
        #     img = self.get_image(path, override_path=False)
        # print("joined_path: ", joined_path)

        img = self.get_image(path, override_path=False)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
