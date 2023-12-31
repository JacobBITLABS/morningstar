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
from scipy.stats import beta
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
    def __init__(self, root_ground, root_aerial, annFile_ground, annFile_aerial, device, image_set, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoGamodDetection, self).__init__(None, transforms, transform, target_transform)
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
        self.device = device
        self.transform = transform
        self.transforms = transforms
        self.img_output_id = 0
        self.ids = None
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
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
        mixup_img, lam = self.mixup(img_aerial, img_ground)
        
        # apply augmentation(s)
        if self.transforms is not None:
            # img_ground, target = self.transforms(img_ground, target_ground)
            mixup_img, target = self.transforms(mixup_img, target_aerial)


        if self.image_set == 'test': 
            # only return aerial image, with no mixup
            LAM = -1
            return img_aerial, LAM, target_ground, target_aerial
        else:
            return mixup_img, lam, target_ground, target_aerial

    def __len__(self):
        # Same number of frames in both sets
        assert(len(self.ids_ground) == len(self.ids_aerial))
        return len(self.ids_ground)

    def get_lam(self, threshold_low, threshold_high, alpha=1.0):
        """ gets lam """
        lam = None
        while(lam is None):
            sample = np.random.beta(alpha, alpha) # 5,  1
            if sample >= threshold_low and sample <= threshold_high:
                lam = sample
        # print("lam: ", lam)
        return lam

    def mixup(self, input_aerial, input_ground, alpha=1.0, share_lam=False):
        """Get aerial and ground image, > return mixup-image """
        lam = self.get_lam(0.85, 1.0)  # np.random.beta(alpha, alpha) # = 0.9 # random.betavariate(5, 1)
        # a, b = 80, 10
        # print("LAM", lam)
        #output_img = Image.blend(input_aerial, input_ground, lam)
        output_img = Image.blend(input_ground, input_aerial, lam)
        # if self.img_output_id < 10:
        #     output_img.save("debug_save_imgs2/"+ str(lam) +"_output_img" + str(self.img_output_id) + ".png")
        #     self.img_output_id += 1
        return output_img, lam




    # def mixup(self, input_aerial, input_ground, alpha=1.0, share_lam=False):
    #     """Get aerial and ground image, > return mixup-image """
    #     pil_to_tensor = torchvision.transforms.PILToTensor()
    #     # input_aerial = pil_to_tensor(input_aerial)
    #     # input_ground = pil_to_tensor(input_ground)
    #     beta = torch.distributions.beta.Beta(alpha, alpha) # (1.0, 1.0)
    #     # randind = torch.randperm(input.shape[0], device=input.device)
    #     if share_lam:
    #         lam = beta.sample().to(device=self.device)
    #         lam = torch.max(lam, 1. - lam)
    #         lam_expanded = lam
    #     else:
    #         lam = beta.sample([input_aerial.dim()]) # .to(device=self.device)
    #         lam = torch.max(lam, 1. - lam)
    #         DIMENSIONS = 3
    #         lam_expanded = lam.view([-1] + [1]*(input_aerial.dim()-1))
    #     # handle img
    #     #lam_expanded = 0.8 # np.random.beta(alpha, alpha) * 
    #     #output_img = lam_expanded * input_aerial + (1. - lam_expanded) * input_ground # input[randind]
    #     lam = np.random.beta(alpha, alpha)
    #     output_img = Image.blend(input_ground, input_aerial, lam)

    #     return output_img, lam