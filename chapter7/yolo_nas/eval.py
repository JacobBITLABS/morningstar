from super_gradients.training import models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch
from super_gradients.training.utils.ssd_utils import SSDPostPredictCallback, DefaultBoxes
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from torch.utils.data import DataLoader

from dataset.coco import build as build_coco
from dataset.torchvision_datasets.coco import CocoDetection
from engine import evaluate
import util.misc as utils
from pycocotools.coco import COCO

# Anchors
def dboxes():
    figsize = 96
    feat_size = [6, 3, 2, 1]
    scales = [9, 36, 64, 91, 96]
    aspect_ratios = [[2], [2], [2], [2]]
    return DefaultBoxes(figsize, feat_size, scales, aspect_ratios)

def base_detection_collate_fn(batch):
    """
    Simple aggregation function to batched the data
    """

    images_batch, labels_batch = list(zip(*batch))
    for i, labels in enumerate(labels_batch):
        # ADD TARGET IMAGE INDEX
        labels[:, 0] = i

    return torch.stack(images_batch, 0), torch.cat(labels_batch, 0)

# creation of proxy dataset to demonstrate usage
def generate_proxy_dataset( write_path):
    # Create training files and text
    os.makedirs(os.path.join(write_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(write_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(write_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(write_path, 'labels', 'val'), exist_ok=True)
    train_fp = open(os.path.join(write_path, 'train.txt'), 'w')
    val_fp = open(os.path.join(write_path, 'val.txt'), 'w')

    for n in range(10):
        a = np.random.rand(96, 96, 3) * 255
        im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
        im_string = '%000d.jpg' % n
        im_out.save(os.path.join(write_path, 'images', 'train', im_string))
        im_out.save(os.path.join(write_path, 'images', 'val', im_string))
        train_fp.write((os.path.join(write_path,'images', 'train', im_string)) + '\n')
        val_fp.write((os.path.join(write_path, 'images', 'val', im_string)) + '\n')

        # Create label files
        train_label_fp = open(os.path.join(write_path, 'labels', 'train', im_string.replace('.jpg','.txt')), 'w')
        val_label_fp = open(os.path.join(write_path, 'labels', 'val', im_string.replace('.jpg','.txt')), 'w')
        for b in range(5):
            cls = np.random.randint(0, 7)
            loc = np.random.uniform(0.25, 0.5)
            train_label_fp.write(f'{cls} {loc - 0.1} {loc + 0.1} {loc - 0.2} {loc + 0.2}' + '\n')
            val_label_fp.write(f'{cls} {loc - 0.1} {loc + 0.1} {loc - 0.2} {loc + 0.2}' + '\n')



class CustomDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.lower()

        assert self.split in {'train', 'val'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '.txt'), 'r') as j:
            self.images = j.readlines()

    def __getitem__(self, i):
        # Read image and label
        image = Image.open(self.images[i].replace("\n",""), mode='r').resize((320, 320))
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
        labels = np.loadtxt(self.images[i].replace("jpg\n","txt").replace("images", "labels"))
        return image_tensor, labels
        

    def __len__(self):
        return len(self.images)

dataset_params = {
    'data_dir':'../../DATASET/SDU-GAMODv4-old',
    'train_images_dir': 'supervised_annotations_yolov5/yolo_aerial/images/train',
    'train_labels_dir': 'supervised_annotations_yolov5/yolo_aerial/labels/train',
    'val_images_dir':   'supervised_annotations_yolov5/yolo_aerial/images/val',
    'val_labels_dir': 'supervised_annotations_yolov5/yolo_aerial/labels/val',
    # 'test_images_dir':'test/images',
    # 'test_labels_dir':'test/labels',
    'classes': ['tram', 'bicycle', 'van', 'truck', 'bus', 'person', 'car', 'other', 'streetlight', 'traffic_light']
    }


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    

def main():
    print("EVALUATION")
    # yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
    model_path = 'results/gamod/aerial/transfer_learning_object_detection_yolox_39_epochs/ckpt_best.pth'
    model = models.get("yolo_nas_l", num_classes=10, checkpoint_path=model_path)

    device = torch.device('cuda')

    args = {"data_path" : '../../DATASET/SDU-GAMODv4-old', "dataset_file": "dvd", "dataset_file": "dvd"}
    dataset_val = build_coco(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=1,
                                 pin_memory=True)

    base_ds = get_coco_api_from_dataset(dataset_val) # GT
    coco_api = COCO("../../DATASET/SDU-GAMODv4-old/supervised_annotations/aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json")

    evaluate(model, None, None, data_loader_val, base_ds, coco_api, device, "/eval")

    # for data in val_data:
    #     print(data)
 
    # # Predict using SG model
    # with torch.no_grad():
    #     raw_predictions = model(transformed_image)
    # # Callback uses NMS, confidence threshold
    # predictions = YoloPostPredictionCallback(conf=0.1, iou=0.4)(raw_predictions)[0].numpy()

    # visualize = False
    # if visualize:
    #     # Visualize results
    #     boxes = predictions[:, 0:4]
    #     plt.figure(figsize=(10, 10))
    #     plt.plot(boxes[:, [0, 2, 2, 0, 0]].T, boxes[:, [1, 1, 3, 3, 1]].T, '.-')
    #     plt.imshow(image.resize([640, 640]))
    #     plt.show()










if __name__ == '__main__':
    main()