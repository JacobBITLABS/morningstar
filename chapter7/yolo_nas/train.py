from super_gradients.training import Trainer, MultiGPUMode
from super_gradients.training.utils.ssd_utils import SSDPostPredictCallback, DefaultBoxes
from super_gradients.training.metrics import DetectionMetrics
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training import models
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

CHECKPOINT_DIR = 'results/gamod/aerial/'
trainer = Trainer(experiment_name='transfer_learning_object_detection_yolox_39_epochs-aerial-V2', ckpt_root_dir=CHECKPOINT_DIR) #, multi_gpu=MultiGPUMode.OFF)


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


def main():
    print("YOLONAS")
    # propably not needed -> we alreadu have coco format
    # generate_proxy_dataset('/content/example_data')

    dataset_params = {
    'data_dir':'../../DATASET/SDU-GAMODv4-old',
    # 'train_images_dir': 'supervised_annotations_yolov5/yolo_ground/images/train',
    # 'train_labels_dir': 'supervised_annotations_yolov5/yolo_ground/labels/train',
    # 'val_images_dir':   'supervised_annotations_yolov5/yolo_ground/images/val',
    # 'val_labels_dir': 'supervised_annotations_yolov5/yolo_ground/labels/val',

    'train_images_dir': 'supervised_annotations_yolov5/yolo_aerial/images/train',
    'train_labels_dir': 'supervised_annotations_yolov5/yolo_aerial/labels/train',
    'val_images_dir':   'supervised_annotations_yolov5/yolo_aerial/images/val',
    'val_labels_dir': 'supervised_annotations_yolov5/yolo_aerial/labels/val',
    # 'test_images_dir':'test/images',
    # 'test_labels_dir':'test/labels',
    'classes': ['tram', 'bicycle', 'van', 'truck', 'bus', 'person', 'car', 'other', 'streetlight', 'traffic_light']
    }

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    # test_data = coco_detection_yolo_format_val(
    #     dataset_params={
    #         'data_dir': dataset_params['data_dir'],
    #         'images_dir': dataset_params['test_images_dir'],
    #         'labels_dir': dataset_params['test_labels_dir'],
    #         'classes': dataset_params['classes']
    #     },
    #     dataloader_params={
    #         'batch_size':16,
    #         'num_workers':2
    #     }
    # )
    
    # direct COCO:
    # train_dataset = CustomDataset("..//content/example_data", split="train")
    # val_dataset = CustomDataset("/content/example_data", split="val")
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=DetectionCollateFN())
    # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=DetectionCollateFN())
    
    # Model Available: 
    # yolo_nas_l
    # yolo_nas_m
    # yolo_nas_s

    model = models.get('yolo_nas_l', 
                   num_classes=len(dataset_params['classes']), 
                   pretrained_weights=""
                   )
    
    train_params = {
        # ENABLING SILENT MODE
        'silent_mode': True,
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
        "max_epochs": 39, # changed to experiments
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            # NOTE: num_classes needs to be defined here
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50' # LIST:  ['PPYoloELoss/loss_cls', 'PPYoloELoss/loss_iou', 'PPYoloELoss/loss_dfl', 'PPYoloELoss/loss', 'Precision@0.50', 'Recall@0.50', 'mAP@0.50', 'F1@0.50']
    }
    
    print

    # Trainer 
    trainer.train(model=model, 
              training_params=train_params, 
              train_loader=train_data, 
              valid_loader=val_data)



if __name__=="__main__":
    main()