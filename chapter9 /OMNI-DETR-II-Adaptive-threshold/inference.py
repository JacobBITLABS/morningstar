# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# https://github.com/ahmed-nady/Deformable-DETR

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T
from pycocotools.coco import COCO

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import time

def draw_obj(image_file, text, vects):
    """Draw a border around the image using the label and vector list."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_file)
    w = im.width
    h = im.height
    draw = ImageDraw.Draw(im)
    fnt = ImageFont.truetype('Verdana.ttf', 40)
    draw.text(
        (vects[0].x * w, vects[0].y*h),
        text,
        font=fnt,
        fill=(255,255,0,255))
    print('1111')
    draw.polygon([
        vects[0].x * w, vects[0].y * h,
        vects[1].x * w, vects[1].y * h,
        vects[2].x * w, vects[2].y * h,
        vects[3].x * w, vects[3].y * h], None, 'red')
    im.save(image_file, 'JPEG')
    
def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='visDrone')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--img_path', type=str, help='input image file for inference')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    #RESUME_PATH = './exps/visDrone_multiscale_600/r50_deformable_detr_multi_scale/checkpoint0049.pth'
    RESUME_PATH = './results/r50_omni_visdrone_10_percent_11_class/checkpoint0149.pth'
    # checkpoint = torch.load(args.resume, map_location='cpu')
    checkpoint = torch.load(RESUME_PATH, map_location='cpu')
    print(checkpoint.keys())

    model.load_state_dict(checkpoint['model_teacher'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    t0 = time.time()
    DATA_DIR = '../../dataset/visDrone2022_det/scaled_label_test/'
    IMG_NAME_LIST = ['0000006_01111_d_0000003.jpg', '0000006_07596_d_0000020.jpg', '9999986_00000_d_0000041.jpg', '9999980_00000_d_0000001.jpg']


    for IMG_NAME in IMG_NAME_LIST:
        # im = Image.open(args.img_path)
        im = Image.open(DATA_DIR+IMG_NAME)
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        img=img.cuda()
        # propagate through the model
        outputs = model(img)

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # print("*******")
        # print(scores)
        keep = scores[0] > 0.20
        boxes = boxes[0, keep]
        labels = labels[0, keep]
        scores = scores[0, keep] # probabilities

        # and from relative [0, 1] to absolute [0, height] coordinates
        im_h,im_w = im.size
        #print('im_h,im_w',im_h,im_w)
        target_sizes =torch.tensor([[im_w,im_h]])
        target_sizes =target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        print(time.time()-t0)

        #read file
        dataDir='../../dataset/annotations/'
        dataType = 'scaled_visDrone_label_train_w_indicator' #cocoFIle
        annFile='{}{}.json'.format(dataDir,dataType)

        # Initialize the COCO api for instance annotations
        coco=COCO(annFile)

        # Load the categories in a variable
        catIDs = coco.getCatIds()
        categories = coco.loadCats(catIDs)
        # print(categories)

        # plot_results
        # source_img = Image.open(args.img_path).convert("RGBA")

        source_img = Image.open(DATA_DIR+IMG_NAME).convert("RGBA")
        draw = ImageDraw.Draw(source_img)

        ANNO_COLOR = (228,26,28) #(221, 40, 252)#(12, 236, 240) #(111, 0, 255)
        # get a font
        # FNT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        
        # print("Boxes:",boxes,boxes.tolist())
        index = 0
        for xmin, ymin, xmax, ymax in boxes[0].tolist():
            category_label_id =  labels[index].item()
            # find the corresponding category name:
            category = coco.loadCats(category_label_id)
            # print("CAT", category)
            category_label_name = str(category[0]['name'])

            probability_score = format(scores[index].item()*100, ".2f") # softmax
            print_text = str(category_label_name) + ',' + str(probability_score)
            # draw text
            draw.text((xmin, ymin), print_text, fill= ANNO_COLOR)
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
            index += 1


        source_img.save(IMG_NAME+'_test.png', "png")
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        #print("Outputs",results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)