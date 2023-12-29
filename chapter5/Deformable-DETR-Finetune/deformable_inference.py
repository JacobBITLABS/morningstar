# ------------------------------------------------------------------------
# Copyright Jacob Nielsen 2023
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# 
# ------------------------------------------------------------------------
# Modfied from https://github.com/ahmed-nady/Deformable-DETR
# ------------------------------------------------------------------------

import argparse
import random
import time
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as T
from pycocotools.coco import COCO
import numpy as np
import torch
import util.misc as utils
from util import box_ops
from models import build_model
    
def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector Inference', add_help=False)
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
    parser.add_argument('--nheads', default=16, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=900, type=int,
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

    # construct model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # loading checkpoint
    resume_path = args.inference_resume
    checkpoint = torch.load(resume_path, map_location='cpu')
    # print(checkpoint.keys())

    # loading model from checkpoint
    model.load_state_dict(checkpoint['model'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval() # eval mode -> node gradient calc.

    # Initialize the COCO api for instance annotations
    coco=COCO(args.coco_file)

    IMG_NAME_LIST = []

    if len(args.img_name_list) == 0:
        for i in range(args.num_random_samples):
            # Get image ID at random
            img_ids = coco.getImgIds()
            # pick at random 
            rand_id = random.randrange(1, (len(img_ids)))
            image = coco.loadImgs(ids=[rand_id])[0]
            #sample = random.choice(os.listdir(DATA_DIR)) #change dir name to whatever
            IMG_NAME_LIST.append(image)
    else:
        # add specific images:
        image = coco.loadImgs(ids=args.inference_img_ids)[0]
        IMG_NAME_LIST.append(image)
        
        img_ids = coco.getImgIds()
        for img_id in img_ids:
            image = coco.loadImgs(ids=[img_id])[0]
            IMG_NAME_LIST.append(image)

    for IMG in IMG_NAME_LIST:
        print("[IMG]: ", IMG)
        im = Image.open(args.data_dir+IMG['file_name']) # PIL Image
        img = transform(im).unsqueeze(0)                # apply tranformations, size, normalization etc.
        img=img.cuda()                                  # send to GPU
        # Through the model
        outputs = model(img)
        # extract outputs
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # extract topK
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), args.topK, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # Threshold predictions - finetune this for your dataset 
        keep = scores > args.keep_threshold
        boxes = boxes[keep]    # boxes (xyxy)
        labels = labels[keep]  # class labels
        scores = scores[keep]  # probabilities

        # and from relative [0, 1] to absolute [0, height] coordinates
        im_h, im_w = im.size
        target_sizes = torch.tensor([[im_w,im_h]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        print("No. boxes: ", len(boxes[0]))
        print("No. scores ", len(scores))
        print("No. labels ", len(labels))

        # Draw Predictions 
        source_img = Image.open(args.data_dir+IMG['file_name']).convert("RGBA")
        draw = ImageDraw.Draw(source_img)
        ANNO_COLOR = args.annotation_color
        index = 0
        for xmin, ymin, xmax, ymax in boxes[0].tolist():
            category_label_id =  labels[index].item()
            # find the corresponding category name:
            if category_label_id != 0:
                category = coco.loadCats(category_label_id)
                category_label_name = str(category[0]['name'])

                probability_score = format(scores[index].item()*100, ".2f") # softmax
                print_text = str(category_label_name) + ': ' + str(probability_score)
                print( str(category_label_name), " with prob.: ", probability_score)
                # draw text
                draw.text((xmin, ymin-10), print_text, fill= ANNO_COLOR)
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
                index += 1
            else: 
                # draw.text((xmin, ymin), 'No-obj', fill= ANNO_COLOR)
                # draw.rectangle(z((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
                print("No object")
            
            if args.draw_ground_truth:
                index = 0
                anno_ids = coco.getAnnIds(imgIds=[IMG['id']])
                gt_annos = coco.loadAnns(ids=anno_ids)
                gt_boxes = []
                gt_labels = []
                for gt_anno in gt_annos:
                    bbox = gt_anno['bbox']
                    gt_label = gt_anno['category_id']
                    gt_boxes.append(bbox)
                    gt_labels.append(gt_label)

                GT_ANNO_COLOR = args.gt_annotation_color
                for xmin, ymin, xmax, ymax in gt_boxes:
                    category_label_id =  gt_labels[index]
                    # # find the corresponding category name:
                    if category_label_id != 0:
                        category = coco.loadCats(category_label_id)
                        category_label_name = str(category[0]['name'])
                        print_text = str(category_label_name)
                        draw.text((xmin, ymin+10), print_text, fill= GT_ANNO_COLOR)
                        draw.rectangle(((xmin, ymin), (xmin+xmax, ymin+ymax)), outline = GT_ANNO_COLOR)
                        index += 1

            print("[Saving Image...]")
            source_img.save(IMG['file_name']+'_inference.png', "png")
            # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Inference Script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Inference parameters
    parser.add_argument('--draw_ground_truth', default=True, action='store_true', help='draw ground truth annos')
    parser.add_argument('--inference_resume', default='../Deformable-DETR-Finetune/results/gamod/ground/checkpoint0039.pth', help='resume from checkpoint')
    parser.add_argument('--data_dir', default='../../DATASET/SDU-GAMODv4-old/scaled_dataset/val/groundview/', help='directory with inference images')
    parser.add_argument('--coco_file', default='../../DATASET/SDU-GAMODv4-old/supervised_annotations/ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json', help='COCO formatted annotation file')
    parser.add_argument('--annotation_color', default=(221, 40, 252), help='BBox color, predictions')
    parser.add_argument('--gt_annotation_color', default=(55, 126, 184), help='BBox color, ground truth')
    parser.add_argument('--keep_threshold', default=0.00, type=float, help='Filter output predictions on confidence score')
    parser.add_argument('--topK', default=40 type=int, help='get topK predictions from model output')

    # decide if you simply want N random images
    parser.add_argument('--num_random_samples', default=3, type=int, help='Filter output predictions on confidence score')

    IMG_IDS = [] # [915]
    parser.add_argument('--inference_img_ids', default=IMG_IDS, help='List with IMG ids to run inference on. If empty, random samples is used')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)