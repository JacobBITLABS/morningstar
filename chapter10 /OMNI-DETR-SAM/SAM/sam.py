"""
The purpose of this module is to preprocess a dataset using SAM from META AI Labs to use the detections to prune the teachers detections in the filtering algorithm.
"""
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import random
import matplotlib.patches as patches
from typing import List
from util.gaussians import gaussian_loss_matcher_v2 
from util import box_ops
import time 
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from scipy.optimize import linear_sum_assignment
import torch

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512' #'garbage_collection_threshold:0.8,max_split_size_mb:512'

class SAM():
    """
    This class implements Segment Anything VIT to interface with OMNI DETRs filtering 
    -> Seemlessly just mangage a queue
    """
    def __init__(self, device, data_path=None) -> None:
        self.sam_checkpoint = "SAM/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = device # "cuda"
        self.sam = None
        self.resize_transform = None
        self.claimed = False

    def setup(self) -> None:
        """ Setup function to prepare SAM model and mask Generator """
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint) # try to limit the number of points per batch
        self.sam = self.sam.to(device=self.device)
        self.resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size) # resize 
    
    def claim(self):
        print("IN CLAIM")
        if not self.claimed:
            self.claimed = True
            print("Claimed: ", str(self.id), " SAM....")
            return self # claimed.
        else:
            return None

    def release(self):
        if self.claimed:
            self.claimed = False

    def show_mask(self, mask, plt, random_color=False):
        """ Function to show masks on img using plt object """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)
        
    def prepare_image(self, image, transform, device=None):
        # convert to np.array as expected
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # std. SA code:
        image = transform.apply_image(image) # transform 
        image = torch.as_tensor(image, device=device.device) 
        image = image.permute(2, 0, 1).contiguous()
        return image
    
    def prepare_image_batch(self, image, input_boxes, resize_transform, sam, save_plt=False):
        # We are running in batch-sizes of 2, so this is not an extreme penalty
        prepared_batch = []
        # convert bbox coordinates
        input_boxes = box_ops.box_cxcywh_to_xyxy(input_boxes)
            
        # # and from relative [0, 1] to absolute [0, height] coordinates
        im_h, im_w = image.size
        #im_h, im_w = image.shape[:2]
        # print("size img: ",  im_h, im_w)
        target_sizes = torch.tensor([[im_w, im_h]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        input_boxes = input_boxes * scale_fct[:, None, :] 
        
        #print("input_boxes: ", input_boxes)
        # # Calculate center coordinates for each box
        # center_x = (input_boxes[:, :, 0] + input_boxes[:, :, 2]) / 2
        # center_y = (input_boxes[:, :, 1] + input_boxes[:, :, 3]) / 2
        # # Combine center_x and center_y to get center coordinates (x, y) for each box
        # center_coordinates = torch.stack((center_x, center_y), dim=-1)

        #print("center_coordinates: ", center_coordinates[0])

        if save_plt:
            ANNO_COLOR = (221, 40, 252) 
            image_with_teacher_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_teacher_boxes)        
            input_boxes = input_boxes[0]
            for xmin, ymin, xmax, ymax in input_boxes.tolist():
                # print(xmin, ymin, xmax, ymax)
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
            # Save the modified image to a file
            image_with_teacher_boxes.save("teacher_predictions_unlabel_k.png")

        prepared_image = {
            'image': self.prepare_image(image, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(input_boxes, (im_w, im_h)), # inputs boxes of form [[x,x,x,x]]
            'original_size': (im_w, im_h),
            #'point_coords': center_coordinates[0],
            #'point_labels:': np.ones(len(center_coordinates[0]))
        }
        prepared_batch.append(prepared_image)
        return prepared_batch

    def get_masks(self, image, boxes): 
        batched_input = self.prepare_image_batch(image, boxes, self.resize_transform, self.sam)
        sam_0 = time.time()
        batched_output = self.sam(batched_input, multimask_output=False)
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        sam_1 = time.time()
        # print("Sam inference: ", sam_1-sam_0)

        # print("output:", batched_output[0].keys())
        #print(batched_output[0]['masks'])
        return batched_output[0]['masks']
    
    def find_bbox_from_boolean_mask(self, mask):
        # print(mask.size())
        indices = torch.nonzero(mask)
        # print("indices: ", indices)

        if indices.numel() > 0:
            # xmin = torch.min(indices[:, 1])
            # ymin = torch.min(indices[:, 0])
            # xmax = torch.max(indices[:, 1])
            # ymax = torch.max(indices[:, 0])
            xmin = torch.min(indices[:, 2])
            ymin = torch.min(indices[:, 1])
            xmax = torch.max(indices[:, 2])
            ymax = torch.max(indices[:, 1])
            
            # print(xmin, ymin, xmax, ymax)
            return torch.tensor([xmin, ymin, xmax, ymax]) #xmin, ymin, xmax, yma
        
    def get_boxes_from_masks(self, image, masks, save_plt=False):
        """
        Get boxes from the generated masks
        """
        # # Example usage:
        # bounding_boxes = []
        # for mask in masks:
        #     bbox = self.find_bbox_from_boolean_mask(mask)
        #     if bbox is not None:
        #         # print("adding bounding box")
        #         bounding_boxes.append(bbox)

        #bounding_boxes = torch.stack([self.find_bbox_from_boolean_mask(mask) for mask in masks])
        bounding_boxes = torch.stack([bbox for bbox in [self.find_bbox_from_boolean_mask(mask) for mask in masks] if bbox is not None])

        if save_plt:
            ANNO_COLOR = (221, 40, 252) 
            image_with_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_boxes)        
            for box in bounding_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                # print(xmin, ymin, xmax, ymax)
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
            # Save the modified image to a file
            image_with_boxes.save("SA_predictions_unlabel_k_bboxes.png")

        # print("NO. bounding boxes:", len(bounding_boxes))

        return bounding_boxes

    def box_list_xyxy_to_cxcywh(self, box_list):
        """
        NOTE: MAYBE WE SHOULD JUST STEAL THE ONE FROM UTILS/WHAT_EVER.py
        """
        # Convert the list of bounding boxes to a tensor
        boxes_tensor = torch.tensor(box_list)

        # Compute the center (cx, cy) and width-height (w, h) from (xmin, ymin, xmax, ymax)
        w = boxes_tensor[:, 2] - boxes_tensor[:, 0]
        h = boxes_tensor[:, 3] - boxes_tensor[:, 1]
        cx = boxes_tensor[:, 0] + 0.5 * w
        cy = boxes_tensor[:, 1] + 0.5 * h

        # Create a new tensor in the format (cx, cy, w, h)
        new_boxes_tensor = torch.stack((cx, cy, w, h), dim=1)

        return new_boxes_tensor

    def replaces_bboxes_with_sam_bboxes(self, image, boxes, satisfied_class, save_plt=False, use_cv=False):
        # print("BOXES")
        # print(boxes)
        # print("CLASSES")
        # print(satisfied_class)
        masks = self.get_masks(image, boxes) # get measks 

        if save_plt:
            plt.figure(figsize=(10, 10))
            image_with_segmentations = image.copy()
            plt.imshow(np.array(image_with_segmentations))
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt, random_color=True)
            plt.axis('off')
            plt.savefig('SA_unlabel_k_segmentations' + '.png', bbox_inches='tight') 
            # plt.clf()
            plt.close()

        box_from_mask_0 = time.time()
        # if use_cv:
        #     # would it be faster do do contour on one image and then trace them, instead of handling the 3D boolean tensor, where each slice is one vector
        #     new_boxes = []
            

        # else:
        new_boxes = self.get_boxes_from_masks(image, masks) # new boxes
        # print("new_boxes")
        # print(new_boxes)
        box_from_mask_1 = time.time()
        # print("Getting boxes from masks: ", box_from_mask_1-box_from_mask_0)

        # We stack the boxes to one big tensor, instead of individual tracked tensors
        # new_boxes = torch.stack(new_boxes, dim=0)
        new_boxes =  self.box_list_xyxy_to_cxcywh(new_boxes)
        new_boxes = new_boxes.to(device='cuda')
        # print("new_boxes:")
        # print(new_boxes)
        # compute cost of assigment
        cost_assigning_bbox = gaussian_loss_matcher_v2(boxes, new_boxes) # NWD
        # hungarian mathcer i,j indicies. 
        # print("cost_assigning_bbox")
        # print(cost_assigning_bbox)
        #row_ind, col_ind = linear_sum_assignment(cost)
        indices = linear_sum_assignment(cost_assigning_bbox.cpu())  
        # print("INDICES")
        # print(indices)
            
        # print("INDICES")
        # print("no assigned indices: ", len(indices[0]))
        C_idx = indices 
        C_box = boxes[indices[0]] = new_boxes[indices[1]]
        C_cls = satisfied_class[indices[0]] = satisfied_class[indices[1]]
    
        # print("C_idx: ", C_idx)
        # print("C_cls: ", C_cls )

        # print("BOXES")
        # print(boxes)

        # print("NEW BOXES:")
        # print(C)
        # replace each i in pred_boxes with j in new_boxes
        if save_plt:
            """
            Because we want to assign boxes based on cxcywh, but we plot them with xyxy
            , we need to take care of that.
            """
            # TODO: add cls.
            ANNO_COLOR = (221, 40, 252) 
            draw = ImageDraw.Draw(image)        
            for box, cls in zip(C_box, C_cls):
                cls = cls.item()
                class_dict = {1: "tram", 2: "bicyle", 3: "van", 4: "truck", 5: "bus", 6: "person", 7: "car", 8: "other", 9: "streetlight", 10: "trafficlight"}
                cx, cy, w, h = box.tolist()
                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2
                # print(xmin, ymin, xmax, ymax)
                print_text = class_dict[cls] 
                draw.text((xmin, ymin-10), print_text, fill= ANNO_COLOR)
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
            # Save the modified image to a file
            image.save("SA_predictions_unlabel_k_bboxes_assigned.png")
            # plt.clf()
            plt.close()

        
        # Now we need to convert bounding boxes from absolute [0, height] coordinates to relative [0, 1] coordinates.
        # Convert image size to a tensor
        im_h, im_w = image.size
        # Step 2: Unbind height and width values
        target_sizes = torch.tensor([[im_w, im_h]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)

        # Create the scale factor tensor
        scale_fct = torch.stack([1/img_w, 1/img_h, 1/img_w, 1/img_h], dim=1)

        # Convert the absolute bounding boxes to relative [0, 1] coordinates
        C_box = C_box * scale_fct[:, None, :]

        # print("C AFter SCALE:")
        # print(C_box)

        # remove outmost dimension after scaling
        C_box =  torch.squeeze(C_box, dim=0) 
        return C_box, C_cls, C_idx

    def replace_boxes_by_sam(self, image, satisfied_outputs, satisfied_class):
        """
        image is a PIL Image
        """
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        if len(satisfied_outputs.tolist()) > 0:
            box_outputs, C_cls, C_idx = self.replaces_bboxes_with_sam_bboxes(image, satisfied_outputs, satisfied_class)
            return box_outputs, C_cls, C_idx
        else:
            # no satisfied pseudo-boxes
            return torch.tensor([], device='cuda:0'), torch.tensor([], device='cuda:0'), (torch.tensor([], device='cuda:0'), torch.tensor([], device='cuda:0'))















# def plot_bounding_boxes(image_path, bounding_boxes, save_path=True):
#     # Load the image
#     image = Image.open(image_path)
#     img_width, img_height = image.size

#     # Create a figure and axis
#     fig, ax = plt.subplots(1)

#     # Display the image
#     ax.imshow(image)

#     # Add bounding boxes to the plot
#     for box in bounding_boxes:
#         x_min, y_min, x_max, y_max = box
#         # Convert coordinates to matplotlib format (y-axis origin at the bottom)
#         rect = patches.Rectangle((x_min, img_height - y_max), x_max - x_min, y_max - y_min,
#                                  linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#     # Set axis limits to match image dimensions
#     ax.set_xlim(0, img_width)
#     ax.set_ylim(0, img_height)

#     # Remove axes
#     ax.axis('off')

#     # Save the image with bounding boxes if a save path is provided
#     if save_path:
#         plt.savefig("here.png", bbox_inches='tight', pad_inches=0, dpi=300)

#     # Display the plot (optional)
#     # plt.show()

# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)

# def main():
#     sys.path.append("..")
#     sam_checkpoint = "sam_vit_h_4b8939.pth"
#     model_type = "vit_h"

#     device = "cuda"
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     # mask_generator = SamAutomaticMaskGenerator(sam)

#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=32,
#         pred_iou_thresh=0.60, #0.86
#         stability_score_thresh=0.92,
#         crop_n_layers=1,
#         crop_n_points_downscale_factor=2,
#         min_mask_region_area=0,  # Requires open-cv to run post-processing
#     )  

#     NUM_IMGs = 5
#     DATA_DIR = '../../../SDU-GAMODv4-old/scaled_dataset/train/droneview/'
#     no = 0
#     # lst_of_imgs = random.sample(os.listdir(DATA_DIR), NUM_IMGs)
#     lst_of_imgs  = ['scene_8_sdu_30Sec_droneView_1_000434.PNG', 'scene_12_sdu_30Sec_droneView_6_000066.PNG', 'scene_10_sdu_30Sec_droneView_3_000668.PNG', 'scene_2_Crossroad_campus_30Sec_droneView_01_000504.PNG', 'scene_5_Harbour_30Sec_droneView_5_000061.PNG']
#     for image in lst_of_imgs:
#         print("image: ", image)
#         image_path = DATA_DIR + image
#         image = cv2.imread(image_path)
#         #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # parse through generator
#         masks = mask_generator.generate(image)
#         # apply on image and save
#         print("masks")
#         print(masks)
#         print(len(masks))
#         print(masks[0].keys())

#         print("[BOXES]")
#         bbox = []
#         for mask in masks:
#             bbox.append(mask['bbox'])

#         print(bbox)
       
#         # print("image_path: ", image_path)
#         # plot_bounding_boxes(image_path, bbox)
#         plt.figure(figsize=(10,10))
#         plt.imshow(image)
#         show_anns(masks)
#         plt.axis('off')
#         plt.savefig('ex_'+ str(no)+ '.png', bbox_inches='tight') 
#         no +=1
#         plt.clf()

#         print("On Images:")
#         print(lst_of_imgs)
        
#     # To generate masks, just run generate on an image.


# main()


"""
OBJECTIVE
- does it work well on small objects in general?
- does it work in dense detections?
- How well does the scale-invariance work, both accross objects within one sample and within the same sample. 

"""





