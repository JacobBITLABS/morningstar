"""
The purpose of this module is to preprocess a dataset using SAM from META AI Labs to use the detections to prune the teachers detections in the filtering algorithm.
"""
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import random
import matplotlib.patches as patches
from PIL import Image



def plot_bounding_boxes(image_path, bounding_boxes, save_path=True):
    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add bounding boxes to the plot
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        # Convert coordinates to matplotlib format (y-axis origin at the bottom)
        rect = patches.Rectangle((x_min, img_height - y_max), x_max - x_min, y_max - y_min,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Set axis limits to match image dimensions
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)

    # Remove axes
    ax.axis('off')

    # Save the image with bounding boxes if a save path is provided
    if save_path:
        plt.savefig("here.png", bbox_inches='tight', pad_inches=0, dpi=300)

    # Display the plot (optional)
    # plt.show()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main():
    sys.path.append("..")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # mask_generator = SamAutomaticMaskGenerator(sam)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.60, #0.86
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=10,  # Requires open-cv to run post-processing
    )  

    NUM_IMGs = 5
    DATA_DIR = '../../../SDU-GAMODv4-old/scaled_dataset/train/droneview/'
    no = 0
    # lst_of_imgs = random.sample(os.listdir(DATA_DIR), NUM_IMGs)
    lst_of_imgs  = ['scene_8_sdu_30Sec_droneView_1_000434.PNG', 'scene_12_sdu_30Sec_droneView_6_000066.PNG', 'scene_10_sdu_30Sec_droneView_3_000668.PNG', 'scene_2_Crossroad_campus_30Sec_droneView_01_000504.PNG', 'scene_5_Harbour_30Sec_droneView_5_000061.PNG']
    for image in lst_of_imgs:
        print("image: ", image)
        image_path = DATA_DIR + image
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # parse through generator
        masks = mask_generator.generate(image)
        # apply on image and save
        print("masks")
        print(masks)
        print(len(masks))
        print(masks[0].keys())

        print("[BOXES]")
        bbox = []
        for mask in masks:
            bbox.append(mask['bbox'])

        print(bbox)
       
        # print("image_path: ", image_path)
        # plot_bounding_boxes(image_path, bbox)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig('ex_'+ str(no)+ '.png', bbox_inches='tight') 
        no +=1
        plt.clf()

        print("On Images:")
        print(lst_of_imgs)

    # mask_generator_2 = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )  
    # masks2 = mask_generator_2.generate(image)
    
    # To generate masks, just run generate on an image.


main()


"""
OBJECTIVE
- does it work well on small objects in general?
- does it work in dense detections?
- How well does the scale-invariance work, both accross objects within one sample and within the same sample. 

"""