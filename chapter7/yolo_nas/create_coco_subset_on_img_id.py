import json
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np

def create_coco_subset(coco_file, output_file, image_id):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Find the image with the specified ID
    images = [img for img in coco_data['images'] if img['id'] == image_id]
    if len(images) == 0:
        print("Image with ID {} not found in the COCO file.".format(image_id))
        return
    image = images[0]

    # Find the annotations for the specified image ID
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    for anno in annotations:
        anno["segmentation"] = []

    # Create a new COCO data structure with the subset
    subset_data = {
        "info": coco_data['info'],
        "licenses": coco_data['licenses'],
        "categories": coco_data['categories'],
        "images": [image],
        "annotations": annotations,
    }

    with open(output_file, 'w') as f:
        json.dump(subset_data, f)

    print("COCO subset file created. Annotations saved to:", output_file)

# Example usage
coco_file = "export.json"
output_file = "export_filtered.json"
image_id = 526  # Replace with the desired image ID
create_coco_subset(coco_file, output_file, image_id)

"""
DRAW
"""
# DATA_DIR = '../../DATASET/SDU-GAMODv4-old/scaled_dataset/test/droneview/'
DATA_DIR = '../../DATASET/SDU-GAMODv4-old/scaled_dataset/test/groundview/'
# handle ground 
img_id = 526
anno_coco = COCO(output_file)
img_info = anno_coco.loadImgs([img_id])[0]
print("image_info: ", img_info)
img_file_name = img_info["file_name"]
ann_ids = anno_coco.getAnnIds(imgIds=[img_id], iscrowd=None)
print("no. annos: ",len( ann_ids))
anns = anno_coco.loadAnns(ann_ids)
im = Image.open(DATA_DIR + img_file_name)
plt.axis('off')
plt.imshow(np.asarray(im))
anno_coco.showAnns(anns, draw_bbox=True)

ground_save = "annotated_" + img_file_name 
plt.savefig(ground_save, bbox_inches="tight", pad_inches=0)