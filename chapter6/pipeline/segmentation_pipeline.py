import io
import json
import os
from PIL import Image
import torch
import torchvision.transforms as T
import argparse
from post_processor import PostProcessPanoptic
import numpy as np
import panopticapi
from panopticapi.utils import rgb2id
import cv2
from shapely import Polygon, MultiPolygon
import copy
import gc
from torch import nn
torch.set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panoptic', help='This makes the output file panoptic segmentation ,→ dataset. This will use less computation', default=True, action="store_true") 
    parser.add_argument('--shapely_roughness', help='decide the simplification tolerance', default=0.9)
    parser.add_argument('--input_path', help='Directory of images', default="") 
    parser.add_argument('--output_path', help='Location to write json file to', default="") 
    args = parser.parse_args()
    
    coco = {
            "info": {
                "description": "Automated Annotation Pipeline",
                "url": "http://hello.org",
                "version": "1.0",
                "year": 2023,
                "contributor": "John Doe",
                "date_created": "2017/09/01"
            },
            "licenses": [
                {
            } ],
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "No License",
            "images": [],
            "annotations": [],
            "categories": [] # insert contents from pano_class.json
    }

    #Add the categories
    pano_classes = open('pano_class.json', encoding='utf-8')
    pano_classes_dict = json.load(pano_classes)
    coco["categories"] = pano_classes_dict

    #Define output path
    output_path = args.output_path
    output_filename = "test.json"


    #Save coco file bcs of memory usage
    with open(output_path + output_filename, "w") as write_file:
        json.dump(coco, write_file, indent=4)
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #Important values by detr people ,→ based on the dataset.
    ])
    #Get the pretrained model with custom postprocessor
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=False, num_classes=250)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    is_thing_map = {i: i <= 90 for i in range(250)}
    postprocessor = PostProcessPanoptic(is_thing_map, threshold=0.85)
    postprocessor.to(device)
    model.eval()

    #Load the images one-by-one
    path_of_the_directory= args.input_path
    annotation_id = 0
    image_id = 0
    for filename in  os.listdir(path_of_the_directory):
        if image_id % 1000 == 1:
            cocofile = open(output_path + output_filename, encoding='utf-8')
            coco = json.load(cocofile)
        image_filename = os.path.join(path_of_the_directory, filename)
        im = Image.open(image_filename).convert(mode="RGB") # To ensure correct format 
        ori_w, ori_h = im.size

        if image_id % 1000 == 0:
                print("current image_id: " + str(image_id))

        img = transform(im).unsqueeze(0)
        img = img.to(device)

        #Get the predictions
        with torch.no_grad():
            out = model(img)
        print(img.shape)
        print(torch.as_tensor(img.shape[-2:]).unsqueeze(0))
        print(out)

        # the post-processor expects as input the target size of the predictions (which is here set ,→ to the image size)
        result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0).to(device))[0]
        del img
        del out

        #Fixing the output map
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))
        final_w, final_h = panoptic_seg.size
        panoptic_seg = T.Resize((ori_h, ori_w))(panoptic_seg)
        #Adding information about the image to cocojson
        anno_image = {
        "id": int(image_id), # current image_id in class "width": ori_w,
        "height": ori_h,
        "file_name": os.path.basename(image_filename)
                }
        coco["images"].append(anno_image)
        #Panoptic segmentation
        if not args.panoptic:
            panoptic_seg.save(output_path + "/panoptic_segmentations/" + os.path.basename(image_filename))
            
            scl_factor = ori_h / final_h
            for e in result["segments_info"]:
                e['area'] = int(scl_factor * e['area'])
                e['bbox'] = [int(scl_factor * c) for c in e['bbox']]
                #Adding information about the segmentations to cocojson
                anno_annotations = {
                    "file_name": os.path.basename(image_filename),
                    "image_id": int(image_id),
                    "segments_info": result["segments_info"]
                }
                coco["annotations"].append(anno_annotations)
        #Instance segmentation
        else:
        #Preparing image
            panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
            panoptic_seg_id = torch.from_numpy(rgb2id(panoptic_seg))
            for e in result["segments_info"]:
                if not e['isthing']:
                    continue
                id = e['id']
                #Create the binary mask
                panoptic_seg[:, :, :] = 0
                panoptic_seg[panoptic_seg_id == id] = 255

                #Make the image better for detection
                img_gray = cv2.cvtColor(panoptic_seg, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("before" + str(id) + ".png", img_gray)
                blur = cv2.GaussianBlur(img_gray,(13,13),0)
                thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
                cv2.imwrite("after" + str(id) + ".png", thresh)

                #Find contours in the object
                contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = []

                # remove 'blob'
                for c in contours2:
                    if cv2.contourArea(c) < 1 :
                        continue
                    contours.append(c)

                #Convert to shapely
                contours = map(np.squeeze, contours) # removing redundant dimensions
                polygons = map(Polygon, contours) # converting to Polygons polygons2 = copy.deepcopy(polygons)
                polygons2 = MultiPolygon(polygons2)
                if polygons2.is_empty:
                    continue
                min_x, min_y, max_x, max_y = polygons2.bounds
                bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]
                segmentations = []

                # simply. This might work as a regulizer!
                for p in polygons:
                    simple_poly = p.simplify(args.shapely_roughness, preserve_topology=False) #decide how rough it should be
                if type(simple_poly) == Polygon:
                        segmentation = np.array(simple_poly.exterior.coords).ravel().tolist()
                        if len(segmentation) != 0:
                            segmentations.append(segmentation)
                else:
                    for poly in simple_poly.geoms:
                        #Add to json
                        segmentation = np.array(poly.exterior.coords).ravel().tolist()
                        if len(segmentation) != 0:
                            segmentations.append(segmentation)
                        annotation = {
                            "id" : int(annotation_id),
                            "image_id" : int(image_id),
                            "category_id" : int(e['category_id']),
                            "area" : bbox[2] * bbox[3],
                            "bbox" : bbox,
                            "iscrowd" : 0
                        }
                        if len(segmentations) > 0:
                            annotation["segmentation"] = segmentations
                        coco["annotations"].append(annotation)
                        annotation_id += 1

        if image_id % 1000 == 0:
            with open(output_path + output_filename, "w") as write_file:
                                json.dump(coco, write_file, indent=4)
        del result
        torch.cuda.empty_cache()
        image_id += 1
        
        with open(output_path + output_filename, "w") as write_file:
                    json.dump(coco, write_file, indent=4)
                    
if __name__ == '__main__':
    print("DETR+Pano Instance Segmentation Pipeline")
    main()
            