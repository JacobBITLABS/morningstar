import fiftyone as fo
from super_gradients.training import models
import numpy as np
from fiftyone import ViewField as F

def convert_bboxes(bboxes, w, h):
    tmp = np.copy(bboxes[:, 1])
    bboxes[:, 1] = h - bboxes[:, 3]
    bboxes[:, 3] = h - tmp
    bboxes[:, 0]/= w
    bboxes[:, 2]/= w
    bboxes[:, 1]/= h
    bboxes[:, 3]/= h
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    bboxes[:, 1] = 1 - (bboxes[:, 1] + bboxes[:, 3])
    return bboxes

def generate_detections(p, width, height):
    class_names = p.class_names
    dp = p.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    if 0 in bboxes.shape:
        return fo.Detections(detections = [])
    
    bboxes = convert_bboxes(bboxes, width, height)
    labels = [class_names[l] for l in labels]
    detections = [
        fo.Detection(
            label = l,
            confidence = c,
            bounding_box = b
        )
        for (l, c, b) in zip(labels, confs, bboxes)
    ]
    return fo.Detections(detections=detections)


#labels_path = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"
#labels_path = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/aerial/aligned_ids/aerial_test_aligned_ids.json"
#labels_path = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"
""" GROUND"""
#labels_path = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json"
labels_path = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/ground/aligned_ids/ground_test_aligned_ids.json"
data_path = "../../DATASET/SDU-GAMODv4-old/scaled_dataset/test/groundview"
#data_path = "../../DATASET/SDU-GAMODv4-old/scaled_dataset/val/groundview"

for dataset in fo.list_datasets():
    dataset = fo.load_dataset(dataset)
    dataset.delete()

name = "YOLO-NAS"
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    # max_samples=20,
    name=name,
)

# import fiftyone.zoo as foz
# dataset = foz.load_zoo_dataset(
#     "coco-2017", 
#     split = "validation", 
#     max_samples= 50
# )
dataset.compute_metadata()

# Print some information about the dataset
# print(dataset)

# Print a ground truth detection
sample = dataset.first()
print(sample)

# for sample in dataset:
#     print(sample)
#     sample["ground_truth"] = sample["detections"]
#     sample.save()

model_path = 'results/gamod/ground/transfer_learning_object_detection_yolox_39_epochs/ckpt_best.pth'
#model_path = 'results/gamod/aerial/transfer_learning_object_detection_yolox_39_epochs-aerial-V2/ckpt_best.pth'# 'results/aerial/ground/transfer_learning_object_detection_yolox_39_epochs/ckpt_best.pth'
model = models.get("yolo_nas_l", num_classes=10, checkpoint_path=model_path)


def add_YOLO_NAS_predictions(dataset, confidence = 0.9):
    ## aggregation to minimize expensive operations
    fps, widths, heights = dataset.values(
        ["filepath", "metadata.width", "metadata.height"]
    )
    
    ## batch predictions
    preds = model.predict(fps, conf = confidence)._images_prediction_lst
    
    ## add all predictions to dataset at once
    dets = [
        generate_detections(pred, w, h)
        for pred, w, h in zip(preds, widths, heights)
    ]
    dataset.set_values("YOLO-NAS", dets)
    #classes = dataset.distinct("detections.label")
    
    # "predictions" = "YOLO-NAS"
    res = dataset.evaluate_detections("YOLO-NAS", gt_field="detections", method = "coco", eval_key="eval_yolonas", compute_mAP=True)
    print("resulting mAP: ", res.mAP())

    classes = ["tram", "bicycle", "van", "truck", "bus", "person", "car", "other", "streetlight", "traffic_light"]
    print("EXPORTING")
    dataset.export(
        dataset_type=fo.types.COCODetectionDataset,
        labels_path="export.json",
        label_field="YOLO-NAS",
        export_media=False,
        abs_paths=False,
        classes=classes,
    )
  

    # ### # ### # ### # ### # ### # ### # ### # ### # ### # #### ### # ###
    # SMALL
    #
    # Create an expression that will match objects whose bounding boxes have
    # area less than 32^2 pixels
    #
    # Bounding box format is [top-left-x, top-left-y, width, height]
    # with relative coordinates in [0, 1], so we multiply by image
    # dimensions to get pixel area
    #
    # bbox_area = (
    #     F("$metadata.width") * F("bounding_box")[2] *
    #     F("$metadata.height") * F("bounding_box")[3]
    # )
    # small_boxes = bbox_area < 32 ** 2
    # large_boxes = bbox_area > 0.7
    # small_boxes = bbox_area < 0.3

    # # Create a view that contains only small-sized objects
    # small_view = (
    #     dataset
    #     .filter_labels(
    #         "detections", 
    #         small_boxes
    #     )
    # )
    # # Create a view that contains only large-sized objects
    # large_view = (
    #     dataset
    #     .filter_labels(
    #         "detections", 
    #         large_boxes
    #     )
    # )

    # print("MEDIUM:")
    # small_res = small_view.evaluate_detections(
    #     "YOLO-NAS", #"predictions",
    #     gt_field="detections",
    #     eval_key="eval_small",
    #     compute_mAP=True,
    # )
    # print("resulting small mAP: ", small_res.mAP())

    # print("LARGE:")
    # large_res = large_view.evaluate_detections(
    #     "YOLO-NAS", #"predictions",
    #     gt_field="detections",
    #     eval_key="eval_large",
    #      compute_mAP=True,
    # )
    # print("resulting large mAP: ", large_res.mAP())

    # ### # ### # ### # ### # ### # ### # ### # ###  # ### # ### # ### # ### # ### # ### # ### # ###

    # bbox_area = (
    #     F("$metadata.width") * F("bounding_box")[2] *
    #     F("$metadata.height") * F("bounding_box")[3]
    # )
    # small_boxes =  (32 ** 2 < bbox_area)
    # medium_boxes = (32 ** 2 < bbox_area) & (bbox_area < 96 ** 2)
    # large_boxes = (bbox_area > 96 ** 2)
    
    # Create a view that contains only medium-sized objects
    # small_view = (
    #     dataset
    #     .filter_labels("detections", small_boxes)
    #     .filter_labels("YOLO-NAS", small_boxes)
    # )
    # medium_view = (
    #     dataset
    #     .filter_labels("detections", medium_boxes)
    #     .filter_labels("YOLO-NAS", medium_boxes)
    # )
    # large_view = (
    #     dataset
    #     .filter_labels("detections", large_boxes)
    #     .filter_labels("YOLO-NAS", large_boxes)
    # )
    # # Evaluate the medium-sized objects
    # small_results = small_view.evaluate_detections(
    #     "YOLO-NAS", # "predictions"
    #     gt_field="detections",
    #     eval_key="eval_small",
    #     compute_mAP=True,
    # )
    # medium_results = medium_view.evaluate_detections(
    #     "YOLO-NAS",
    #     gt_field="detections",
    #     eval_key="eval_medium",
    #     compute_mAP=True,
    # )
    # large_results = large_view.evaluate_detections(
    #     "YOLO-NAS",
    #     gt_field="detections",
    #     eval_key="eval_large",
    #     compute_mAP=True,
    # )
    # print("mAP: ", res.mAP(), " small mAP: ", small_results.mAP(), "medium mAP: ", medium_results.mAP(), " large mAP: ", large_results.mAP())
    # res.print_report()

    """
    GROUND TEST 
    mAP:  0.4119771934033902  small mAP:  0.710526627938435 medium mAP:  0.7026206870396048  large mAP:  1.0
    
      bicycle       0.25      0.34      0.29      1344
          bus       0.00      0.00      0.00        22
          car       0.57      0.81      0.67     12824
        other       0.70      0.64      0.67     11476
       person       0.26      0.37      0.31      2166
  streetlight       0.69      0.78      0.73     11611
traffic_light       0.78      0.70      0.74      2265
         tram       0.59      0.97      0.74        33
        truck       0.83      1.00      0.91       254
          van       0.38      0.74      0.50       445

    micro avg       0.60      0.71      0.66     42440
    macro avg       0.50      0.64      0.55     42440
 weighted avg       0.62      0.71      0.66     42440

    """


    # small mAP:  0.710526627938435 medium mAP:  0.7026206870396048  large mAP:  1.0
    # small_boxes = bbox_area < 32 ** 2 
    
    # medium_boxes = bbox_area > 32 ** 2 and bbox_area < 96 ** 2
    # large_boxes = bbox_area > 96 ** 2

    # Only contains boxes whose area is between 32^2 and 96^2 pixels
    # medium_boxes = dataset.filter_labels(
    #     "detections", (32 ** 2 < bbox_area) & (bbox_area < 96 ** 2)
    # )
    # small_boxes = medium_boxes

    # Create a view that contains only small GT and predicted boxes
    # small_boxes_eval_view = (
    #       dataset
    #     .filter_labels("detections", small_boxes, only_matches=False)
    #     .filter_labels("YOLO-NAS", small_boxes, only_matches=False)
    # )

    # # # Run evaluation
    # small_boxes_results = small_boxes_eval_view.evaluate_detections(
    #     "YOLO-NAS", gt_field="detections", eval_key="eval_yolonas", compute_mAP=True
    # )
    # print("resulting small mAP: ", small_boxes_results.mAP())

add_YOLO_NAS_predictions(dataset, confidence = 0.25)
session = fo.launch_app(dataset, remote=True)
session.wait()