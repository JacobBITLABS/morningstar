from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    print("EVALUATION")

    #initialize COCO ground truth api
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    
    #gt_annFile = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/ground/aligned_ids/ground_test_aligned_ids.json"
    # gt_annFile = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json"
    # gt_annFile = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/aerial/aligned_ids/aerial_test_aligned_ids.json"
    gt_annFile = "../../DATASET/SDU-GAMODv4-old/supervised_annotations/aerial/aligned_ids/aerial_valid_aligned_ids_w_indicator.json"
    cocoGt=COCO(gt_annFile)
    # resFile = "predictions_coco_annotations.json"
    resFile = "predictions_coco_annotations_res_format.json"
    cocoDt=cocoGt.loadRes(resFile)
    
    imgIds=sorted(cocoGt.getImgIds())
   
   # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

main()