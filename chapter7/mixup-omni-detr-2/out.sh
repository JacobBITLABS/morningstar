+ GPUS=8
+ RUN_COMMAND=./configs/r50_ut_detr_omni_dvd.sh
+ '[' 8 -lt 8 ']'
+ GPUS_PER_NODE=8
+ MASTER_ADDR=127.0.0.1
+ MASTER_PORT=29500
+ NODE_RANK=0
+ let NNODES=GPUS/GPUS_PER_NODE
+ python ./tools/launch.py --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 29500 --nproc_per_node 8 ./configs/r50_ut_detr_omni_dvd.sh
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
+ EXP_DIR=results/sdugamod/aerial/
+ PY_ARGS=
+ python -u main.py --output_dir results/sdugamod/aerial/ --BURN_IN_STEP 20 --TEACHER_UPDATE_ITER 1 --EMA_KEEP_RATE 0.9996 --annotation_json_label aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json --annotation_json_unlabel Aerial_8605_scaled_labels_w_perspective_aligned_ids.json --CONFIDENCE_THRESHOLD 0.7 --data_path ../../../data/jlorray1/SDU-GAMODv4/ --lr 2e-4 --epochs 58 --lr_drop 150 --pixels 600 --num_queries 900 --nheads 16 --num_feature_levels 4 --save_freq 4 --dataset_file dvd
| distributed init (rank 4): env://
| distributed init (rank 0): env://
| distributed init (rank 3): env://
| distributed init (rank 2): env://
| distributed init (rank 6): env://
| distributed init (rank 5): env://
| distributed init (rank 1): env://
| distributed init (rank 7): env://
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
git:
  sha: d1f802f64e5c9079726a99e8b46a6925cdd6ed84, status: has uncommited changes, branch: main

Namespace(lr=0.0002, lr_backbone_names=['backbone.0'], lr_backbone=2e-05, lr_linear_proj_names=['reference_points', 'sampling_offsets'], lr_linear_proj_mult=0.1, batch_size=1, weight_decay=0.0001, epochs=58, lr_drop=150, lr_drop_epochs=None, clip_max_norm=0.1, sgd=False, with_box_refine=False, two_stage=False, frozen_weights=None, backbone='resnet50', dilation=False, position_embedding='sine', position_embedding_scale=6.283185307179586, num_feature_levels=4, enc_layers=6, dec_layers=6, dim_feedforward=1024, hidden_dim=256, dropout=0.1, nheads=16, num_queries=900, dec_n_points=4, enc_n_points=4, masks=False, aux_loss=True, set_cost_class=2, set_cost_bbox=5, set_cost_giou=2, mask_loss_coef=1, dice_loss_coef=1, cls_loss_coef=2, bbox_loss_coef=5, giou_loss_coef=2, focal_alpha=0.25, dataset_file='dvd', data_path='../../../data/jlorray1/SDU-GAMODv4/', coco_panoptic_path=None, remove_difficult=False, output_dir='results/sdugamod/aerial/', device='cuda', seed=42, resume='', start_epoch=0, eval=False, num_workers=2, cache_mode=False, percent='10', BURN_IN_STEP=20, TEACHER_UPDATE_ITER=1, EMA_KEEP_RATE=0.9996, annotation_json_label='aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json', annotation_json_unlabel='Aerial_8605_scaled_labels_w_perspective_aligned_ids.json', CONFIDENCE_THRESHOLD=0.7, PASSED_NUM='passed_num.txt', pixels=600, w_p=0.5, w_t=0.5, save_freq=4, eval_freq=1, rank=0, world_size=8, gpu=0, dist_url='env://', distributed=True, dist_backend='nccl')
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/jlorray1/anaconda3/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
number of student model params: 41320474
number of teacher model params: 41320474
loading annotations into memory...
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
Done (t=3.53s)
creating index...
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
index created!
loading annotations into memory...
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/./tools/launch.py", line 192, in <module>
    main()
  File "/home/jlorray1/mixup-omni-detr-2/./tools/launch.py", line 187, in main
    raise subprocess.CalledProcessError(returncode=process.returncode,
subprocess.CalledProcessError: Command '['./configs/r50_ut_detr_omni_dvd.sh']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
Traceback (most recent call last):
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 546, in <module>
    main(args)
  File "/home/jlorray1/mixup-omni-detr-2/main.py", line 193, in main
    dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/__init__.py", line 128, in build_dataset
    return build_coco_semi_unlabel(image_set, args)
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 684, in build_semi_unlabel
    dataset = CocoDetection_semi(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/coco.py", line 67, in __init__
    super(CocoDetection_semi, self).__init__(img_folder, ann_file,
  File "/home/jlorray1/mixup-omni-detr-2/datasets/torchvision_datasets/coco.py", line 104, in __init__
    self.coco = COCO(annFile)  # self.coco is initialized to a COCO object, and takes the annotation file found at annFile and converts it into a dictionary
  File "/home/jlorray1/anaconda3/lib/python3.9/site-packages/pycocotools/coco.py", line 81, in __init__
    with open(annotation_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../../data/jlorray1/SDU-GAMODv4/unsupervised_annotations/Aerial_8605_scaled_labels_w_perspective_aligned_ids.json'
