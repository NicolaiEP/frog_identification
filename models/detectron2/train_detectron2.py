import os
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from utils.data_utils import register_frog_dataset


def custom_config(num_classes):
    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    # Solver
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 70000
    cfg.SOLVER.STEPS = (20, 10000, 20000, 30000, 40000, 50000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000


    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    #cfg.TEST.EVAL_PERIOD = 5

    # INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    # DATASETS
    cfg.DATASETS.TEST = ('frog_stomach_val',)
    cfg.DATASETS.TRAIN = ('frog_stomach_train',)

    # OUTPUT
    cfg.OUTPUT_DIR = r"saved_models/detectron2/trained_weights"

    return cfg


def train_detectron2_model(args):
    # Register the dataset if not already registered
    dataset_name = "frog_stomach_train"
    if dataset_name not in DatasetCatalog.list():
        register_frog_dataset(dataset_name, "data/images", "data/labels/train.json")

    dataset_name = "frog_stomach_val"
    if dataset_name not in DatasetCatalog.list():
        register_frog_dataset(dataset_name, "data/images", "data/labels/validation.json")

    # Set up Detectron2 configuration
    cfg = custom_config(num_classes=1)

    # Train the model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
