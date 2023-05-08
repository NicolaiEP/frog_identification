from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_frog_dataset(name, images_dir, json_file):
    register_coco_instances(name, {}, json_file, images_dir)
    MetadataCatalog.get(name).set(thing_classes=["frog_stomach"])
