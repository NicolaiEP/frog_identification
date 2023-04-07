import json
import pathlib
import datasets

try:
    from PIL import Image
except ImportError:
    raise RuntimeError("Could not important Pillow. Please install Pillow using pip install Pillow")


class FrogStomachBuilderConfig(datasets.BuilderConfig):

    def __init__(self, name, splits, image_dir, **kwargs):
        super().__init__(name, **kwargs)
        self.splits = splits
        self.image_dir = image_dir

_CITATION = """\
TODO
}
"""
_DESCRIPTION = """\
TODO
"""
_HOMEPAGE = "TODO"
_LICENSE = ""
_URLs = {}


class FrogStomachDataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) COCO dataset"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = FrogStomachBuilderConfig
    BUILDER_CONFIGS = [
        FrogStomachBuilderConfig(name='default', splits=['train', 'val'], image_dir=None),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        feature_dict = {
            "image_id": datasets.Value("int64"),
            "category_id": datasets.Value("int64"),
            "image_path": datasets.Value("string"),
            "bbox": datasets.Sequence(datasets.Value("int64")),
            "segmentation": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
        }
        features = datasets.Features(feature_dict)
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def list_contains(self, l, l2):
        return any([x in l2 for x in l])

    def _split_generators(self, dl_manager):

        data_files = dl_manager.download({
            "train": "https://huggingface.co/datasets/perara/pelophylax-lessonae/resolve/main/data/train.json",
            "validation": "https://huggingface.co/datasets/perara/pelophylax-lessonae/resolve/main/data/validation.json"
        })

        """Returns SplitGenerators."""
        script_dir = pathlib.Path(__file__).parent

        print(script_dir.absolute())

        image_dir = self.config.image_dir
        data_splits = self.config.splits
        # annotation_dir = self.config.annotation_dir

        if not image_dir or image_dir is None:
            raise ValueError(
                "This script is supposed to work with local image dataset. The argument `image_dir` in `load_dataset("
                ")` is required. "
            )

        image_dir = pathlib.Path(image_dir)

        splits = []
        if self.list_contains(data_splits, ["train"]):
            splits.append(datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "json_path": data_files["train"],
                    "image_dir": image_dir,
                    "split": "train",
                }
            ))
        if self.list_contains(data_splits, ['val', 'valid', 'validation', 'dev']):
            splits.append(datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "json_path": data_files["validation"],
                    "image_dir": image_dir,
                    "split": "val",
                },
            ))


        return splits

    def _generate_examples(
            # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
            self, json_path, image_dir, split
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        _features = ["image_id", "category_id", "image_path", "segmentation", "bbox"]  # "segmentation", "contour"
        features = list(_features)

        image_data = list(image_dir.rglob("*.jpg"))
        image_name = [x.name for x in image_data]

        with pathlib.Path(json_path).open("rb+") as j:
            data = json.load(j)

        # list of dict
        images = data["images"]
        entries = images

        # build a dict of image_id -> image info dict
        d = {image["id"]: image for image in images}

        # list of dict
        if split in ["train", "val"]:
            annotations = data["annotations"]

            # build a dict of image_id ->
            for annotation in annotations:
                _id = annotation["id"]
                image_info = d[annotation["image_id"]]
                annotation.update(image_info)
                annotation["id"] = _id

            entries = annotations

        for id_, entry in enumerate(entries):
            image_sample_path = image_data[image_name.index(entry["file_name"])]
            entry["image_path"] = str(image_sample_path.absolute())

            entry = {k: v for k, v in entry.items() if k in features}
            if split == "test":
                entry["image_id"] = entry["id"]
                entry["id"] = -1
                entry["caption"] = -1

            entry = {k: entry[k] for k in _features if k in entry}
            unique_id = str((entry["image_id"], entry["category_id"]))

            yield unique_id, entry
