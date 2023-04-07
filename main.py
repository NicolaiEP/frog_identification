import argparse
from models.detectron2.train_detectron2 import train_detectron2_model
from models.siamese_net.siameese import *

def main(args):
    if args.train_detectron2:
        train_detectron2_model(args)
        print("world domination")

    if args.train_mrcnnn:
        print("this is a good code")

def siamese_cat():
    parser = argparse.ArgumentParser(description="Train a Siamese network for frog identification")

    parser.add_argument("--data-dir", type=str, default="frog_identification\data",
                        help="Path to the directory containing the training data and annotations")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--input-shape", type=str, default="224,224,3",
                        help="Input shape of the images in the format 'height,width,channels'")
    parser.add_argument("--model-file", type=str, default="frog_identification\saved_models\detrectron2\pretrained_weights\mask_rcnn_frog.h5",
                        help="Path to the output model file")
    parser.add_argument("--save-frequency", type=int, default=5,
                        help="Save model every `save_frequency` epochs")
    parser.add_argument("--train-val-split", type=float, default=0.8,
                        help="Fraction of data to use for training, the rest is used for validation")

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frog Identification System")
    parser.add_argument("--train_detectron2", action="store_true", help="Train the Detectron2 model")
    # Add more arguments for other operations, e.g., testing, visualization, etc. (not all is implemented)

    args = parser.parse_args()

    print(args.data_dir)
    print(args.batch_size)
    print(args.epochs)
    print(args.lr)
    print(args.input_shape)
    print(args.model_file)
    print(args.save_frequency)
    print(args.train_val_split)

    args = parser.parse_args()
    main(args)



