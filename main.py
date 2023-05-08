import argparse
from models.detectron2.train_detectron2 import train_detectron2_model


def main(args):
    if args.train_detectron2:
        train_detectron2_model(args)
        print("world domination")

    print("this is a good code")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frog Identification System")
    parser.add_argument("--train_detectron2", action="store_true", help="Train the Detectron2 model")
    # Add more arguments for other operations, e.g., testing, visualization, etc. (not all is implemented)

    args = parser.parse_args()
    main(args)

