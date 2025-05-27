import argparse
from .model_utils import load_model_and_predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Text to classify")
    parser.add_argument("--kaggle", action="store_true", help="Print Kaggle ID")
    args = parser.parse_args()

    if args.kaggle:
        print("marcelatorrelledo")  # Replace with your actual Kaggle ID
    elif args.input:
        label = load_model_and_predict(args.input)
        print(label)
    else:
        print("Please use --input \"some text\" or --kaggle")

if __name__ == "__main__":
    main()
