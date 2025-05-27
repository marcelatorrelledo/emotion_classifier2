import click
import torch
import pickle
import os
from pathlib import Path
import torch.nn.functional as F

from .model import MyModel


@click.command()
@click.option('--input', default=None, help='Input text to classify.')
@click.option('--kaggle', is_flag=True, help='Print Kaggle ID')
def main(input, kaggle):
    if kaggle:
        print("marcelatorre")  # Replace with your actual Kaggle ID
        return

    if input is None:
        print("Please provide an input using --input")
        return

    # Load resources
    import inference
    base_path = Path(inference.__file__).parent

    vocab_path = base_path / "vocab.pkl"
    label_encoder_path = base_path / "label_encoder.pkl"
    model_path = base_path / "model_weights.pth"

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    pad_idx = vocab.get("<PAD>", 0)
    vocab_size = len(vocab)
    embed_dim = 100
    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    model = MyModel(vocab_size, embed_dim, hidden_dim, output_dim, pad_idx)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Encode input
    tokens = input.lower().split()
    encoded = [vocab.get(tok, vocab.get("<OOV>", 1)) for tok in tokens]
    if len(encoded) < 200:
        encoded += [pad_idx] * (200 - len(encoded))
    else:
        encoded = encoded[:200]

    input_tensor = torch.tensor([encoded])

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()

    emotion = label_encoder.inverse_transform([pred])[0]
    print(emotion)
