import torch
import pickle
from pathlib import Path
from .model import MyModel  # You can move your class definition here

def load_model_and_predict(text):
    # Load vocab
    with open(Path(__file__).parent / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Encode input
    tokens = text.lower().split()
    encoded = [vocab.get(t, vocab["<OOV>"]) for t in tokens]
    padded = encoded[:256] + [vocab["<PAD>"]] * (256 - len(encoded)) if len(encoded) < 256 else encoded[:256]
    input_tensor = torch.tensor([padded])

    # Load label encoder
    with open(Path(__file__).parent / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load model
    model_path = Path(__file__).parent / "model_weights.pth"
    model = MyModel(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=len(label_encoder.classes_), embedding_matrix=None)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return label_encoder.inverse_transform([predicted])[0]
