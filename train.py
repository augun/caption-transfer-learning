import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import ClipCaptionDataset
from model import ClipCaptionModel
from transformers import AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

# Load dataset
dataset = ClipCaptionDataset(
    image_dir="data/Flick30k_Images",
    caption_csv="data/results.csv",
    tokenizer_name="gpt2"
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Build model
model = ClipCaptionModel(vocab_size=vocab_size)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()
        _, loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "clip_caption_model.pt")
