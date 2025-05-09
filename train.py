import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import ClipCaptionDataset
from model import ClipCaptionModel
from transformers import AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = tokenizer.vocab_size

# Load full Flickr30k dataset
dataset = ClipCaptionDataset(
    image_dir="data/Flick30k_Images",
    caption_csv="data/results.csv",
    # limit = 100
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Build model
model = ClipCaptionModel(vocab_size=vocab_size)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss

        optimizer.zero_grad()
        _, loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # Save checkpoint after each epoch
    checkpoint_path = f"clip_caption_model_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Model saved to {checkpoint_path}")


# Save trained model
torch.save(model.state_dict(), "clip_caption_model.pt")
print("✅ Model saved to clip_caption_model.pt")
