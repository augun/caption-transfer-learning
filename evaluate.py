import torch
from transformers import AutoTokenizer
from PIL import Image
from model import ClipCaptionModel
from utils import generate_square_subsequent_mask
from transformers import CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(image_path, model, tokenizer, max_len=40):
    model.eval()

    # Load and preprocess image
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # Step 1: Get CLIP image embedding and projection
    with torch.no_grad():
        image_features = model.clip.get_image_features(pixel_values=pixel_values)
        projected = model.project(image_features).view(1, model.prefix_length, model.decoder_dim)

    # Step 2: Start generation with BOS token
    generated = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

    for _ in range(max_len):
        # Predict logits
        with torch.no_grad():
            logits = model.decoder(generated, memory=projected)  # [1, seq_len, vocab]
            next_token_logits = logits[:, -1, :]  # last token only
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # greedy decoding

        generated = torch.cat((generated, next_token), dim=1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    image_path = "cat.jpg"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ClipCaptionModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("clip_caption_model.pt", map_location=device))  # You’ll need to save this during training
    model.to(device)

    caption = generate_caption(image_path, model, tokenizer)
    print("Generated caption:", caption)
