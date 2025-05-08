import os
import torch
from transformers import AutoTokenizer, CLIPProcessor
from PIL import Image
from model import ClipCaptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(image, model, tokenizer, max_len=40):
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        image_features = model.clip.get_image_features(pixel_values=pixel_values)
        projected = model.project(image_features).view(1, model.prefix_length, model.d_model)

    # Start with token "A"
    start_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    eos_token_id = tokenizer.eos_token_id or start_token_id

    generated = torch.tensor([[start_token_id]], dtype=torch.long).to(device)

    for i in range(max_len):
        logits = model.decoder(generated, memory=projected)
        next_token_logits = logits[:, -1, :]

        temperature = 1.2
        probs = torch.softmax(next_token_logits / temperature, dim=-1)

        if i < 2:
            probs[0][eos_token_id] = 0
            probs = probs / probs.sum()

        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    image_dir = "data/Flick30k_Images"
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ClipCaptionModel(vocab_size=tokenizer.vocab_size)
    state_dict = torch.load("clip_caption_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"📷 Found {len(image_files)} images in {image_dir}\n")

    for filename in sorted(image_files)[:10]:  # ← limit to 10 for now
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        caption = generate_caption(image, model, tokenizer)
        print(f"🖼️ {filename} → {caption}")
