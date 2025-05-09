import os
import torch
from transformers import AutoTokenizer, CLIPProcessor
from PIL import Image
from model import ClipCaptionModel
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {device}")

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    logits = logits.clone()

    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(logits.size(0)):
            indices = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices] = filter_value

    return logits


def generate_caption(image, model, tokenizer, max_len=80, temperature=0.6, top_k=50, top_p=0.95):
    model.eval()
    device = next(model.parameters()).device

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        image_features = model.clip.get_image_features(pixel_values=pixel_values)
        memory = model.project(image_features).view(1, model.prefix_length, model.d_model)

    # Start with a prompt
    prompt = "A photo of"
    generated = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    repetition_tracker = {}

    for i in range(max_len):
        logits = model.decoder(generated, memory=memory)
        next_token_logits = logits[:, -1, :]

        # Filter and sample
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits / temperature, dim=-1)

        # Avoid early stopping
        if i < 3 and tokenizer.eos_token_id is not None:
            probs[0][tokenizer.eos_token_id] = 0.0
            probs = probs / probs.sum()

        # Encourage EOS after 5 tokens
        if i > 5 and tokenizer.eos_token_id is not None:
            probs[0][tokenizer.eos_token_id] += 0.1
            probs = probs / probs.sum()

        # Clamp low-prob tokens to avoid sampling junk
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        generated = torch.cat((generated, next_token), dim=1)

        # Repetition tracking
        repetition_tracker[token_id] = repetition_tracker.get(token_id, 0) + 1
        if repetition_tracker[token_id] > 5:
            break

        # Normal EOS stop
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    image_dir = "data/Flick30k_Images"
    model_path = "clip_caption_model_epoch_8.pt"

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ClipCaptionModel(vocab_size=tokenizer.vocab_size, max_caption_len=100)
 
    state_dict = torch.load(model_path, map_location=device)

    # Delete the incompatible buffer
    if "decoder.pos_enc.pe" in state_dict:
        del state_dict["decoder.pos_enc.pe"]

    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded with positional encoding re-initialized")


    model.to(device)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"📷 Found {len(image_files)} images in {image_dir}\n")

    for filename in sorted(image_files)[:10]:  # limit to 10 for preview
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        try:
            caption = generate_caption(image, model, tokenizer)
        except Exception as e:
            caption = f"[ERROR] {e}"

        print(f"🖼️ {filename} → {caption}")
