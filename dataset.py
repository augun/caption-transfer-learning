import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor, AutoTokenizer
import torch

class ClipCaptionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        caption_csv,
        tokenizer_name="openai/clip-vit-base-patch32",
        clip_model_name="openai/clip-vit-base-patch32",
        max_length=40,
        limit=None
    ):
        self.image_dir = image_dir
        self.max_length = max_length

        # Load captions
        df = pd.read_csv(caption_csv, sep="|")
        df.columns = [col.strip() for col in df.columns]

        # Pick the first comment per image
        grouped = df.groupby("image_name")["comment"].first().reset_index()

        # Apply limit if requested
        if limit:
            grouped = grouped.head(limit)

        self.samples = list(zip(grouped["image_name"], grouped["comment"]))

        # Load CLIP tokenizer + processor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_filename, caption = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_filename.strip())

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption_text": caption  # optional, for debugging
        }
