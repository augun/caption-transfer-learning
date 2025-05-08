import torch
import torch.nn as nn
from transformers import CLIPModel

from decoder import FullCaptionDecoder

class ClipCaptionModel(nn.Module):
    def __init__(self, 
                 clip_model_name="openai/clip-vit-base-patch32",
                 d_model=512,
                 vocab_size=49408,  # default GPT2 vocab
                 prefix_length=10,
                 num_layers=2,
                 heads=8,
                 ff_dim=2048,
                 max_caption_len=40):
        super().__init__()

        # Load CLIP and freeze it
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False

        # Get CLIP embedding dimension
        clip_dim = self.clip.config.projection_dim
        self.prefix_length = prefix_length
        self.d_model = d_model

        # Project image features to prefix token embeddings
        self.project = nn.Linear(clip_dim, d_model * prefix_length)

        # Store token_embed at top level to match saved checkpoint
        self.token_embed = self.clip.text_model.embeddings.token_embedding

        # Decoder
        self.decoder = FullCaptionDecoder(
            clip_text_embedding=self.token_embed,
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            heads=heads,
            ff_dim=ff_dim,
            max_len=max_caption_len
        )

    def forward(self, pixel_values, input_ids, labels=None, mask=None):
        batch_size = input_ids.size(0)

        # If image features are passed, process with CLIP
        if pixel_values is not None:
            with torch.no_grad():
                image_features = self.clip.get_image_features(pixel_values=pixel_values)
            projected = self.project(image_features)
            memory = projected.view(batch_size, self.prefix_length, self.d_model)
        else:
            memory = None

        # Run the decoder
        logits = self.decoder(input_ids, memory, mask=mask)

        # Optional loss computation
        loss = None
        if labels is not None:
            if labels.dim() == 3:
                labels = labels.squeeze(1)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss
