# Installation
```
pip install -e .
```

# Usage
```
from PIL import Image
import open_clip
import torch

device = "cuda:6" if torch.cuda.is_available() else "cpu"
model_name = "ViT-SO400M-14-SigLIP"
# model_name = "ViT-B-16-SigLIP"
pretrained = ""

model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model.prepare_latent()
model = model.to(device).to(torch.bfloat16)
tokenizer = open_clip.get_tokenizer(model_name)

pooled_features_text, last_hidden_states_text = model.text(tokenizer('a beautiful tree').to(image.device), use_ssl_head=False)

input_image = transform(Image.open('input img')).unsqueeze(0).to(device).to(torch.bfloat16)
pooled_features_vision, last_hidden_states_vision = model.visual(image, use_ssl_head=False)
```