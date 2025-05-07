import argparse
import torch
import torch.nn as nn
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

# Define a simple module for a learnable embedding.
class LearnableEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(LearnableEmbedding, self).__init__()
        # Initialize a learnable parameter.
        self.embedding = nn.Parameter(torch.randn(embedding_dim))
    
    def forward(self, batch_size):
        # Expand the parameter to the batch dimension: (batch_size, embedding_dim).
        return self.embedding.unsqueeze(0).expand(batch_size, -1)

# Define the MLP that takes Gaussian noise and produces a noise embedding.
class NoiseMLP(nn.Module):
    def __init__(self, noise_dim, embedding_dim, hidden_dim=128):
        super(NoiseMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, noise):
        # noise is expected to be of shape (batch_size, noise_dim).
        return self.mlp(noise)

def load_pipeline(model_path: str, device: torch.device):
    """
    Loads the StableDiffusionInstructPix2PixPipeline from the given model path.
    
    Args:
        model_path (str): Path or identifier for the pretrained model.
        device (torch.device): Device on which to load the model.
        
    Returns:
        StableDiffusionInstructPix2PixPipeline: The loaded pipeline.
    """
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)
    return pipe

def generate_images(input_image: Image.Image,
                    pipe,
                    checkpoint_path: str = None,
                    n: int = 3,
                    image_guidance_scale: float = 7.5,
                    guidance_scale: float = 7.5,
                    embedding_dim: int = 512,
                    noise_dim: int = 100,
                    noise_embedding_dim: int = 512,
                    device: torch.device = None):
    """
    Generates images using a provided pipeline with custom learnable and noise embeddings.
    
    Args:
        input_image (PIL.Image.Image): The initial image.
        pipe: The loaded diffusion pipeline.
        checkpoint_path (str, optional): Path to a checkpoint file with custom module states.
        n (int): Number of images to generate for each of the positive and negative features.
        image_guidance_scale (float): Guidance scale for image conditioning.
        guidance_scale (float): Guidance scale for embedding conditioning.
        embedding_dim (int): Dimension for the learnable embeddings.
        noise_dim (int): Dimension of the input Gaussian noise.
        noise_embedding_dim (int): Dimension of the output from the noise MLP.
        device (torch.device, optional): Device to use for computation.
        
    Returns:
        tuple: (positive_generated, negative_generated) lists of generated images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate custom modules.
    positive_embedding_model = LearnableEmbedding(embedding_dim).to(device)
    negative_embedding_model = LearnableEmbedding(embedding_dim).to(device)
    noise_mlp = NoiseMLP(noise_dim, noise_embedding_dim).to(device)

    # Optionally load checkpoint for custom modules.
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "positive_embedding" in checkpoint:
            positive_embedding_model.load_state_dict(checkpoint["positive_embedding"])
        if "negative_embedding" in checkpoint:
            negative_embedding_model.load_state_dict(checkpoint["negative_embedding"])
        if "noise_mlp" in checkpoint:
            noise_mlp.load_state_dict(checkpoint["noise_mlp"])
        print("Checkpoint loaded successfully.")

    batch_size = 1  # Modify if you need a different batch size.

    # Generate two separate Gaussian noise vectors.
    noise_vector_0 = torch.randn(batch_size, noise_dim, device=device)
    noise_vector_1 = torch.randn(batch_size, noise_dim, device=device)

    # Get noise embeddings from the MLP.
    noise_embedding_0 = noise_mlp(noise_vector_0)  # Shape: (batch_size, noise_embedding_dim)
    noise_embedding_1 = noise_mlp(noise_vector_1)  # Shape: (batch_size, noise_embedding_dim)

    # Get the learnable embeddings (expanded to the batch size).
    pos_emb = positive_embedding_model(batch_size)  # Shape: (batch_size, embedding_dim)
    neg_emb = negative_embedding_model(batch_size)  # Shape: (batch_size, embedding_dim)

    # Concatenate along the feature dimension to create the final features.
    positive_features = torch.cat([pos_emb, noise_embedding_0], dim=1)
    negative_features = torch.cat([neg_emb, noise_embedding_1], dim=1)

    # Generate images using the positive features.
    positive_generated = []
    for _ in range(n):
        output = pipe(
            embedding=positive_features,
            image=input_image,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1
        ).images
        positive_generated.extend(output)

    # Generate images using the negative features.
    negative_generated = []
    for _ in range(n):
        output = pipe(
            embedding=negative_features,
            image=input_image,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1
        ).images
        negative_generated.extend(output)

    return positive_generated, negative_generated

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using custom learnable and noise embeddings with a provided pipeline."
    )
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained model path")
    parser.add_argument("--im_path", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--n", type=int, default=3, help="Number of images to generate per feature")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the diffusion pipeline.
    pipe = load_pipeline(args.pretrained, device)
    
    # Load the input image.
    init_image = Image.open(args.im_path)
    
    # Generate images using the provided pipeline.
    pos_images, neg_images = generate_images(
        input_image=init_image,
        pipe=pipe,
        checkpoint_path=args.checkpoint,
        n=args.n,
        device=device
    )
    
    for idx, img in enumerate(pos_images):
        img.save(f"positive_generated_{idx}.png")
    for idx, img in enumerate(neg_images):
        img.save(f"negative_generated_{idx}.png")

if __name__ == "__main__":
    main()
