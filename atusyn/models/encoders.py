import os
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms
import open_clip

class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for both text and images
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, pretrained, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS

        # Load the OpenCLIP model
        if not os.path.isfile(pretrained):
            raise FileNotFoundError(pretrained)
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        del model.text_projection
        # Removing the visual model part, which will be replaced with actual image processing
        self.model = model
        # del model.encode_text

        self.device = device
        self.max_length = max_length
        self.image_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整大小到 224x224
            # transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图的规范化（单通道）
        ])

        if freeze:
            self.freeze()

        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_data, modality="text"):
        """
        :param input_data: text or image input (text is a list of strings, image is a batch of images)
        :param modality: string, "text" or "image" to indicate which modality to process
        """
        if modality == "text":
            tokens = open_clip.tokenize(input_data) #1,77
            z = self.encode_with_transformer(tokens.to(self.device))
        elif modality == "image":
            input_data = transforms.functional.resize(input_data, (224, 224))  # 调整到 224x224
            z = self.encode_with_visual_model(input_data)
        else:
            raise ValueError("modality must be either 'text' or 'image'")
        return z

    def encode_with_transformer(self, tokens):
        """
        Text encoding with the transformer model.
        """
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND #77,1,,1024
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def encode_with_visual_model(self, images):
        """
        Image encoding using the visual part of OpenCLIP.
        :param images: a batch of images [batch_size, C, H, W]
        """
        # Get visual embeddings (features)
        visual_features = self.model.visual.forward(images.to(self.device))  # [batch_size, feature_dim]
        return visual_features

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        """
        Forward pass through the text transformer layers.
        """
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, input_data, modality="text"):
        """
        Wrapper function to encode input data based on modality (text or image)
        """
        return self(input_data, modality)
