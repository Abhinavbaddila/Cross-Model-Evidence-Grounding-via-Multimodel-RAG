import torch
from transformers import CLIPTokenizer, CLIPModel


class CLIPTextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        """
        texts: list[str]
        returns: torch tensor (N, D)
        """

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # ✅ THIS is the correct way to get embeddings
        outputs = self.model.get_text_features(**inputs)

        # normalize
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        return outputs.cpu()