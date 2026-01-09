import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import io

class CardVectorizer:
    def __init__(self):
        # Load pre-trained MobileNetV3 Large
        self.weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=self.weights)
        
        # Try to load fine-tuned weights if available
        try:
            self.model.load_state_dict(torch.load('mobilenetv3_tcg_finetuned.pth', map_location='cpu'))
            print("✅ Loaded fine-tuned TCG model")
        except FileNotFoundError:
            print("ℹ️ Using base MobileNetV3 model (no fine-tuning found)")
        
        # Remove the classification layer to get raw embeddings (1280 dim)
        self.model.classifier[3] = torch.nn.Identity()
        
        self.model.eval() # Set to evaluation mode
        
        # Define image preprocessing pipeline
        self.preprocess = self.weights.transforms()

    def vectorize_image(self, image_path_or_bytes):
        """
        Generates a feature vector for an image.
        Args:
            image_path_or_bytes: File path (str) or Image bytes.
        Returns:
            List[float]: A 1280-dimensional vector.
        """
        try:
            if isinstance(image_path_or_bytes, bytes):
                img = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
            else:
                img = Image.open(image_path_or_bytes).convert("RGB")
            
            # Preprocess image
            batch = self.preprocess(img).unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                features = self.model(batch)
            
            # Normalize the embedding vector (L2 normalization for cosine similarity)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            # Flatten and convert to list
            return features.squeeze().tolist()
            
        except Exception as e:
            print(f"Error vectorizing image: {e}")
            return None
