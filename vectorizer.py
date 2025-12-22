import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import io

class CardVectorizer:
    def __init__(self):
        # Load pre-trained MobileNetV3 Large
        # We use "DEFAULT" weights (best available)
        self.weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=self.weights)
        
        # Remove the classification layer to get raw embeddings (1280 dim)
        # MobileNetV3 classifier structure: Sequential(Linear, Hardswish, Dropout, Linear)
        # We want the output of the penultimate layer or adjust the classifier.
        # A common trick: Replace the last fully connected layer with Identity or just grab the features.
        # MobileNetV3 structure ends with a 'classifier' block.
        # We will modify the classifier to output the 1280 features directly.
        # The last layer is Linear(in=1280, out=1000). We can replace it.
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
            
            # Flatten and convert to list
            return features.squeeze().tolist()
            
        except Exception as e:
            print(f"Error vectorizing image: {e}")
            return None
