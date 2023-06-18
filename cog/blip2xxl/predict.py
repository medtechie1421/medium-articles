from cog import BasePredictor, Input, Path
import os
from lavis.models import load_model_and_preprocess
from PIL import Image
import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'

class Predictor(BasePredictor):
    def setup(self,device = "cuda" , model_type = "pretrain_flant5xxl" , model_name = "blip2_t5"):
        self.model, self.vis_processors, _ = load_model_and_preprocess(
    name=model_name , model_type=model_type, is_eval=True, device=device
)

    # The arguments and types the model takes as input
    def predict(self,
            image: Path = Input(description="Image to run inference on"),
            input_query: str = Input(description="Insert Your query",default="What is the color of the image")
    ) -> str:
        """Run a single prediction on the model"""
        filepath = str(image)
        device = "cuda"
        raw_image = Image.open(filepath).convert("RGB")
        raw_image.resize((224, 224))
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        prediction = self.model.generate({"image": image, "prompt": f"Question: {input_query}? Answer:"})[0]
        return str(prediction)