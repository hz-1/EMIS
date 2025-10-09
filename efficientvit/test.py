# segment anything
from efficientvit.sam_model_zoo import create_sam_model

efficientvit_sam = create_sam_model(
  name="l0", weight_url="./assets/checkpoints/sam/l0.pt",
)
efficientvit_sam = efficientvit_sam.cuda().eval()

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)