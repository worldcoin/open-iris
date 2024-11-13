# loads a model with optim and saves it without optim etc..
from segment_anything import  sam_model_registry
from utils.common import save_model, load_model


model_path = "checkpoints/sam_model_no_threshold/model.pt"
save_path = "checkpoints/sam_model_no_threshold/model_no.pt"


model_type = 'vit_h'
checkpoint = 'weights/sam_vit_h_4b8939.pth'

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)

load_model(sam_model, model_path)

save_model(sam_model, save_path, make_dict=False)
