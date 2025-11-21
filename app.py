import gradio as gr
import os
from inference import load_model, infer
from PIL import Image
from src.entity.config_entity import ModelParamConfig

cwd = os.getcwd()
model_path = r'model\best.pth'
model_config = ModelParamConfig()

# class_name = {0:'Cat', 1:'Dog', 2:'person'}
model = load_model(model_path,model_config)


def person_deection(image):
    image_path = "uploaded_image.jpg"
    image.save(image_path)
    
    model = load_model(model_path,model_config)
    output_path = infer(model,image_path)
    
    return  Image.open(output_path)
    
    
demo = gr.Interface(
    fn=person_deection,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Image(label='labeled image')],
    title = "Person Detection gradio app",
    description = 'upload an image to detect preson'
)

if __name__ == "__main__":
    demo.launch()