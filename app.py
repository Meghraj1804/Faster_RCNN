import gradio as gr
import os
from src.task.inference import Infer
from PIL import Image
from src.entity.config_entity import ModelParamConfig

cwd = os.getcwd()
model_path = r'model\best.pth'
model_infer = Infer()

# class_name = {0:'Cat', 1:'Dog', 2:'person'}
model = model_infer.load_model(model_path)


def person_deection(image):
    image_path = "uploaded_image.jpg"
    image.save(image_path)
    
    output_path = model_infer.infer(model,image_path)
    
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