import base64
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
#img=cv2.imread("fertilizer1.png")


#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
#plt.show()
#plt.close('all')
def generate(user_input,image_path):
    vertexai.init(project="fresh-capsule-406715", location="us-central1")
    model = GenerativeModel("gemini-1.0-pro-vision-001")
    
    # Read image file as binary data
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Convert binary image data to base64 encoding
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Create an image part from the base64-encoded image data
    image_part = Part.from_data(data=base64.b64decode(image_base64), mime_type="image/png")

    responses = model.generate_content(
        [user_input, image_part],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=True,
    )
    ans=""
    for response in responses:
        ans+=" "+response.text
    return ans

 
#print(generate("what is its image","fertilizer1.png"))

