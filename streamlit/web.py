import streamlit as st 
import os 
import sys
import numpy as np

from PIL import Image

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from tools.infer import MushroomClassifier

# Init model
class_names_path = 'model/scripted_model/resnet50/class_names.txt'
classifier = MushroomClassifier(model_path='model/scripted_model/resnet50/scripted_model.pt', file_names_path=class_names_path)

# Introduce
st.header("Mushrooms recognition sample app")
st.write("Mushrooms recognition app – your ultimate companion for accurate identification of mushroom species. Powered by advanced image recognition and machine learning technology, this app ensures safety by helping you distinguish between edible and poisonous varieties. Discover, learn, and explore the fascinating world of mushrooms with ease and confidence.")
st.write("This model can predict 9 types of mushrooms: \
         Suillus, Cortinarius, Russula, Entoloma, Amanita, Hygrocybe, Lactarius, Agaricus, Boletus")

# Upload image
image_file = st.file_uploader("Choose image", type=['jpg', 'jpeg'])

if image_file :
    # Save image
    origin_path = "data/user_data"
    if not os.path.exists(origin_path):
        os.mkdir(origin_path)
        
    path = os.path.join(origin_path, image_file.name)
    with open(path, "wb") as f:
        f.write((image_file).getbuffer())
        
    # Predict
    result = classifier.predict(path)
    class_name = list(result.values())[0]

    # Print result
    img = Image.open(image_file)
    st.image(img, width=250)
    st.write(f"## Result: *{class_name}*")
