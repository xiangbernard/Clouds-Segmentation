import numpy as np
import pandas as pd
import os
import random
import ultralytics

import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
from Models_predict import Show_predictions, load_and_preprocess
from Models_predict import MultiClassDiceLoss,MultiClassDiceMetric,IoUMetric,PrecisionMetric,RecallMetric


# local paths
file_path = 'C:/Users/yanmei.peng/Downloads/ML/Cloud_dataset/Cloud Seg train.csv'
train_folder_path = 'C:/Users/yanmei.peng/Downloads/ML/Cloud_dataset/train_images'
train_masks_path = 'C:/Users/yanmei.peng/Downloads/ML/Cloud_dataset/train_masks'
trained_model_path = 'C:/Users/yanmei.peng/Downloads/ML/0_Streamlit/trained_models/'

# file_path = 'C:/Users/pym12/Downloads/ML/Cloud_dataset/Cloud Seg train.csv'
# train_folder_path = 'C:/Users/pym12/Downloads/ML/Cloud_dataset/train_images'
# train_masks_path = 'C:/Users/pym12/Downloads/ML/Cloud_dataset/train_masks'
# trained_model_path = 'C:/Users/pym12/Downloads/ML/0_Streamlit/trained_models/'

# Set parameters
BATCH_SIZE = 1

# Define test images files names
test_image_files =['ff5e79c.jpg','ff7f590.jpg','ffbf254.jpg', 'ffc31af.jpg', 'ffd11b6.jpg','ffd6680.jpg','ff2b181.jpg','ff38d0a.jpg','ff5d4cf.jpg','ff6d2fd.jpg','ff81435.jpg','ffb3266.jpg','ff961b3.jpg']

# Create un dataframe containing the paths of images and masks
#@st.cache_data()
def create_dataset(test_image_files,train_folder_path):
    input_img_paths = sorted([os.path.join(train_folder_path, fname)
        for fname in os.listdir(train_folder_path)
        if fname.endswith(".jpg") and fname in test_image_files])

    annotation_img_paths = sorted( [os.path.join(train_masks_path, fname[:-3] + "png")
        for fname in test_image_files
        if fname[:-3] + "png" in os.listdir(train_masks_path)])
    
    Val = tf.data.Dataset.from_tensor_slices((input_img_paths, annotation_img_paths))
    Val = (Val.map(load_and_preprocess, num_parallel_calls=-1) # map function to load image and mask
    .batch(BATCH_SIZE) # Split in batch
    .prefetch(-1))
    return Val

Val = create_dataset(test_image_files,train_folder_path)

custom_objects = {'MultiClassDiceLoss': MultiClassDiceLoss, 'IoUMetric': IoUMetric,'PrecisionMetric':PrecisionMetric}

@st.cache_resource()
def Load_model(model_path, custom_objects=custom_objects):
    model = load_model(model_path, custom_objects=custom_objects)
    return model

# Streamlit part
st.set_option('deprecation.showPyplotGlobalUse', False)
chapters = ["EfficientNetB0 TL", "FCN VGG16 TL", "Unet from scratch", "PSPnet from scratch","4 models"]
chapter = st.sidebar.radio("Select one model :", chapters)

# page : EfficientNetB0 TL
if chapter == chapters[0] : 
    st.write("EfficientNetB0 TL")  
    st.write("Prediction") 
    # image_file_option = st.selectbox(label = "Select one image to predict", options = test_image_files)
    trained_model = Load_model(trained_model_path+'EfficientNetB0_metrics_opt.h5', custom_objects=custom_objects)
    if st.checkbox("Get a new image") : 
        random_value = random.randint(0, 12)
    st.pyplot(Show_predictions(trained_model, Val, random_value))
    
# page : FCN VGG16 TL
elif chapter == chapters[1] : 
    st.write("FCN VGG16 TL") 
    st.write("Prediction") 
    # image_file_option = st.selectbox(label = "Select one image to predict", options = test_image_files)
    trained_model = Load_model(trained_model_path+'fcn_metrics_opt.h5', custom_objects=custom_objects)
    if st.checkbox("Get a new image") : 
        random_value = random.randint(0, 12)
    st.pyplot(Show_predictions(trained_model, Val, random_value))
    
# page : Unet from scratch
elif chapter == chapters[2] : 
    st.write("Unet from scratch") 
    st.write("Prediction") 
    # image_file_option = st.selectbox(label = "Select one image to predict", options = test_image_files)
    trained_model = Load_model(trained_model_path+'unet_metrics_opt.h5', custom_objects=custom_objects)
    if st.checkbox("Get a new image") : 
        random_value = random.randint(0, 12)
    st.pyplot(Show_predictions(trained_model, Val, random_value))


# page : PSPnet from scratch
elif chapter == chapters[3] :
    st.write("PSPnet from scratch")
    st.write("Prediction") 
    # image_file_option = st.selectbox(label = "Select one image to predict", options = test_image_files)
    trained_model = Load_model(trained_model_path+'pspnet_metrics_opt.h5', custom_objects=custom_objects)
    if st.checkbox("Get a new image") : 
        random_value = random.randint(0, 12)
    st.pyplot(Show_predictions(trained_model, Val, random_value))
    
# 4 models
elif chapter == chapters[4] :
   st.write("models comparason")
   st.write("Prediction") 
   display_option =st.multiselect(label = "Choose models", options=['EfficientNetB0 TL', 'FCN VGG16 TL','Unet from scratch', 'PSPnet from scratch'], placeholder="Choose an option")
   if st.checkbox("Get a new image") : 
       random_value = random.randint(0, 12)
   for i in display_option:
       if i =='EfficientNetB0 TL': 
           st.write("EfficientNetB0 TL")
           model_name='EfficientNetB0_metrics_opt.h5'
       elif i=='FCN VGG16 TL':
           st.write("FCN VGG16 TL")
           model_name='fcn_metrics_opt.h5'
       elif i=='Unet from scratch':
           st.write("Unet from scratch")
           model_name='unet_metrics_opt.h5'
       else:
           st.write("PSPnet from scratch")
           model_name='pspnet_metrics_opt.h5'          
            
       trained_model = Load_model(trained_model_path+model_name, custom_objects=custom_objects)

       st.pyplot(Show_predictions(trained_model, Val, random_value))