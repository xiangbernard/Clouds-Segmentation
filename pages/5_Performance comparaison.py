import pandas as pd
import os
import random
import streamlit as st 
from dea import df_transform, check_images_uniques, plot_images_per_label, plot_labels_per_images
from mask_display import Display_choice

# local paths

file_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/Cloud Seg train.csv'
train_folder_path = 'E:/Xiang/00_Datasets/Project//Cloud_dataset/train_images'
test_folder_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/test_images'

# Load train.csv
train = pd.read_csv(file_path)
# Split image name & lable
train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
# Load train images
train_image_files = os.listdir(train_folder_path)

st.set_option('deprecation.showPyplotGlobalUse', False)
chapters = ["Our built models", "YoloV8"]
chapter = st.sidebar.radio("Select one model :", chapters)

# page : Our built models
if chapter == chapters[0] : 
    st.write("Our built models")  

# page : YoloV8
elif chapter == chapters[1] :
    st.write("YoloV8")