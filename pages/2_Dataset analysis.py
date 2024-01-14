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

@st.cache_data
def Load_DataCSV(file_path):
    # Load train.csv
    train = pd.read_csv(file_path)
    train_copy = train.copy()

    # Split image name & lable
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

    # Transform Dataset for analysis usage
    df_train = df_transform(train)
    
    return train_copy, df_train

train_copy, df_train = Load_DataCSV(file_path)

@st.cache_data
def Load_TrainImages(train_folder_path):
    train_image_files = os.listdir(train_folder_path)
    return train_image_files

# Load train images
train_image_files = Load_TrainImages(train_folder_path)

# Streamlit part
st.set_option('deprecation.showPyplotGlobalUse', False)
chapters = ["Load dataset", "Transform Dataset", "Explore images & labels", "Show some images & masks"]
chapter = st.sidebar.radio("Select one section :", chapters)

# page : Load dataset
if chapter == chapters[0] : 
    st.write("The provided .csv file is as following")  
    st.dataframe(train_copy.head(8))

# page : Transform Dataset
elif chapter == chapters[1] : 
    st.write("We transformed the dataset into one image per row format") 
    st.dataframe(df_train.head()) 

    # Check unique images
    st.write("check if there any multipled image in dataset:") 
    unique_image_nb, total_entries = check_images_uniques(df_train)
    st.write(f"Total number of images: {total_entries}")
    st.write(f"Number of unique image names: {unique_image_nb}")

# page : Explore images & labels
elif chapter == chapters[2] : 
    fig_1 = plot_images_per_label(df_train)
    st.pyplot(fig_1)
    fig_2 = plot_labels_per_images(df_train)
    st.pyplot(fig_2)

# page : Show some images & masks
elif chapter == chapters[3] :
    display_option = st.selectbox(label = "Display images & masks", 
                           options = ['Display_four_simple_mask_multi_images', 'Display_multiple_masks_multi_images',
                                      'Display_four_simple_bbox_multi_images', 'Display_multiple_bboxes_multi_images'
                                      ])
    if st.checkbox("Get a new image") : 
        random.shuffle(train_image_files) # get ramdom images
    st.pyplot(Display_choice(display_option, train_image_files))
