import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 # import OpenCV
import os
import streamlit as st 

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
    
    return train


# Load train.csv
train = Load_DataCSV(file_path)

@st.cache_data
def Load_TrainImages(train_folder_path):
    train_image_files = os.listdir(train_folder_path)
    return train_image_files

# Load train images
train_image_files = Load_TrainImages(train_folder_path)

# Set colors used to display
labels_colors_codes = {'Fish': [0, 255, 255], 'Sugar': [255, 255, 0], 'Flower': [255, 0, 0], 'Gravel': [0, 255, 0]}
labels_colors_names = {'Fish': 'cool', 'Sugar': 'Wistia', 'Flower': 'autumn', 'Gravel': 'winter_r'}

# Define a size threshold
min_contour_area = 400

### Chapter 1) Get labels values

# Get image's labels, in a list of 4 records, including NaN value, useful for display
def Labels_with_NaN (image_name:str):
    image_records = train.loc[train['image'] == image_name]
    labels = image_records['label'].values
    return labels

# Get image's labels, by eliminating NaN value
def Labels_without_NaN (image_name:str):
    image_records = train.loc[(train['image'] == image_name) & (pd.notna(train['EncodedPixels']))]
    labels = image_records['label'].values
    return labels

# Get labels for image title display, by eliminating NaN value
def Labels_without_NaN_str (image_name:str):
    image_records = train.loc[(train['image'] == image_name) & (pd.notna(train['EncodedPixels']))]
    image_labels = image_records['label'].values
    labels = ' '
    for label in image_labels:
        labels=' '.join(image_labels)
    return labels

### Chapter 2) Get masks values

# Decode RLE to get simple mask (per lable / EncodedPixels)
def Get_simple_mask_rleDecode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    mask_rle = str(mask_rle)
    if mask_rle != 'nan':
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])] # The general syntax for slicing is [start:stop:step]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    # if NaN, img keeps its initial value : zeros
    return img.reshape(shape, order='F')

# Generate multiple masks for an image, including NaN value
def Get_multi_masks_per_image(image_name:str):
    masks = []
    image_records = train.loc[train['image'] == image_name]
    encoding = image_records['EncodedPixels'].values
    for RLEcode in encoding:
        masks.append(Get_simple_mask_rleDecode(RLEcode, (1400, 2100)))
    # print(np.shape(masks)) --> to be sure there are 4 masks, even with NaN value.
    return masks

### Chapter 3) Draw and disply four masks seperately : one mask per image

# Get the mask ready to display, with tranparency
def Get_mask_display_value_color (mask:list, label:str):

    # Get the color for the current label
    mask_color = labels_colors_names.get(label)  # Default color is white if label is not found

    mask = np.clip(mask,0,1) # to ensure that all values in the mask array are between 0 and 1

    # tranform color mask to transparent mask (0: masked - transparent, else : unmasked - visible)
    mask = np.ma.masked_where(mask == 0, mask)

    return mask, mask_color

# Display 4 seperate images for one original image,  with one labeled mask per image
def Display_four_simple_mask_per_row (image:list, masks:list, labels:list, image_name:str):

    fig, axes = plt.subplots(1, 4, subplot_kw=dict(xticks=[], yticks=[]), figsize=(15, 8))

    for i, ax in enumerate(axes.flat): # 4 sub-graphics

        ax.imshow(image)

        mask, mask_color = Get_mask_display_value_color (masks[i], labels[i])
        ax.imshow(mask, alpha=0.6, cmap=mask_color)

        ax.set_title(image_name + ' // ' + labels[i])

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

# Display several rows of images, with per row 4 labeled mask per image
def Display_four_simple_mask_multi_images (nb_images:int, image_files:list):

    for i in range (nb_images):
        if i < len(image_files):
            file_path = os.path.join(train_folder_path, image_files[i])
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            labels = Labels_with_NaN(image_files[i]) # get multi-labels for an image
            masks = Get_multi_masks_per_image(image_files[i]) # get multi-masks for an image

            Display_four_simple_mask_per_row (image, masks, labels,image_files[i]) # Display one mask per image

### Chapter 4) Draw and disply masks - aggragated : all masks for an image

# Display several rows of images, with multiple masks on the same image
def Display_multiple_masks_multi_images (nb_rows: int, nb_cols: int, image_files:list):

    # nb_rows : Number of rows in the grid
    # nb_cols : Number of columns in the grid

    total_images = nb_rows * nb_cols

    fig, axes = plt.subplots(nb_rows, nb_cols, subplot_kw=dict(xticks=[], yticks=[]), figsize=(15, 8))

    for i, ax in enumerate(axes.flat):
        if i < len(image_files):
            file_path = os.path.join(train_folder_path, image_files[i])
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            labels = Labels_with_NaN(image_files[i]) # get multi-labels for an image
            masks = Get_multi_masks_per_image(image_files[i]) # get multi-masks for an image

            ax.imshow(image)

            for mask, label in zip(masks, labels):
                mask, mask_color = Get_mask_display_value_color (mask, label)
                ax.imshow(mask, alpha=0.6, cmap=mask_color)

            ax.set_title(image_files[i] + ' // ' + Labels_without_NaN_str(image_files[i]))

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show();

### Chapter 5) Draw and disply bboxes - seperately : one bbox per label per image

# Get bboxes for one label of an image, return a list of bboxes
def Get_bbox_per_label (mask:list):

    # Find the contours (in case of maks=NaN, no contours will find and no followed action)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the bounding rectangle for each contour
    bboxes_per_label = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_contour_area: # Exclude small masks
            bbox = cv2.boundingRect(cnt)
            bboxes_per_label.append(list(bbox))
    return bboxes_per_label

# Get multiple bboxes of multiple labels on the same image
def Get_multiple_bboxes_per_image (masks:list):
    bboxes_multilabels=[]
    for mask in masks:
        if np.any(mask != 0):  # Add condition mask != 0
            bboxes_multilabels.append(Get_bbox_per_label (mask))
    return bboxes_multilabels

# Draw bbox per label on an image
def Draw_simple_bbox (image:list, bboxes_per_label:list, label:str, width:int=10):

    # Get the color for the current label
    color = labels_colors_codes.get(label, (255, 255, 255))  # Default color is white if label is not found

    # Draw the bboxes and labels on the image
    for rect in bboxes_per_label:
        x, y, w, h = rect     # (x,y) is the point on the top-left
        cv2.rectangle(image, (x, y), (x + w, y + h), color, width)
        # Display the label just above the contour
        cv2.putText(image, label, (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 10)
    return image

# Draw 4 seperate images for one original image,  with one bbox per image
def Draw_four_simple_bbox_per_row_withNaN (image:list, masks:list, labels:list, image_name:str):

    # Predefine 4 subplot, as there are 4 possible labeled masks per image
    fig, axes = plt.subplots(1, 4, subplot_kw=dict(xticks=[], yticks=[]), figsize=(15, 8))

    for i, ax in enumerate(axes.flat): # 4 sub-graphics
        current_image = np.copy(image)  # Create a new copy of the original image for each sub-plot
        bboxes = Get_bbox_per_label (masks[i])
        Draw_simple_bbox (current_image, bboxes, labels[i])

        # Display the current image with the bboxes and label in the corresponding sub-plot
        ax.imshow(current_image)
        ax.set_title(image_name + ' // ' + labels[i])

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

# Display several rows of images, with per row 4 bbox per image
def Display_four_simple_bbox_multi_images (nb_images:int, image_files:list):

    for i in range (nb_images):
        if i < len(image_files):
            file_path = os.path.join(train_folder_path, image_files[i])
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            labels = Labels_with_NaN(image_files[i]) # get multi-labels for an image
            masks = Get_multi_masks_per_image(image_files[i]) # get multi-masks for an image

            Draw_four_simple_bbox_per_row_withNaN (image, masks, labels,image_files[i] ) # Display one mask per image

### Chapter 6) Draw and disply bboxes - aggragated : all bboxes for an image

# Draw multiple bboxes on the same image
def Draw_multiple_bboxes_per_image (image:list, bboxes_multilabels:list, labels:list):
    for bboxes, label in zip(bboxes_multilabels, labels):
        Draw_simple_bbox (image, bboxes, label)
    return image

# Display several rows of images, with multiple bboxes on the same image
def Display_multiple_bboxes_multi_images (nb_rows:int, nb_cols:int, image_files:list):

    # nb_rows : Number of rows in the grid
    # nb_cols : Number of columns in the grid

    total_images = nb_rows * nb_cols

    fig, axes = plt.subplots(nb_rows, nb_cols, subplot_kw=dict(xticks=[], yticks=[]), figsize=(15, 8))

    for i, ax in enumerate(axes.flat):
        if i < len(image_files):
            file_path = os.path.join(train_folder_path, image_files[i])
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            labels = Labels_without_NaN(image_files[i]) # get multi-labels for an image
            masks_multilabels = Get_multi_masks_per_image(image_files[i]) # get multi-masks for an image
            bboxes_multilabels = Get_multiple_bboxes_per_image (masks_multilabels)

            image_with_rectangles = Draw_multiple_bboxes_per_image(image, bboxes_multilabels, labels) # Display multiple masks bboxes

            ax.imshow(image_with_rectangles)
            ax.set_title(image_files[i] + ' // ' + Labels_without_NaN_str(image_files[i]))

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show();

def Display_choice (display_option, image_files:list ):
    if display_option == 'Display_four_simple_bbox_multi_images':
        Display_four_simple_bbox_multi_images(1, image_files)
    elif display_option == 'Display_multiple_bboxes_multi_images':
        Display_multiple_bboxes_multi_images(1, 2, image_files)
    elif display_option == 'Display_four_simple_mask_multi_images':
        Display_four_simple_mask_multi_images(1, image_files)
    elif display_option == 'Display_multiple_masks_multi_images':
        Display_multiple_masks_multi_images(1, 2, image_files)
    