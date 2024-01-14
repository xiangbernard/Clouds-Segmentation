import cv2
import os
import matplotlib.pyplot as plt

# local paths

file_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/Cloud Seg train.csv'
train_folder_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/train_images'
test_folder_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/test_images'
masks_for_comparaison_path = 'E:/Xiang/00_Datasets/Project/YOLOV8_SEGMENT_dataset/true masks'

def Shwo_predictions_myYolo (models, model_names, image_file:list, save_folder: str):
    # load true mask, saved into the same name as image
    mask_id = image_file.split('.')[0]
    mask = cv2.imread(os.path.join(masks_for_comparaison_path, f"{mask_id}.png"))
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))  # Adjust figure size for horizontal images
    fig.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Display true mask
    axs[0].imshow(mask_rgb)
    axs[0].set_title("True Mask")
    axs[0].axis('off')

    # Path of the image to be predicted
    img_path = os.path.join(train_folder_path, image_file)

    for i, (model, model_name) in enumerate(zip(models, model_names), start=1):
        model_name = model_names[i-1]
        axs[i].set_title(f"{model_name} / {mask_id}")
        
        # Predict using each model
        results = model.predict(img_path)
        for r in results:
            im_array = r.plot()  # Plot a BGR numpy array of predictions
            im_array_rgb = im_array[..., ::-1]  # Convert BGR to RGB
            axs[i].imshow(im_array_rgb)
            axs[i].axis('off')  # Turn off the axis/grid

    # Save the plotted images as .png files with a horizontal rectangle layout
    save_path = os.path.join(save_folder, f"{mask_id}_predictions.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()