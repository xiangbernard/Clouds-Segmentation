import streamlit as st 
from Yolov8_predict import Shwo_predictions_myYolo

# !pip install ultralytics --quiet
from ultralytics import YOLO

# local paths

file_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/Cloud Seg train.csv'
train_folder_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/train_images'
test_folder_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/test_images'
masks_for_comparaison_path = 'E:/Xiang/00_Datasets/Project/YOLOV8_SEGMENT_dataset/true masks'

my_trained_model_DET_nano_path = 'E:/Xiang/00_Datasets/Project/runs/detect/yolov8n_60_filtre10/weights/best.pt'
my_trained_model_SEG_nano_path = 'E:/Xiang/00_Datasets/Project/runs/segment/yolov8n-Seg_60_filtre10/weights/best.pt'
my_trained_model_DET_small_path = 'E:/Xiang/00_Datasets/Project/runs/detect/yolov8s_42_filtre10/weights/best.pt'
my_trained_model_SEG_small_path = 'E:/Xiang/00_Datasets/Project/runs/segment/yolov8s-Seg_38_filtre10/weights/best.pt'
pred_masks_path = 'E:/Xiang/00_Datasets/Project/runs/z_prediction'

# Load test images files names
# test_image_files = os.listdir(test_folder_path)
test_image_files =['ff5e79c.jpg','ff7f590.jpg','ffbf254.jpg', 'ffc31af.jpg', 'ffd11b6.jpg','ffd6680.jpg','ff2b181.jpg','ff38d0a.jpg','ff5d4cf.jpg','ff6d2fd.jpg','ff81435.jpg','ffb3266.jpg','ff961b3.jpg']

@st.cache_resource()
def Load_YOLO(model_path):
    model = YOLO(model_path)
    return model

# load the my trained YOLOv8 models
model_DET_nano = Load_YOLO(my_trained_model_DET_nano_path)
model_SEG_nano = Load_YOLO(my_trained_model_SEG_nano_path)
model_DET_small = Load_YOLO(my_trained_model_DET_small_path)
model_SEG_small = Load_YOLO(my_trained_model_SEG_small_path)

models=[model_DET_nano, model_SEG_nano, model_DET_small, model_SEG_small]

models_dict = {
    'DET_nano': model_DET_nano,
    'SEG_nano': model_SEG_nano,
    'DET_small': model_DET_small,
    'SEG_small': model_SEG_small
}
# Get the variable names as strings for each model
model_names = [name for name, model in models_dict.items() if model in models]

# Streamlit part
st.set_option('deprecation.showPyplotGlobalUse', False)
chapters = ["Models presentation", "Prediction"]
chapter = st.sidebar.radio("Select one model :", chapters)

# page : Models presentation
if chapter == chapters[0] : 
    st.write("Models presentation")  

# page : Prediction
elif chapter == chapters[1] : 
    st.write("Prediction") 
    image_file_option = st.selectbox(label = "Select one image to predict", 
                           options = ['ff5e79c.jpg','ff7f590.jpg','ffbf254.jpg', 'ffc31af.jpg', 'ffd11b6.jpg','ffd6680.jpg','ff2b181.jpg',
                                      'ff38d0a.jpg','ff5d4cf.jpg','ff6d2fd.jpg','ff81435.jpg','ffb3266.jpg','ff961b3.jpg'])
    st.pyplot(Shwo_predictions_myYolo(models, model_names, image_file_option, pred_masks_path))
