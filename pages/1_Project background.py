import streamlit as st 

st.set_page_config(
    page_title="Background"
)

st.markdown(
    """
    ### Understand and classify clouds from satellite images
    The topic of my project is [a Kaggle competition challenge](https://www.kaggle.com/competitions/understanding_cloud_organization/leaderboard), 
    but it was originally a research project, as described in the document [Combining crowd-sourcing and deep learning to explore the meso-scale 
    organization of shallow convection](https://arxiv.org/abs/1906.01906)
    
    ### What is the project goal?
    The original goal of the project is to combine crowd-sourcing and deep learning to explore the meso-scale organization of shallow convection. 
    It means using the collective efforts of many people (crowd-sourcing) and advanced computer algorithms (deep learning) to understand how 
    small-scale atmospheric processes called shallow convection are organized. Deep learning techniques, particularly computer vision, have shown promise in mimicking human pattern recognition abilities, 
    including in the analysis of satellite cloud imagery. 
    
    The challenge of the current project is to build a model to classify cloud organization patterns from satellite images.

    ### What is the target ouput?
    Object detection focuses on detecting and localizing specific objects or features of interest within an image. 
    The output of object detection algorithms is typically a set of bounding boxes that indicate the location and extent of the detected objects. 
    These bounding boxes will be associated with predefined 4-class labels.
    
    Semantic segmentation, on the other hand, aims to classify every pixel in an image according to the specific category or class it belongs to. 
    The output of semantic segmentation is the prediction and segmentation masks generated for each input image.
    """
)

st.image('target output.png')
