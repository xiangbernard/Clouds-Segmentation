import streamlit as st 

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to our project! 👋")

st.sidebar.success("Select a chapter above")

st.markdown(
    """    
    The challenge of the current project is to build a model to classify cloud organization patterns from satellite images.

    To automate the detection of the four organization patterns, we can proceed with 2 methods : 
    - Object Detection: draw boxes around features of interest, essentially mirroring what the human labelers were doing
    - Semantic / Instance Segmentation: classify every pixel of the image into labels
    
    In our project, we have explored both techniques with different models.
    
    **👈 Select a chapter from the sidebar** to see more details
"""
)


