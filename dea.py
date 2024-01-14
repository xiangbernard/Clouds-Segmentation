import pandas as pd
import matplotlib.pyplot as plt

"""
file_path = '/content/Cloud_dataset/train.csv'
train_folder_path = '/content/Cloud_dataset/train_images'
test_folder_path = '/content/Cloud_dataset/test_images'


# local paths

file_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/Cloud Seg train.csv'
train_folder_path = 'E:/Xiang/00_Datasets/Project//Cloud_dataset/train_images'
test_folder_path = 'E:/Xiang/00_Datasets/Project/Cloud_dataset/test_images'

# Load train.csv
train = pd.read_csv(file_path)
train.head()

# Split image name & lable
train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train.head()
"""

def df_transform(df):
    
    # Remove rows with NaN in the 'EncodedPixels' column
    df = df.dropna(subset=['EncodedPixels'])
    
    # Perform one-hot encoding
    one_hot = pd.get_dummies(df['label'])
    # Concatenate the one-hot encoded columns to the original DataFrame
    df = pd.concat([df, one_hot], axis=1)
    
    # Group by 'image' and aggregate the values in the last four columns
    consolidated_df = df.groupby('image', as_index=False).agg({'Fish': 'max','Flower': 'max','Gravel': 'max','Sugar': 'max'})
    
    # Pivot the DataFrame to create separate columns for each label's 'EncodedPixels'
    pivot_df = df.pivot(index='image', columns='label', values='EncodedPixels')
    # Rename the columns with "_encoding"
    pivot_df.columns = [f'{label}_encoding' for label in pivot_df.columns]
    # Reset the index
    pivot_df = pivot_df.reset_index()
    
    merged_df = pd.merge(pivot_df, consolidated_df, on='image')
    return merged_df

def check_images_uniques(df):
    # Get the number of unique values in the image column
    unique_image_nb = df['image'].nunique()

    # Total number of entries in the DataFrame
    total_entries = len(df)
    return unique_image_nb, total_entries

def plot_images_per_label(df):
    class_columns = ['Fish', 'Flower', 'Gravel', 'Sugar']

    # Summing up the one-hot encoded values across rows to get the count for each class
    class_counts = df[class_columns].sum()

    # Plotting the bar chart
    class_counts.plot(kind='bar', width = 0.4, color='g')
    plt.title('Number of images per label')
    plt.xticks(rotation=0)  # Set the x-axis labels without rotation
    plt.show()

def plot_labels_per_images(df):
    class_columns = ['Fish', 'Flower', 'Gravel', 'Sugar']
    # Summing up the one-hot encoded values across columns to get the count of classes for each image
    df['num_classes'] = df[class_columns].sum(axis=1)

    # Counting the occurrences of images with different numbers of classes
    num_classes_count = df['num_classes'].value_counts().sort_index()
    
    plt.pie(x=num_classes_count, labels=num_classes_count.index, autopct = lambda x: str(round(x))+'%', pctdistance = 0.7)
    plt.title('Number of images per nb_labels (1/2/3/4 labels)')
    plt.legend(loc='lower right')
    plt.show()