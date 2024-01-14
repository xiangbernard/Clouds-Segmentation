import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

# Set parameters
image_shape_resized = (256, 384,3)    # the predictions should be scaled down
BATCH_SIZE = 1

# Laod images & DL_masks, and preprocessing for DL model
#@st.cache_data
def load_and_preprocess(img_filepath, mask_filepath):

    img = tf.io.read_file(img_filepath)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_shape_resized[:2])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]

    mask = tf.io.read_file(mask_filepath)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, image_shape_resized[:2], method = "nearest")

    return img, mask

# Display images and their associated masks, with save option
def Display_dataset(display_list, save_path=None):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']   # Elements in "display_list"

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i == 0:  # input image.
            plt.imshow(display_list[i])
        else:       # a masque (true or predicted)
            plt.imshow(tf.squeeze(display_list[i],-1))
        plt.axis('off')
    plt.tight_layout()

    if save_path:
        # Save the image
        plt.savefig(save_path)

    plt.show()
    
@st.cache_data
def get_unique_classes_and_probabilities(pred_mask):
    # Use tf.argmax to find the predicted class for each pixel
    predicted_class_indices = tf.argmax(pred_mask, axis=-1)

    # Calculate the highest probability for each class
    highest_probabilities = tf.reduce_max(pred_mask, axis=-1)

    # Get unique predicted classes
    unique_classes = np.unique(predicted_class_indices)
    print("unique_classes in pred_mask:", unique_classes)

    # Print unique classes and their associated highest probabilities
    class_names = [0,1,2,3,4]
    for class_index in unique_classes:
        class_name = class_names[class_index]  # Replace with your class names
        class_probability = highest_probabilities[predicted_class_indices == class_index]
        highest_prob = np.max(class_probability)
        print(f"Class: {class_name}, Highest Probability: {highest_prob}")

@st.cache_data
def Create_pred_mask(pred_mask_batch_n):
    get_unique_classes_and_probabilities(pred_mask_batch_n)
    pred_mask_hp = tf.argmax(pred_mask_batch_n, axis=-1)
    pred_mask_formated = pred_mask_hp[..., tf.newaxis]
    return pred_mask_formated

def Show_predictions(model, dataset, nb_images):
    for i, (image, mask) in enumerate(dataset.take(nb_images)):
        pred_mask = model.predict(image)
        # [n] get the image at the n posiition in each batch; n<BATCH_SIZE
        n=0
        display_list = [image[n], mask[n], Create_pred_mask(pred_mask[n])]
        # Display and save the images
        Display_dataset(display_list, save_path=None)
        
from tensorflow.keras.utils import register_keras_serializable

# @register_keras_serializable()
class MultiClassDiceLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, smooth=1e-6, gama=2,reduction=tf.keras.losses.Reduction.AUTO, name='MultiClassDiceLoss'):
        super(MultiClassDiceLoss, self).__init__(reduction=reduction, name=name)
        self.num_classes = num_classes
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        dice_loss = self._calculate_dice_loss(y_true, y_pred)
        return dice_loss / self.num_classes

    def _calculate_dice_loss(self, y_true, y_pred):
        y_true = tf.one_hot(tf.squeeze(tf.cast(y_true, dtype=tf.int32), axis=-1), depth=self.num_classes)
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        dice_loss = 0
        for class_idx in range(self.num_classes):
            class_true = y_true[:, :, :, class_idx]
            class_pred = y_pred[:, :, :, class_idx]

            nominator = 2 * tf.reduce_sum(class_pred * class_true) + self.smooth
            denominator = tf.reduce_sum(class_pred ** self.gama) + tf.reduce_sum(class_true ** self.gama) + self.smooth

            class_dice_loss = 1 - (nominator / denominator)
            dice_loss += class_dice_loss

        return dice_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'smooth': self.smooth,
            'gama': self.gama,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# @register_keras_serializable()
class MultiClassDiceMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='dice_coefficient', **kwargs):
        super(MultiClassDiceMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_dice = self.add_weight('total_dice', initializer='zeros')
        self.num_samples = self.add_weight('num_samples', initializer='zeros')

    def _calculate_dice(self, y_true, y_pred):
        y_true = tf.one_hot(tf.squeeze(tf.cast(y_true, dtype=tf.int32), axis=-1), depth=self.num_classes)
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        dice = 0
        for class_idx in range(self.num_classes):
            class_true = y_true[:, :, :, class_idx]
            class_pred = y_pred[:, :, :, class_idx]

            nominator = 2 * tf.reduce_sum(class_pred * class_true)
            denominator = tf.reduce_sum(class_pred ** 2) + tf.reduce_sum(class_true ** 2)

            class_dice = (nominator + 1e-6) / (denominator + 1e-6)
            dice += class_dice

        return dice

    def update_state(self, y_true, y_pred, sample_weight=None):
        dice = self._calculate_dice(y_true, y_pred)
        self.total_dice.assign_add(tf.reduce_sum(dice))
        self.num_samples.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        return self.total_dice / self.num_samples

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#Metrics : IoU, Precision, Recall
# @register_keras_serializable()
class IoUMetric(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='iou', **kwargs):
        super(IoUMetric, self).__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.cast(y_true, dtype=tf.uint8), axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes', None)
        return cls(num_classes,**config)

# @register_keras_serializable()
class PrecisionMetric(tf.keras.metrics.Precision):
    def __init__(self, num_classes, name='precision', **kwargs):
        super(PrecisionMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.cast(y_true, dtype=tf.uint8), axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes', None)
        return cls(num_classes,**config)

# @register_keras_serializable()
class RecallMetric(tf.keras.metrics.Recall):
    def __init__(self, num_classes, name='recall', **kwargs):
        super(RecallMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.cast(y_true, dtype=tf.uint8), axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes', None)
        return cls(num_classes,**config)