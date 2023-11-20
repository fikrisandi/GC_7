import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import streamlit as st
import pandas as pd
import glob
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

def app():
    st.markdown(
    """
    <style>
        .skyblue-text {
            text-align: center;
            color: #87CEEB; /* kode hex untuk skyblue */
        }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 class="skyblue-text">Exploratory Data Analysis (EDA)</h1>
        """,
        unsafe_allow_html=True
    )

    # Menggunakan HTML untuk mengatur teks lebih rapi
    st.markdown(
        """
        <h3 style='text-align: center;'>Informasi Dataset</h3>
        <p  style='text-align: justify;'>Dataset yang digunakan yaitu <b class="skyblue-text">Pothole Detection Dataset</b>, dataset ini berisi gambar jalan normal dan jalan berlubang. Dataset ini sudah dibagi menjadi data untuk trainin, validation, dan testing. <br><br>
        Dataset ini digunakan untuk melakukan classification <b class="skyblue-text">image prepocessing</b> mengggunakan metode <b class="skyblue-text">Convolutional Neural Network</b>. Pada dataset ini nilai `1` (positif) ditandakan dengan gambar jalan normal, sedangkan untuk nilai `0` ditandakan nilai (negatif). Evaluasi metrics yang digunakan yaitu memfokuskan pada ... <br><br>
        Model ini akan digunakan untuk mengetahui jalan yang berlubang atau tidak sehingga dapat digunakan untuk kebutuhan perbaikan jalan kedepannya nanti, agar jalanan tidak menimbulkan kecelakaan.</p>
        """,
        unsafe_allow_html=True  # Mengizinkan penggunaan HTML
    )

    st.markdown('---')
    
    def display_images(dataset_path, category):
        # Get file paths of normal and pothole images
        files = [os.path.join(dataset_path, category, filename) for filename in os.listdir(os.path.join(dataset_path, category))][:8]

        # Display the images using Streamlit
        fig, ax = plt.subplots(ncols=8, figsize=(30, 3))

        for i, image_file in enumerate(files):
            image = Image.open(image_file)
            ax[i].imshow(image)
            ax[i].axis('off')

        # Show the images in the Streamlit app
        st.pyplot(fig)

    # Streamlit App

    st.markdown(
        """
        <h3 style='text-align: center;'>Visualisasi Dataset</h3>
        """,
        unsafe_allow_html=True  # Mengizinkan penggunaan HTML
    )

    # Dropdown for selecting dataset (training or validation)
    selected_dataset = st.selectbox("Select Dataset", ["Training", "Validation"])

    # Dropdown for selecting image category
    selected_category = st.selectbox("Select Image Category", ["Normal", "Pothole"])

    # Display the images based on user's selection
    if selected_dataset == "Training":
        dataset_path_train = '../Dataset/Train/'
        display_images(dataset_path_train, selected_category)
    elif selected_dataset == "Validation":
        dataset_path_val = '../Dataset/Val/'
        display_images(dataset_path_val, selected_category)
        
        
    st.markdown('---')
    
    st.markdown(
        """
        <h3 style='text-align: center;'>Distribusi Data</h3>\
        """,
        unsafe_allow_html=True  # Mengizinkan penggunaan HTML
    )
    
    def display_label_distribution(dataset_path):
        # Get labels
        labels = [os.path.basename(label) for label in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(label)]

        # Count the number of images for each label
        label_distribution = Counter()

        for label in labels:
            label_folder = os.path.join(dataset_path, label)
            num_images = len(glob.glob(os.path.join(label_folder, '*')))
            label_distribution[label] = num_images

        # Display the distribution using Streamlit bar chart
        st.bar_chart(label_distribution)

        # Optional: Display the distribution in tabular format
        st.write("Label Distribution Table:")
        st.table(label_distribution)

    # Button to choose dataset distribution
    selected_dataset = st.selectbox("Select Dataset Distribution", ["Train", "Validation"])

    # Display the selected dataset distribution
    if selected_dataset == "Train":
        st.write("Diagram Distribution Data Train:")
        display_label_distribution(dataset_path_train)
    elif selected_dataset == "Validation":
        st.write("Diagram Distribution Data Validation:")
        display_label_distribution(dataset_path_val)
    
        
    st.markdown('---')
    
    # Create an ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        brightness_range=[0.2, 1.2]
    )

    # Generate augmented images using the ImageDataGenerator
    train_generator = datagen.flow_from_directory(
        '../Dataset/Train/',
        target_size=(64, 64),
        batch_size=9,
        class_mode='categorical'
    )

    # Streamlit App
    st.markdown(
        """
        <h3 style='text-align: center;'>Visualisasi data yang diagmentasi pada train set</h3>
        """,
        unsafe_allow_html=True  # Mengizinkan penggunaan HTML
    )
    
    # Display the images using Streamlit
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    n = 0

    for i in range(3):
        for j in range(3):
            img = train_generator.next()[0][n].astype('uint8')
            ax[i][j].imshow(img)
            ax[i][j].set_title(np.argmax(train_generator.next()[1][n]))
            n += 1

    # Remove axis labels
    for ax_row in ax:
        for ax_i in ax_row:
            ax_i.axis('off')

    # Show the images in the Streamlit app
    st.pyplot(fig)
