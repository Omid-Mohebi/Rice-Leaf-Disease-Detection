# Rice Leaf Disease Detection

This repository contains a project aimed at detecting diseases in rice leaves using deep learning. The dataset and implementation focus on building a convolutional neural network (CNN) to classify images of rice leaves into different disease categories.

## Dataset

The dataset is sourced from Kaggle: [Rice Leaf Disease Image Dataset](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image). It contains images of rice leaves categorized into different classes representing healthy and diseased conditions.

### Dataset Details:

- **Classes**: Multiple categories for healthy and diseased leaves.
- **Format**: JPEG and PNG images.
- **Size**: Resized to 128x128 for this project.

## Project Structure

The implementation is structured in Jupyter Notebooks, with some variations in architecture. The key steps include:

### 1. Data Preparation

- **Loading Dataset**: Downloaded using `kagglehub` and structured into class folders.
- **Preprocessing**:
  - Resizing images to 128x128 pixels.
  - Converting to RGB format.
  - Label encoding for classification.

### 2. Data Visualization

- Displayed sample images using Matplotlib for better understanding of the dataset.

### 3. Model Training

- **Architecture** (File without MaxPooling):
  - Convolutional layers with ReLU activation.
  - No MaxPooling layers.
  - Dense layers for classification.

  ```python
  x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
  x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
  x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
  x = Flatten()(x)
  ```
  
- **Architecture** (File with MaxPooling):
  - Convolutional layers with ReLU activation.
  - MaxPooling2D layers after each convolutional layer for feature reduction.
  - Dense layers for classification.

  ```python
  x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
  x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
  x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
  x = Flatten()(x)
  ```

- **Compilation**:
  - Optimizer: Adam.
  - Loss Function: Sparse Categorical Crossentropy.
- **Training**:
  - 15 epochs.
  - Training/Validation split: 67%/33%.

### 4. Results and Training Comparison

#### File without MaxPooling
- **Training Time per Epoch**: ~83-140 seconds
- **Validation Accuracy (Final Epoch)**: 96.45%
- **Loss (Final Epoch)**: 0.1480

![image](https://github.com/user-attachments/assets/dfe0b437-40f7-433a-a3f9-e47cc707ddac)

![image](https://github.com/user-attachments/assets/3b6aa0f3-77f0-4f94-813c-b2cdc9098ebc)
  
#### File with MaxPooling
- **Training Time per Epoch**: ~20 seconds
- **Validation Accuracy (Final Epoch)**: 96.03%
- **Loss (Final Epoch)**: 0.1650

![image](https://github.com/user-attachments/assets/e949fbc2-ce8f-43ec-8d7e-6290c82e5cd6)

![image](https://github.com/user-attachments/assets/8bfadc52-93db-4447-b7d4-06c6cdae9706)

### Observations
1. **Training Efficiency**:
   - The model without MaxPooling required up to **140 seconds** per epoch due to the absence of dimensionality reduction.
   - The model with MaxPooling trained significantly faster, completing each epoch in about **20 seconds**.

2. **Performance**:
   - The non-MaxPooling model also had a marginally lower final loss.
   - Both models achieved comparable validation accuracies, with the non-MaxPooling model slightly outperforming at **96.45% vs 96.03%**.

3. **Trade-Off**:
   - The absence of MaxPooling leads to finer feature extraction but at a higher computational cost.
   - The MaxPooling model is more efficient, trading a minor decrease in performance for significant gains in training speed.

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Omid-Mohebi/Rice-Leaf-Disease-Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Acknowledgements

- **Dataset**: [Nirmal Sankalana on Kaggle](https://www.kaggle.com/nirmalsankalana).
- **Tools**: TensorFlow, Scikit-learn, Matplotlib.

---

Feel free to contribute by opening issues or pull requests!

This code structure can also be adapted for more complex image processing tasks in the healthcare field and other domains, enabling broader applications beyond rice leaf disease detection.

**P.S**: The repository contains the Python export of the Jupyter notebooks for easy reproducibility.

