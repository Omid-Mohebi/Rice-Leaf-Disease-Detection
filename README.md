# Rice Leaf Disease Detection

This repository contains a project aimed at detecting diseases in rice leaves using deep learning. The dataset and implementation focus on building a convolutional neural network (CNN) to classify images of rice leaves into different disease categories.

## Dataset

The dataset is sourced from Kaggle: [Rice Leaf Disease Image Dataset](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image). It contains images of rice leaves categorized into different classes representing healthy and diseased conditions.

### Dataset Details:

- **Classes**: Multiple categories for healthy and diseased leaves.
- **Format**: JPEG and PNG images.
- **Size**: Resized to 128x128 for this project.

## Project Structure

The implementation is structured in a Jupyter Notebook, which includes the following key steps:

### 1. Data Preparation

- **Loading Dataset**: Downloaded using `kagglehub` and structured into class folders.
- **Preprocessing**:
  - Resizing images to 128x128 pixels.
  - Converting to RGB format.
  - Label encoding for classification.

### 2. Data Visualization

- Displayed sample images using Matplotlib for better understanding of the dataset.

### 3. Model Training

- **Architecture**:
  - Convolutional layers with ReLU activation.
  - MaxPooling for feature extraction.
  - Dense layers for classification.
- **Compilation**:
  - Optimizer: Adam.
  - Loss Function: Sparse Categorical Crossentropy.
- **Training**:
  - 15 epochs.
  - Training/Validation split: 67%/33%.

### 4. Results

- Plotted training and validation loss and accuracy over epochs.
- Achieved high accuracy on the validation set, indicating effective disease classification.

### 5. Key Metrics

- **Validation Accuracy**: Over 96% in the final epochs.
- **Loss**: Reduced significantly over training.
  
![image](https://github.com/user-attachments/assets/8b41e37f-aea8-4904-9a5b-29ebb76ed0ab)

![image](https://github.com/user-attachments/assets/ddbc8087-1296-4b42-b0d4-a406c3b23c72)

## Code Example

### Data Preprocessing

```python
images = []
labels = []

for class_name in class_names:
    class_folder = os.path.join(path, class_name)

    for filename in os.listdir(class_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(class_folder, filename)

            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize((128, 128))
            img_array = np.array(img)

            images.append(img_array)
            labels.append(class_name)

images_array = np.array(images)
labels_array = np.array(labels)
```

### Model Definition

```python
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model

K = len(set(y_train))

i = Input(shape=X_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)
```

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

**P.S**: I've also added the Python export of the Jupyter notebook it the repository.
