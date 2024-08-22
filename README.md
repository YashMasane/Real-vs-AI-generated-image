# AI vs Real Image Classification 🖼️🤖


<p align="center">
  <img src="/img/real_elephent_demo.png" alt="Real Image" width="400"/>
  <img src="/img/ai_dog_demo.png" alt="AI generated Image" width="400"/>
</p>

## Project Overview 🌟

This project aims to classify images as either AI-generated or real. The quality of AI-generated images has rapidly increased, leading to concerns about authenticity and trustworthiness. The project leverages convolutional neural networks (CNNs) for feature extraction and classification, with additional components such as data augmentation and regularization to enhance performance.
 
 ***Try application [here](https://ai-generated-image-detector.onrender.com/)***.
 
 The app allows you to upload images and classify them as either AI-generated or real.


### Usefulness of the Project 📈

1. **Enhanced Detection of AI-Generated Content**: As AI technology advances, the ability to create realistic synthetic images becomes more sophisticated. This project helps identify and classify such images, providing valuable tools for detecting AI-generated content.

2. **Applications in Various Fields**:
   - **Security and Fraud Prevention**: Helps in detecting manipulated or fake images in security systems, preventing potential fraud and misuse.
   - **Media and Journalism**: Assists journalists and media outlets in verifying the authenticity of images before publication, ensuring credibility and trustworthiness.
   - **Content Moderation**: Useful for platforms that need to filter and manage user-generated content, ensuring that AI-generated images are identified and handled appropriately.

3. **Foundation for Further Research**: Provides a baseline model and dataset for researchers and developers working on similar projects, enabling further advancements in image classification and detection techniques.

4. **Educational Purpose**: Offers insights into the application of deep learning techniques for image classification, serving as a valuable resource for learning and experimentation in the field of computer vision.

## Dataset 📚

The quality of AI-generated images has rapidly increased, leading to concerns of authenticity and trustworthiness. CIFAKE is a dataset designed to address this issue, containing:

- **60,000 synthetically-generated images** (FAKE)
- **60,000 real images** (REAL) collected from the CIFAR-10 dataset.

The dataset aims to explore if computer vision techniques can effectively detect whether an image is real or AI-generated.

- **REAL images**: Collected from Krizhevsky & Hinton's CIFAR-10 dataset.
- **FAKE images**: Generated using Stable Diffusion version 1.4.


You can access and download the dataset from the link below:

- [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

## Model Architecture 🏗️

**EfficientNetB6** is used as the base model for feature extraction due to its efficiency and high performance. EfficientNetB6 provides a good balance between model size and performance, making it suitable for this project. For more information and to explore other models, you can visit the [Keras Applications page](https://keras.io/api/applications/).

## Project Workflow 🔄

1. **Dataset Collection**:
   - The CIFAKE dataset is used, consisting of labeled images, where each image is classified as either AI-generated or real. The dataset is split into training, validation, and test sets.

2. **Data Preprocessing**:
   - Images are resized to a uniform size suitable for the model.
   - Data augmentation techniques, including random flips, brightness adjustments, and contrast adjustments, are applied to increase the diversity of the training data.

3. **Model Architecture**:
   - **Base Model**: EfficientNetB6 is used as the base model for feature extraction. EfficientNetB6 is known for its efficiency and performance in image classification tasks.
   - **Custom ANN**: An artificial neural network (ANN) is built on top of the EfficientNetB6 model for classification purposes.
   - **Regularization**: Batch normalization, dropout layers, and L2 regularization are employed to prevent overfitting and enhance generalization.

4. **Training**:
   - The model is trained for 15 epochs based on validation performance to prevent overfitting. The best weights are saved based on the validation accuracy.

5. **Evaluation**:
   - The model's performance is evaluated on the test set. Metrics such as accuracy, precision, recall, F1-score are used to assess the model's effectiveness.
   - **Test Loss:** `0.16084866225719452`  
   - **Test Binary Accuracy:** `0.941047191619873`  
   - **Test Precision:** `0.9449597001075745`  
   - **Test Recall:** `0.9366506934165955`
   
<p align="center">
  <img src="/img/trf_acc.png" alt="Real Image" width="400"/>
  <img src="/img/trf_loss.png" alt="AI generated Image" width="400"/>
</p>

6. **Deployment**:
   - The trained model is deployed in a production environment using a Streamlit application. The app provides an interface for users to upload images and classify them as AI-generated or real.


## Getting Started 🚀

### Prerequisites

- Python 3.x
- TensorFlow (GPU or CPU version)
- Streamlit
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-vs-real-image-classification.git
   cd ai-vs-real-image-classification

2. Install the required packages:
    
   pip install -r requirements.txt
   
## Usage 🚀

### Running the Streamlit Application

1. **Run the Streamlit Application**:
   ```bash
   streamlit run app/app.py
   
This will start a local server and open the Streamlit application in your web browser.

### Upload an Image:
- Use the Streamlit app interface to upload an image.
- The app will classify the image as either AI-generated or real and display the result.

## Project Structure 📁

```bash
ai-vs-real-image-classification/
│
├── app                      # Streamlit application
│   ├── app.py               # user app
│   ├── ui.py                # user interface
│   └── my_model.h5          # Trained model weights
|
|                                    
├── data/                    # Directory for dataset
│   ├── train/               # Training images       
│   └── test/                # Test images
|
├── ai_real.ipynb            # ipynb file for model
├── requirements.txt         # List of required packages
└── README.md                # Project documentation

```

## Contributing 🤝

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss what you would like to change.   

## Acknowledgements 🙏

- **[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)**
- **EfficientNetB6 model pre-trained weights**
- **[Keras Applications](https://keras.io/api/applications/)**
- **Streamlit for creating interactive web applications**
- **TensorFlow for providing robust tools for deep learning**

   

