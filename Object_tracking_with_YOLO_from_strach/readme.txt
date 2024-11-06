Object Detection with YOLO Architecture using TensorFlow and Keras
This project implements a YOLO (You Only Look Once)-based object detection model using TensorFlow and Keras. The code preprocesses image and XML annotation data to train a model that can detect various objects in images. It leverages the EfficientNetB1 model as a feature extractor and customizes the architecture to predict bounding boxes and class labels for objects in an image.

Overview
Framework: TensorFlow/Keras
Preprocessing: Parses XML files to extract bounding boxes and class information.
Model Architecture: EfficientNetB1 backbone followed by convolutional and fully connected layers to predict bounding boxes and classes.
Loss Function: Custom YOLO loss function, including object presence, no-object, and bounding box regression losses.
Installation
Clone the repository:

bash
Kodu kopyala
git clone https://github.com/yourusername/object-detection-yolo.git
cd object-detection-yolo
Install dependencies:

bash
Kodu kopyala
pip install -r requirements.txt
Download dataset: Place images and XML annotation files in appropriate directories for training and validation.

Files and Functions
Core Components
preprocess_xml(filename): Parses XML annotations, returning bounding box coordinates and class IDs.
generate_output(bounding_boxes): Creates a target tensor of shape (7, 7, 25) to match the model's output format.
get_imbboxes(im_path, xml_path): Reads images, resizes them, and pairs them with their respective bounding boxes.
Model Structure
EfficientNetB1 Backbone: Utilizes EfficientNetB1 as the base model.
Custom Layers: Adds convolutional layers with LeakyReLU activations, followed by dense and reshape layers to output bounding boxes and class predictions.
Training
Optimizer: Adam
Callbacks:
Learning Rate Scheduler: Adjusts learning rate during training.
ModelCheckpoint: Saves the model at each epoch.
Loss Function: yolo_loss function, which computes object, no-object, and class loss, as well as bounding box loss using IoU (Intersection over Union).
Custom Loss: yolo_loss
The yolo_loss function combines multiple components:

Object Loss: Penalizes the network for inaccurate object predictions.
No-object Loss: Reduces false positives by penalizing predictions in regions without objects.
Bounding Box Loss: Computes IoU between predicted and target boxes, focusing on accurate object localization.
Class Loss: Ensures accurate class predictions.
Usage
To train the model, ensure the dataset paths are correctly set, then run:

python
Kodu kopyala
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=90,
    callbacks=[lr_callback, callback]
)
Model Evaluation and Prediction
The trained model can be evaluated on a test dataset to compute its accuracy in detecting and classifying objects within images.