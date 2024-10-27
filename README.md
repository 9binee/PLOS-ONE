# PLOS-ONE

This repository contains the codes used in our PLOS ONE paper:  
**"Multimodal feature fusion-based graph convolutional networks for Alzheimer’s disease stage classification using F-18 florbetaben brain PET images and clinical indicators."**

## Files Description

### Model Implementations
- **3D_DenseNet.py**  
  Contains the implementation of the 3D DenseNet model for Alzheimer's Disease stage classification.

- **GCN.py**  
  Includes the implementation of the Graph Convolutional Network (GCN) model.

- **MLP_3HL.py**  
  Implements a Multi-Layer Perceptron model with 3 hidden layers.

### Training, Validation, and Testing Scripts
- **CNNs_train_valid_test_learning.py**  
  Provides functions for training, validation, testing, and learning for Convolutional Neural Networks (CNNs).

- **GNNs_train_valid_test_learning.py**  
  Provides functions for training, validation, testing, and learning for Graph Neural Networks (GNNs).

- **MLPs_train_valid_test_learning.py**  
  Provides functions for training, validation, testing, and learning for Multi-Layer Perceptron (MLP) models.

### Data Preprocessing and Graph Construction
- **Graph_Construction.py**  
  Contains functions for graph construction using either cosine similarity or Euclidean distance to create edge weights between combined image and non-image data.

- **Image_Preprocessing.py**  
  Handles loading and resizing of NIfTI images.

---

These files collectively cover the implementation of models, training procedures, and necessary preprocessing steps used in our PLOS ONE paper.
