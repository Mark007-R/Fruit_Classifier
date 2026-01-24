# Fruit Image Classification System

A comprehensive machine learning application for classifying fruit images using **Support Vector Machine (SVM)**, **Decision Tree**, and **K-Nearest Neighbors (KNN)** algorithms with a user-friendly GUI.

## Features

- **Multiple ML Algorithms**: Train and compare SVM, Decision Tree, and KNN classifiers
- **Custom Dataset Support**: Load your own dataset with any number of fruit classes
- **Feature Extraction**: Automatic extraction of color, texture, and shape features
- **GUI Interface**: Easy-to-use graphical interface built with Tkinter
- **Model Persistence**: Save and load trained models for future use
- **Ensemble Predictions**: Majority voting across all three models
- **Real-time Classification**: Instant predictions with confidence scores




## Model Details

### Support Vector Machine (SVM)
- Kernel: Radial Basis Function (RBF)
- Finds optimal hyperplane to separate classes
- Best for: High-dimensional data

### Decision Tree
- Max depth: 10
- Rule-based classification
- Best for: Interpretable results

### K-Nearest Neighbors (KNN)
- K = 5 neighbors
- Distance-based classification
- Best for: Simple, effective classification

## Feature Extraction

The system extracts 11 features from each image:

1. **Color Features** (6):
   - Mean RGB values
   - Standard deviation of RGB channels

2. **Texture Features** (4):
   - Histogram mean
   - Histogram standard deviation
   - Texture energy
   - Texture entropy

3. **Shape Features** (1):
   - Edge density

## Results Interpretation

The application provides:
- Individual predictions from each model
- Confidence scores (probability percentages)
- Probability distribution across all classes
- Ensemble prediction (majority vote)




## Model Files

Saved model files (.pkl) contain:
- Trained SVM, Decision Tree, and KNN models
- Feature scaler
- Class names

**Note**: Keep these files safe to avoid retraining!

## Dependencies

- **NumPy**: Numerical computations
- **OpenCV**: Image processing
- **scikit-learn**: Machine learning algorithms
- **Pillow**: Image handling in GUI
- **Matplotlib**: Plotting (if needed)
- **Tkinter**: GUI framework (included with Python)


## License

This project is free to use for educational and personal purposes.

## Features Summary

Three ML algorithms in one application  
Custom dataset support  
User-friendly GUI  
Feature extraction pipeline  
Model save/load functionality  
Ensemble predictions  
Detailed accuracy metrics  

## Educational Value

Perfect for:
- Learning machine learning concepts
- Understanding image classification
- Comparing different ML algorithms
- Building practical AI applications


**Made for Machine Learning Education**