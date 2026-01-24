# 🍎 Fruit Image Classification System

A comprehensive machine learning application for classifying fruit images using **Support Vector Machine (SVM)**, **Decision Tree**, and **K-Nearest Neighbors (KNN)** algorithms with a user-friendly GUI.

## 📋 Features

- **Multiple ML Algorithms**: Train and compare SVM, Decision Tree, and KNN classifiers
- **Custom Dataset Support**: Load your own dataset with any number of fruit classes
- **Feature Extraction**: Automatic extraction of color, texture, and shape features
- **GUI Interface**: Easy-to-use graphical interface built with Tkinter
- **Model Persistence**: Save and load trained models for future use
- **Ensemble Predictions**: Majority voting across all three models
- **Real-time Classification**: Instant predictions with confidence scores

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Instructions

1. **Extract the ZIP file** to your desired location

2. **Open terminal/command prompt** in the extracted folder

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Dataset Structure

Organize your dataset in the following folder structure:

```
dataset/
│
├── apple/
│   ├── apple1.jpg
│   ├── apple2.jpg
│   └── ...
│
├── banana/
│   ├── banana1.jpg
│   ├── banana2.jpg
│   └── ...
│
├── orange/
│   ├── orange1.jpg
│   ├── orange2.jpg
│   └── ...
│
└── grape/
    ├── grape1.jpg
    ├── grape2.jpg
    └── ...
```

**Important Notes:**
- Each subfolder name represents a fruit class
- Place all images of the same fruit type in its corresponding folder
- Supported formats: JPG, JPEG, PNG, BMP
- Recommended: At least 20-30 images per class for better accuracy

## 🎮 Usage

### Running the Application

```bash
python main.py
```

### Step-by-Step Guide

1. **Select Dataset**
   - Click "📁 Select Dataset Folder"
   - Choose the folder containing your organized fruit images
   - The app will show the selected dataset path

2. **Train Models**
   - Click "🚀 Train All Models"
   - Wait for the training process to complete
   - View training accuracies for each model

3. **Classify Images**
   - Click "🖼️ Upload Image"
   - Select a fruit image you want to classify
   - Click "🔍 Classify Image"
   - View predictions from all three models

4. **Save/Load Models** (Optional)
   - **Save**: Click "💾 Save Models" to save trained models
   - **Load**: Click "📂 Load Models" to load previously saved models

## 📊 Model Details

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

## 🔧 Feature Extraction

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

## 📈 Results Interpretation

The application provides:
- Individual predictions from each model
- Confidence scores (probability percentages)
- Probability distribution across all classes
- Ensemble prediction (majority vote)

## 🎯 Tips for Better Accuracy

1. **Dataset Quality**:
   - Use high-quality, clear images
   - Ensure consistent lighting conditions
   - Include variety in angles and backgrounds

2. **Dataset Size**:
   - More images = better accuracy
   - Aim for balanced classes (equal images per fruit)

3. **Image Preprocessing**:
   - Remove blurry or corrupted images
   - Ensure images show the fruit clearly

## 🐛 Troubleshooting

### "No images found in dataset!"
- Check your folder structure
- Ensure images are in supported formats (JPG, PNG, JPEG, BMP)
- Verify subfolders contain actual image files

### Low Accuracy
- Increase dataset size (more images)
- Ensure better image quality
- Check if classes are well-separated visually

### Import Errors
- Reinstall requirements: `pip install -r requirements.txt --upgrade`
- Check Python version: `python --version` (should be 3.7+)

## 📝 Example Dataset

You can use public fruit datasets:
- Kaggle Fruit 360 Dataset
- Fruits 262 Dataset
- Or create your own by collecting images

## 🔒 Model Files

Saved model files (.pkl) contain:
- Trained SVM, Decision Tree, and KNN models
- Feature scaler
- Class names

**Note**: Keep these files safe to avoid retraining!

## 📚 Dependencies

- **NumPy**: Numerical computations
- **OpenCV**: Image processing
- **scikit-learn**: Machine learning algorithms
- **Pillow**: Image handling in GUI
- **Matplotlib**: Plotting (if needed)
- **Tkinter**: GUI framework (included with Python)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset structure is correct

## 📄 License

This project is free to use for educational and personal purposes.

## 🌟 Features Summary

✅ Three ML algorithms in one application  
✅ Custom dataset support  
✅ User-friendly GUI  
✅ Feature extraction pipeline  
✅ Model save/load functionality  
✅ Ensemble predictions  
✅ Detailed accuracy metrics  

## 🎓 Educational Value

Perfect for:
- Learning machine learning concepts
- Understanding image classification
- Comparing different ML algorithms
- Building practical AI applications

---

**Made with ❤️ for Machine Learning Education**