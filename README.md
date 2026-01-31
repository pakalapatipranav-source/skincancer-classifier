# 🏥 AI-Powered Skin Cancer Detection System

> A deep learning application that classifies skin lesions as benign or malignant using transfer learning, achieving **79.2% test accuracy** with a professional web interface for real-time predictions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

This project implements a state-of-the-art deep learning system for skin cancer detection, leveraging transfer learning with EfficientNetB0 to classify skin lesions as benign or malignant. The system includes a complete machine learning pipeline from data preprocessing to deployment, featuring a modern web interface for real-time predictions.

**Key Achievement**: Achieved **79.2% test accuracy** with **90.6% accuracy** for benign lesions and **65.7% accuracy** for malignant lesions, demonstrating effective handling of imbalanced medical datasets.

### Real-World Impact

This project addresses a critical healthcare challenge: early detection of skin cancer. By providing an accessible tool for preliminary skin lesion analysis, the system has the potential to:
- Assist healthcare professionals in preliminary screening
- Increase awareness about skin cancer detection
- Serve as an educational tool for understanding AI in healthcare
- Demonstrate the practical application of deep learning in medical imaging

---

## ✨ Project Highlights

### Technical Achievements

- **Transfer Learning**: Implemented EfficientNetB0 pre-trained on ImageNet for feature extraction
- **Advanced Architecture**: Custom classification head with GlobalAveragePooling2D, batch normalization, and dropout regularization
- **Two-Phase Training**: Frozen base model training followed by fine-tuning for optimal performance
- **Class Imbalance Handling**: Developed aggressive inverse-frequency weighting system (2x amplification) to address dataset imbalance
- **Data Augmentation**: Comprehensive augmentation pipeline (rotation, shifts, zoom, flips) for improved generalization
- **Professional Web Interface**: Modern Streamlit application with real-time predictions and confidence visualization

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **79.2%** |
| **Benign Accuracy** | 90.6% |
| **Malignant Accuracy** | 65.7% |
| **Test Loss** | 0.387 |
| **Training Samples** | 2,109 |
| **Validation Samples** | 528 |
| **Test Samples** | 660 |

### Technical Stack

- **Deep Learning**: TensorFlow/Keras, EfficientNetB0
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Image Processing**: PIL/Pillow

---

## 🚀 Features & Capabilities

- **Real-Time Classification**: Upload skin lesion images and receive instant predictions
- **Confidence Visualization**: Interactive charts showing prediction confidence for each class
- **Professional UI**: Modern, intuitive interface designed for both technical and non-technical users
- **Robust Error Handling**: Comprehensive validation for file types, sizes, and image formats
- **Model Metadata Display**: View training details, accuracy metrics, and model information
- **Medical Disclaimers**: Appropriate warnings about the research nature of the tool
- **Batch Processing Ready**: Architecture supports future batch prediction capabilities

---

## 🏗️ Technical Architecture

### Model Architecture

The system uses a sophisticated transfer learning approach:

```
Input Image (224×224×3 RGB)
    ↓
EfficientNetB0 Base (Frozen in Phase 1)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
BatchNormalization
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(2) + Softmax
    ↓
Output: [Benign Probability, Malignant Probability]
```

**Key Architectural Decisions**:
- **GlobalAveragePooling2D**: More efficient than Flatten, reduces overfitting
- **Batch Normalization**: Stabilizes training and improves convergence
- **Progressive Dropout**: Higher dropout (0.5) in first layer, lower (0.3) in second
- **Two Dense Layers**: Allows model to learn complex feature combinations

### Training Strategy

**Phase 1: Frozen Base Training**
- EfficientNetB0 base model frozen (ImageNet weights)
- Only custom classification head trained
- Learning rate: 0.0001
- Focus: Learn dataset-specific patterns

**Phase 2: Fine-Tuning**
- Unfreeze last 30 layers of EfficientNetB0
- Lower learning rate: 0.00001
- Fine-tune features for skin lesion classification
- Focus: Adapt pre-trained features to medical domain

**Training Enhancements**:
- **Data Augmentation**: Rotation (±20°), shifts (±20%), zoom (±20%), horizontal flips
- **Class Weighting**: Inverse-frequency weighting with 2x amplification for minority class
- **Callbacks**: Early stopping, learning rate reduction on plateau, model checkpointing
- **Preprocessing**: EfficientNet-specific preprocessing (normalizes to [-1, 1] range)

---

## 🧩 Challenges & Solutions

This section highlights the problem-solving and iterative improvement process that led to the final model.

### Challenge 1: Class Imbalance Causing Single-Class Predictions

**Problem**: Initial model predicted only the majority class (benign) with ~54% confidence, achieving essentially random performance.

**Root Cause**: Dataset imbalance (1,440 benign vs 1,197 malignant) combined with insufficient class weighting.

**Solution**: 
- Implemented aggressive inverse-frequency class weighting
- Calculated weights as: `weight = (1 / frequency) × 2.0`
- Result: Class weights of 3.70 (benign) and 4.35 (malignant)
- **Impact**: Model learned to predict both classes, accuracy improved from 54.5% to 79.2%

### Challenge 2: Preprocessing Mismatch Between Training and Inference

**Problem**: Training used EfficientNet's `preprocess_input()` (normalizes to [-1, 1]), while inference used simple `/255.0` normalization (to [0, 1]), causing prediction errors.

**Solution**:
- Standardized preprocessing across all components
- Updated both training script and web application to use `preprocess_input()`
- Ensured consistent preprocessing pipeline from data loading to prediction
- **Impact**: Predictions became accurate and consistent with training performance

### Challenge 3: Model Not Learning Effectively

**Problem**: Validation accuracy stuck at ~57% with no improvement, loss not decreasing.

**Solution**:
- Implemented two-phase training strategy
- Added learning rate scheduling (ReduceLROnPlateau)
- Implemented early stopping to prevent overfitting
- Reduced initial learning rate from 0.001 to 0.0001 for stability
- **Impact**: Model converged properly, validation accuracy reached 87.5%

### Challenge 4: Label Encoding Issues for Binary Classification

**Problem**: LabelBinarizer returned shape (n, 1) for binary classification, causing model to have only 1 output neuron instead of 2.

**Solution**:
- Explicitly converted to one-hot encoding using `to_categorical()` with `num_classes=2`
- Added validation checks to ensure correct label shape
- **Impact**: Model correctly outputs probabilities for both classes

---

## 📈 Results & Performance

### Test Set Performance

The final model achieves strong performance on the test set:

```
Test Accuracy: 79.2%
Test Loss: 0.387
```

### Confusion Matrix

|                | Predicted Benign | Predicted Malignant |
|----------------|------------------|---------------------|
| **Actual Benign** | 326 (90.6%) | 34 (9.4%) |
| **Actual Malignant** | 103 (34.3%) | 197 (65.7%) |

### Per-Class Accuracy

- **Benign Lesions**: 90.6% accuracy (326 correct out of 360)
- **Malignant Lesions**: 65.7% accuracy (197 correct out of 300)

### Training Progress

The model showed consistent improvement:
- Phase 1: Validation accuracy improved from 71% to 85%
- Phase 2: Fine-tuning further improved to 87.5% validation accuracy
- Final test accuracy: 79.2% (strong generalization)

---

## 📁 Project Structure

```
ml_skincancer/
├── skin_cancer_detection_train.py  # Main training script with two-phase training
├── streamlit_app.py                 # Professional web application (recommended)
├── app.py                           # Flask web application (legacy)
├── utils/
│   ├── __init__.py
│   └── model_utils.py               # Model loading and preprocessing utilities
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
├── static/                          # Static web assets (Flask)
│   ├── css/
│   │   └── style.css
│   └── uploads/
├── templates/                       # HTML templates (Flask)
│   └── index.html
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
│
├── Generated Files (after training):
│   ├── skin_cancer_model.h5         # Trained model
│   ├── best_model.h5                # Best model from Phase 1
│   ├── best_model_phase2.h5        # Best model from Phase 2
│   ├── label_binarizer.pkl         # Label encoder
│   ├── class_names.json            # Class names
│   ├── model_metadata.json         # Training metadata
│   └── training_history.json       # Training metrics
│
└── SkinCancerDS/                    # Dataset (not included in repo)
    ├── train/
    │   ├── benign/                 # 1,440 training images
    │   └── malignant/               # 1,197 training images
    └── test/
        ├── benign/                 # 360 test images
        └── malignant/               # 300 test images
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8-3.12 (TensorFlow compatibility)
- pip package manager
- 4GB+ RAM (8GB+ recommended for training)
- GPU optional but recommended for faster training

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ml_skincancer
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   - Organize images in the following structure:
   ```
   SkinCancerDS/
   ├── train/
   │   ├── benign/
   │   │   └── *.jpg
   │   └── malignant/
   │       └── *.jpg
   └── test/
       ├── benign/
       │   └── *.jpg
       └── malignant/
           └── *.jpg
   ```

---

## 💻 Usage Instructions

### Training the Model

Train the model with default settings (15 epochs per phase, batch size 32):

```bash
python skin_cancer_detection_train.py
```

**Custom Training Options**:

```bash
# Specify epochs and batch size
python skin_cancer_detection_train.py --epochs 20 --batch-size 64

# Disable two-phase training (only Phase 1)
python skin_cancer_detection_train.py --no-two-phase

# Disable data augmentation
python skin_cancer_detection_train.py --no-augmentation

# Custom data directory
python skin_cancer_detection_train.py --data-dir /path/to/dataset
```

**Training Output**:
- Model files saved automatically
- Training metrics logged to console
- Best models saved at each phase
- Comprehensive evaluation metrics displayed

### Running the Web Application

**Streamlit Application (Recommended)**:

```bash
streamlit run streamlit_app.py
```

The application will open at `http://localhost:8501`

**Features**:
- Drag-and-drop image upload
- Real-time predictions with confidence scores
- Interactive confidence visualization
- Model metadata display
- Medical disclaimers and recommendations

**Flask Application (Legacy)**:

```bash
python app.py
```

Available at `http://localhost:5000`

---

## 🎓 Technical Skills Demonstrated

### Deep Learning
- **Convolutional Neural Networks (CNNs)**: Understanding of CNN architecture and feature extraction
- **Transfer Learning**: Implementation of pre-trained models (EfficientNetB0) for domain adaptation
- **Fine-Tuning**: Two-phase training strategy with layer unfreezing
- **Regularization**: Dropout, batch normalization, and early stopping
- **Optimization**: Learning rate scheduling and adaptive optimizers (Adam)

### Machine Learning
- **Classification**: Binary classification for medical diagnosis
- **Imbalanced Data Handling**: Advanced class weighting techniques
- **Evaluation Metrics**: Accuracy, confusion matrix, per-class metrics
- **Cross-Validation**: Train/validation/test split strategy
- **Hyperparameter Tuning**: Learning rates, batch sizes, dropout rates

### Software Engineering
- **Modular Code Design**: Separated utilities, training, and application code
- **Error Handling**: Comprehensive exception handling and validation
- **Web Development**: Streamlit and Flask applications
- **Code Documentation**: Clear comments and docstrings
- **Version Control**: Git workflow and repository management

### Data Science
- **Data Preprocessing**: Image normalization, resizing, and augmentation
- **Feature Engineering**: Transfer learning for feature extraction
- **Data Pipeline**: End-to-end ML pipeline from raw data to predictions
- **Model Evaluation**: Comprehensive metrics and visualization
- **Reproducibility**: Consistent preprocessing and training procedures

---

## 🔮 Future Improvements

- **Multi-Class Classification**: Extend to specific cancer types (melanoma, basal cell carcinoma, etc.)
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Explainability**: Add Grad-CAM or similar techniques to visualize model attention
- **Mobile Deployment**: Optimize model for mobile devices using TensorFlow Lite
- **API Development**: Create RESTful API for integration with other systems
- **Database Integration**: Store predictions and user feedback for continuous improvement
- **Advanced Augmentation**: Implement mixup, cutout, or other advanced techniques
- **Model Versioning**: Implement MLflow or similar for experiment tracking
- **Docker Containerization**: Package application for easy deployment
- **Performance Optimization**: Model quantization and pruning for faster inference

---

## 📚 Learning Outcomes

### Technical Knowledge Gained

- **Deep Learning Fundamentals**: Gained hands-on experience with CNNs, transfer learning, and fine-tuning
- **Medical AI Applications**: Understanding of applying ML to healthcare challenges
- **Data Preprocessing**: Mastered image preprocessing, augmentation, and normalization techniques
- **Model Evaluation**: Learned to interpret confusion matrices, per-class metrics, and validation curves
- **Class Imbalance**: Developed strategies for handling imbalanced datasets in medical contexts

### Problem-Solving Skills

- **Debugging ML Models**: Identified and resolved issues with model convergence and prediction accuracy
- **Iterative Improvement**: Systematically improved model from 54% to 79% accuracy through experimentation
- **Preprocessing Consistency**: Recognized importance of consistent preprocessing across training and inference
- **Architecture Design**: Made informed decisions about model architecture and training strategies

### Software Development

- **Full-Stack ML Application**: Built complete system from data processing to web deployment
- **Code Organization**: Learned to structure ML projects for maintainability and scalability
- **User Interface Design**: Created intuitive interfaces for non-technical users
- **Documentation**: Developed skills in technical writing and project documentation

---

## 🙏 Acknowledgments & Resources

### Datasets
- Skin cancer image dataset organized into train/test splits with benign and malignant classes

### Libraries & Frameworks
- **TensorFlow/Keras**: Deep learning framework
- **EfficientNet**: Pre-trained model architecture
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning utilities
- **Plotly**: Interactive visualizations
- **PIL/Pillow**: Image processing

### Learning Resources
- Transfer learning concepts from TensorFlow documentation
- EfficientNet architecture papers and implementations
- Medical AI best practices and ethical considerations
- Class imbalance handling techniques from ML literature

---

## ⚠️ Important Notes

- **Medical Disclaimer**: This is a research/educational project and should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical concerns.
- **Dataset**: The dataset is not included in this repository due to size. Users must provide their own dataset following the structure outlined above.
- **Model Files**: Trained model files (`.h5`) are excluded from the repository. Users must train the model or provide their own trained model.
- **Computational Requirements**: Training requires significant computational resources. GPU is recommended but not required.

---

## 📄 License

[Specify your license here - MIT, Apache 2.0, etc.]

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 📞 Contact & Contributions

For questions, suggestions, or contributions, please open an issue or submit a pull request.

**Note for College Admissions**: This project demonstrates practical application of machine learning to real-world healthcare challenges, showcasing skills in deep learning, software engineering, and problem-solving. The iterative improvement process (from 54% to 79% accuracy) highlights persistence and analytical thinking.

---

*Built with ❤️ for advancing healthcare through AI*
