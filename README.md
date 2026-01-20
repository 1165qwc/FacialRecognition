# Facial Emotion Detection System

A comprehensive facial emotion detection system using Ekman's 7 basic emotions framework, trained on the RecruitView dataset for HR applications and affective computing research.

## Overview

This project implements a complete pipeline for facial emotion recognition using computer vision and deep learning techniques. The system is designed to analyze video interviews and detect emotional expressions based on Ekman's psychological framework of 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and contempt.

## Features

- Real-time video emotion analysis
- Batch processing of interview videos
- Ekman's 7 emotions classification
- Expressiveness level detection (Reserved, Balanced, Expressive)
- Face tracking and cropping
- Training data expansion capabilities
- Multiple output formats (JSON, visualizations)

## Dataset

This project uses the **RecruitView dataset**, which contains:
- 2,011 "in-the-wild" video responses from over 300 participants
- Expert-annotated personality traits (Big Five) and interview performance metrics
- Facial expression scores derived from pairwise comparisons by clinical psychologists
- Ground truth labels for training and evaluation

### Data Source
- **Name**: RecruitView: Multimodal Dataset for Personality & Interview Performance
- **License**: CC BY-NC 4.0
- **URL**: https://huggingface.co/datasets/AI4A-lab/RecruitView

## Technical Stack

- **Deep Learning**: TensorFlow 2.16.1, Keras 3.0.5
- **Computer Vision**: OpenCV 4.9.0.80
- **Data Processing**: NumPy 1.26.4, Pandas 2.2.2
- **Visualization**: Matplotlib 3.8.4, Seaborn 0.13.2
- **Machine Learning**: Scikit-learn 1.4.2

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd facial-emotion-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the RecruitView dataset from HuggingFace and place it in the `RecruitView_Data/` directory.

## Usage

### Basic Usage

1. Run the emotion detection comparison:
```bash
python simple_emotion_comparison.py
```

2. Train the emotion recognition model:
```bash
python facial_emotion_system.py
```

3. View the final results and analysis:
```bash
python clean_final_summary.py
```

### Advanced Usage

#### Real-time Emotion Detection
The system includes capabilities for real-time video analysis (implementation can be extended).

#### Batch Processing
Process multiple videos for offline analysis:
```python
from facial_emotion_system import FacialEmotionSystem

# Initialize system
emotion_system = FacialEmotionSystem()

# Process videos
X, y = emotion_system.create_training_dataset(num_videos=100)
```

## Architecture

### CNN Model Architecture
- **Input**: 64x64 RGB face images
- **Layers**: 4 convolutional blocks with batch normalization
- **Dropout**: 0.25 after conv blocks, 0.5 after dense layers
- **Output**: 3 expressiveness categories (Reserved, Balanced, Expressive)
- **Activation**: ReLU for hidden layers, Softmax for output

### Emotion Mapping
- **Reserved Expression** → sadness, contempt
- **Balanced Expression** → surprise, happiness
- **Expressive** → anger, fear, disgust, happiness

## Results

### Model Performance
- **Validation Accuracy**: ~62%
- **Training Epochs**: 11 (with early stopping)
- **Dataset Size**: 143 face images from 50 videos

### Library Comparison
| Library | Status | Emotions | Speed |
|---------|--------|----------|-------|
| Py-Feat | Complex dependencies | Ekman + more | Slow |
| DeepFace | Installation issues | Basic 7 emotions | Medium |
| MediaPipe | Face tracking only | None built-in | Fast |
| Custom CNN | Fully functional | Expressiveness → Ekman | Fast inference |

## Project Structure

```
facial-emotion-detection/
├── RecruitView_Data/              # Dataset directory
│   ├── videos/                   # Video files (2011 MP4 files)
│   ├── analyzed_data_with_categories.csv  # Metadata with labels
│   ├── category_thresholds.npy   # Classification thresholds
│   └── README.md                 # Dataset documentation
├── facial_emotion_model.keras    # Trained CNN model
├── basic_features_analysis.png   # Feature correlation plots
├── expressiveness_analysis.png   # Dataset analysis visualization
├── requirements.txt              # Python dependencies
├── facial_emotion_system.py      # Main CNN training system
├── simple_emotion_comparison.py  # Basic analysis and comparison
├── emotion_detection_comparison.py  # Advanced library comparison
├── clean_final_summary.py        # Results summary
└── README.md                     # This file
```

## Methodology

### Data Processing
1. **Frame Extraction**: Every 10th frame from videos for training
2. **Face Detection**: OpenCV Haar cascades for face localization
3. **Preprocessing**: Resize to 64x64, normalization, face cropping
4. **Augmentation**: Ready for data augmentation techniques

### Training Process
1. **Stratified Split**: Maintain class distribution in train/val sets
2. **Early Stopping**: Monitor validation accuracy with patience=10
3. **Learning Rate**: Adaptive scheduling with ReduceLROnPlateau
4. **Regularization**: Dropout and batch normalization

### Evaluation
- **Metrics**: Accuracy, AUC, confusion matrix
- **Cross-validation**: Stratified k-fold validation
- **Visualization**: Training curves, feature correlations

## Ethical Considerations

This project adheres to responsible AI practices:
- **Academic Research Only**: Not for commercial hiring decisions
- **No Automated Hiring**: Models should not replace human judgment
- **Privacy Protection**: No personal identification or misuse
- **Bias Awareness**: Dataset primarily contains university students

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{facial_emotion_detection_2024,
  title={Facial Emotion Detection System using Ekman's Framework},
  author={Your Name},
  year={2024},
  note={Based on RecruitView dataset}
}
```

And the original dataset:

```bibtex
@misc{gupta2025recruitview,
  title={RecruitView: A Multimodal Dataset for Predicting Personality and Interview Performance for Human Resources Applications},
  author={Amit Kumar Gupta and Farhan Sheth and Hammad Shaikh and Dheeraj Kumar and Angkul Puniya and Deepak Panwar and Sandeep Chaurasia and Priya Mathur},
  year={2025},
  eprint={2512.00450},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.00450},
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Manipal University Jaipur** for research infrastructure
- **RecruitView Dataset Authors** for providing the research dataset
- **Paul Ekman** for the foundational work on emotion psychology
- **TensorFlow/Keras Team** for the deep learning framework