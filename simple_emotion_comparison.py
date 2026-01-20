import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ekman's 7 basic emotions
EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'contempt']

class SimpleEmotionComparison:
    def __init__(self, data_path="RecruitView_Data"):
        self.data_path = data_path
        self.frames_dir = os.path.join(data_path, "extracted_frames")
        self.metadata_file = os.path.join(data_path, "analyzed_data_with_categories.csv")

        # Create frames directory
        os.makedirs(self.frames_dir, exist_ok=True)

    def load_metadata(self):
        """Load the dataset metadata"""
        print("Loading metadata...")
        self.df = pd.read_csv(self.metadata_file)
        print(f"Loaded {len(self.df)} samples")
        print(f"Expressiveness categories: {self.df['expressiveness_category'].value_counts()}")
        return self.df

    def extract_frames_from_video(self, video_path, video_id, frame_interval=10, max_frames=5):
        """Extract frames from video at specified intervals"""
        frames = []
        frame_count = 0

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return frames

            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                frame_count += 1

            cap.release()

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")

        return frames

    def extract_frames_batch(self, num_videos=10, frame_interval=10):
        """Extract frames from a batch of videos"""
        print(f"Extracting frames from {num_videos} videos...")

        # Get video files
        videos_dir = os.path.join(self.data_path, "videos")
        if not os.path.exists(videos_dir):
            print(f"Videos directory not found: {videos_dir}")
            return [], []

        video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')][:num_videos]

        all_frames = []
        frame_labels = []

        for video_file in tqdm(video_files, desc="Extracting frames"):
            video_path = os.path.join(videos_dir, video_file)
            video_id = video_file.replace('vid_', '').replace('.mp4', '')

            # Debug: print video path
            print(f"\nProcessing video: {video_path}")
            print(f"Video exists: {os.path.exists(video_path)}")

            # Get metadata for this video
            video_data = self.df[self.df['id'] == int(video_id)]
            if len(video_data) == 0:
                print(f"No metadata found for video ID: {video_id}")
                continue

            expressiveness = video_data['expressiveness_category'].iloc[0]
            facial_expression_score = video_data['facial_expression'].iloc[0]

            # Extract frames
            frames = self.extract_frames_from_video(video_path, video_id, frame_interval)
            print(f"Extracted {len(frames)} frames from {video_file}")

            for i, frame in enumerate(frames):
                all_frames.append(frame)
                frame_labels.append({
                    'video_id': video_id,
                    'frame_idx': i,
                    'expressiveness': expressiveness,
                    'facial_expression_score': facial_expression_score
                })

        print(f"Extracted {len(all_frames)} frames from {len(video_files)} videos")
        return all_frames, frame_labels

    def analyze_expressiveness_patterns(self, frames, labels):
        """Analyze expressiveness patterns in the dataset"""
        print("\nAnalyzing expressiveness patterns...")

        # Group by expressiveness category
        expressiveness_stats = {}

        for label in labels:
            category = label['expressiveness']
            if category not in expressiveness_stats:
                expressiveness_stats[category] = []

            expressiveness_stats[category].append(label['facial_expression_score'])

        # Print statistics
        print("\nExpressiveness Category Statistics:")
        for category, scores in expressiveness_stats.items():
            print(f"{category}:")
            print(f"  Count: {len(scores)}")
            print(".3f")
            print(".3f")
            print(".3f")
            print()

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Box plot of facial expression scores by category
        categories = list(expressiveness_stats.keys())
        scores_data = [expressiveness_stats[cat] for cat in categories]

        ax1.boxplot(scores_data, labels=categories)
        ax1.set_title('Facial Expression Scores by Expressiveness Category')
        ax1.set_ylabel('Facial Expression Score')
        ax1.grid(True, alpha=0.3)

        # Distribution plot
        all_scores = []
        all_categories = []
        for cat, scores in expressiveness_stats.items():
            all_scores.extend(scores)
            all_categories.extend([cat] * len(scores))

        sns.histplot(data=pd.DataFrame({'score': all_scores, 'category': all_categories}),
                    x='score', hue='category', ax=ax2, alpha=0.7)
        ax2.set_title('Distribution of Facial Expression Scores')
        ax2.set_xlabel('Facial Expression Score')

        plt.tight_layout()
        plt.savefig('expressiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return expressiveness_stats

    def create_emotion_detection_pipeline(self):
        """Create a basic emotion detection pipeline using OpenCV and simple heuristics"""
        print("\nCreating basic emotion detection pipeline...")

        # This would be where we implement a custom emotion detection model
        # For now, we'll create a framework that can be extended

        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def detect_faces_opencv(frame):
            """Detect faces using OpenCV Haar cascades"""
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces

        def extract_basic_features(frame, face_coords):
            """Extract basic facial features for emotion analysis"""
            x, y, w, h = face_coords

            # Extract face region
            face_roi = frame[y:y+h, x:x+w]

            if face_roi.size == 0:
                return None

            # Convert to grayscale for basic analysis
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

            # Simple feature extraction (this is very basic)
            # In a real system, you'd use landmark detection and proper feature extraction
            features = {
                'face_area': w * h,
                'face_aspect_ratio': w / h,
                'mean_intensity': np.mean(gray_face),
                'std_intensity': np.std(gray_face)
            }

            return features

        return detect_faces_opencv, extract_basic_features

    def demonstrate_pipeline(self, frames, labels):
        """Demonstrate the emotion detection pipeline"""
        print("\nDemonstrating emotion detection pipeline...")

        detect_faces, extract_features = self.create_emotion_detection_pipeline()

        successful_detections = 0
        features_list = []

        for i, (frame, label) in enumerate(zip(frames[:20], labels[:20])):  # Test on first 20 frames
            faces = detect_faces(frame)

            if len(faces) > 0:
                # Use the first detected face
                features = extract_features(frame, faces[0])
                if features:
                    features['expressiveness'] = label['expressiveness']
                    features['facial_expression_score'] = label['facial_expression_score']
                    features_list.append(features)
                    successful_detections += 1

        print(f"Successfully analyzed {successful_detections} faces out of {len(frames[:20])} frames")

        if features_list:
            # Convert to DataFrame for analysis
            features_df = pd.DataFrame(features_list)

            # Basic correlation analysis
            numeric_cols = ['face_area', 'face_aspect_ratio', 'mean_intensity', 'std_intensity', 'facial_expression_score']
            correlations = features_df[numeric_cols].corr()

            print("\nFeature Correlations with Facial Expression Score:")
            for col in ['face_area', 'face_aspect_ratio', 'mean_intensity', 'std_intensity']:
                corr = correlations.loc[col, 'facial_expression_score']
                print(".3f")

            # Visualize
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Basic Facial Features vs Facial Expression Score')

            features_to_plot = ['face_area', 'face_aspect_ratio', 'mean_intensity', 'std_intensity']

            for i, feature in enumerate(features_to_plot):
                ax = axes[i//2, i%2]
                ax.scatter(features_df[feature], features_df['facial_expression_score'], alpha=0.6)
                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel('Facial Expression Score')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('basic_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

        return features_list

def main():
    # Initialize comparison system
    comparator = SimpleEmotionComparison()

    # Load metadata
    df = comparator.load_metadata()

    # Extract frames from videos
    frames, labels = comparator.extract_frames_batch(num_videos=20, frame_interval=10)

    if not frames:
        print("No frames extracted. Please check video files.")
        return

    # Analyze expressiveness patterns
    expressiveness_stats = comparator.analyze_expressiveness_patterns(frames, labels)

    # Demonstrate basic emotion detection pipeline
    features = comparator.demonstrate_pipeline(frames, labels)

    print("\n=== Summary ===")
    print(f"Total frames analyzed: {len(frames)}")
    print(f"Expressiveness categories found: {list(expressiveness_stats.keys())}")
    print(f"Basic feature extraction completed: {'Yes' if features else 'No'}")
    print("\nResults saved to:")
    print("- expressiveness_analysis.png")
    print("- basic_features_analysis.png")

if __name__ == "__main__":
    main()