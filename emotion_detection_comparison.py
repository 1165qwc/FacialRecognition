import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import emotion detection libraries
try:
    from feat import Detector
    PYFEAT_AVAILABLE = True
except ImportError:
    PYFEAT_AVAILABLE = False
    print("Py-Feat not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("DeepFace not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available")

# Ekman's 7 basic emotions
EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'contempt']

class EmotionDetectionComparison:
    def __init__(self, data_path="RecruitView_Data"):
        self.data_path = data_path
        self.frames_dir = os.path.join(data_path, "extracted_frames")
        self.metadata_file = os.path.join(data_path, "analyzed_data_with_categories.csv")

        # Create frames directory
        os.makedirs(self.frames_dir, exist_ok=True)

        # Check library availability
        self.pyfeat_available = self._check_pyfeat()
        self.deepface_available = self._check_deepface()
        self.mediapipe_available = self._check_mediapipe()

        # Initialize detectors
        self.detectors = {}
        if self.pyfeat_available:
            try:
                self.detectors['pyfeat'] = Detector()
            except:
                self.pyfeat_available = False

        if self.mediapipe_available:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils

    def _check_pyfeat(self):
        try:
            from feat import Detector
            return True
        except ImportError:
            return False

    def _check_deepface(self):
        try:
            from deepface import DeepFace
            return True
        except ImportError:
            return False

    def _check_mediapipe(self):
        try:
            import mediapipe as mp
            return True
        except ImportError:
            return False

    def load_metadata(self):
        """Load the dataset metadata"""
        print("Loading metadata...")
        self.df = pd.read_csv(self.metadata_file)
        print(f"Loaded {len(self.df)} samples")
        print(f"Expressiveness categories: {self.df['expressiveness_category'].value_counts()}")
        return self.df

    def extract_frames_from_video(self, video_path, video_id, frame_interval=10):
        """Extract frames from video at specified intervals"""
        frames = []
        frame_count = 0

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return frames

            while True:
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

    def extract_frames_batch(self, num_videos=50, frame_interval=10):
        """Extract frames from a batch of videos"""
        print(f"Extracting frames from {num_videos} videos...")

        # Get video files
        videos_dir = os.path.join(self.data_path, "videos")
        if not os.path.exists(videos_dir):
            print(f"Videos directory not found: {videos_dir}")
            return []

        video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')][:num_videos]

        all_frames = []
        frame_labels = []

        for video_file in tqdm(video_files, desc="Extracting frames"):
            video_path = os.path.join(videos_dir, video_file)
            video_id = video_file.replace('vid_', '').replace('.mp4', '')

            # Get metadata for this video
            video_data = self.df[self.df['id'] == video_id.zfill(4)]
            if len(video_data) == 0:
                continue

            expressiveness = video_data['expressiveness_category'].iloc[0]
            facial_expression_score = video_data['facial_expression'].iloc[0]

            # Extract frames
            frames = self.extract_frames_from_video(video_path, video_id, frame_interval)

            for frame in frames:
                all_frames.append(frame)
                frame_labels.append({
                    'video_id': video_id,
                    'expressiveness': expressiveness,
                    'facial_expression_score': facial_expression_score
                })

        print(f"Extracted {len(all_frames)} frames from {len(video_files)} videos")
        return all_frames, frame_labels

    def detect_emotions_pyfeat(self, frame):
        """Detect emotions using Py-Feat"""
        if not self.pyfeat_available:
            return None

        try:
            # Detect faces and emotions
            detections = self.detectors['pyfeat'].detect_faces(frame)
            if len(detections) == 0:
                return None

            emotions = self.detectors['pyfeat'].detect_emotions(frame, detections[0])
            if len(emotions) == 0:
                return None

            # Map to Ekman's emotions (subset of available emotions)
            emotion_scores = emotions[0]
            ekman_mapping = {
                'anger': emotion_scores.get('anger', 0),
                'disgust': emotion_scores.get('disgust', 0),
                'fear': emotion_scores.get('fear', 0),
                'happiness': emotion_scores.get('joy', 0),  # joy maps to happiness
                'sadness': emotion_scores.get('sadness', 0),
                'surprise': emotion_scores.get('surprise', 0),
                'contempt': emotion_scores.get('disgust', 0) * 0.5  # approximation
            }

            return ekman_mapping

        except Exception as e:
            print(f"Py-Feat detection error: {e}")
            return None

    def detect_emotions_deepface(self, frame):
        """Detect emotions using DeepFace"""
        if not self.deepface_available:
            return None

        try:
            # Analyze emotions
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']

            # Map to Ekman's emotions
            ekman_mapping = {
                'anger': emotions.get('angry', 0),
                'disgust': emotions.get('disgust', 0),
                'fear': emotions.get('fear', 0),
                'happiness': emotions.get('happy', 0),
                'sadness': emotions.get('sad', 0),
                'surprise': emotions.get('surprise', 0),
                'contempt': 0  # DeepFace doesn't have contempt
            }

            return ekman_mapping

        except Exception as e:
            print(f"DeepFace detection error: {e}")
            return None

    def detect_emotions_mediapipe(self, frame):
        """Detect emotions using MediaPipe (basic face landmarks, not full emotion detection)"""
        if not self.mediapipe_available:
            return None

        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:

                # Convert to RGB if needed
                if frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame

                results = face_mesh.process(rgb_frame)

                if not results.multi_face_landmarks:
                    return None

                # MediaPipe doesn't have built-in emotion detection
                # We'll use a simple heuristic based on facial landmarks
                landmarks = results.multi_face_landmarks[0]

                # Extract key points for basic emotion indicators
                # This is a simplified approach - real emotion detection would need a trained model
                left_eye = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y)
                                   for i in [33, 160, 158, 133, 153, 144]], axis=0)
                right_eye = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y)
                                    for i in [362, 385, 387, 263, 373, 380]], axis=0)
                mouth = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y)
                                for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]], axis=0)
                nose = (landmarks.landmark[1].x, landmarks.landmark[1].y)

                # Simple heuristics (these are not accurate emotion detection)
                eye_distance = np.linalg.norm(left_eye - right_eye)
                mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
                mouth_height = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)

                # Return neutral scores since MediaPipe doesn't do emotion detection
                return {emotion: 0.1 for emotion in EKMAN_EMOTIONS}

        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return None

    def compare_libraries(self, frames, labels, max_frames=100):
        """Compare emotion detection across libraries"""
        print("Comparing emotion detection libraries...")

        results = {
            'pyfeat': [],
            'deepface': [],
            'mediapipe': []
        }

        frame_labels = []

        for i, (frame, label) in enumerate(tqdm(zip(frames[:max_frames], labels[:max_frames]),
                                              desc="Analyzing frames")):

            frame_result = {
                'frame_id': i,
                'expressiveness': label['expressiveness'],
                'facial_expression_score': label['facial_expression_score']
            }

            # Py-Feat detection
            if PYFEAT_AVAILABLE:
                pyfeat_emotions = self.detect_emotions_pyfeat(frame)
                if pyfeat_emotions:
                    frame_result['pyfeat'] = pyfeat_emotions
                    results['pyfeat'].append(frame_result.copy())

            # DeepFace detection
            if DEEPFACE_AVAILABLE:
                deepface_emotions = self.detect_emotions_deepface(frame)
                if deepface_emotions:
                    frame_result['deepface'] = deepface_emotions
                    results['deepface'].append(frame_result.copy())

            # MediaPipe detection (limited)
            if MEDIAPIPE_AVAILABLE:
                mediapipe_emotions = self.detect_emotions_mediapipe(frame)
                if mediapipe_emotions:
                    frame_result['mediapipe'] = mediapipe_emotions
                    results['mediapipe'].append(frame_result.copy())

        return results

    def analyze_results(self, results):
        """Analyze and visualize comparison results"""
        print("\n=== Emotion Detection Comparison Results ===")

        # Count successful detections
        for library in results:
            success_count = len(results[library])
            print(f"{library}: {success_count} successful detections")

        # Analyze emotion distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Emotion Detection Distributions by Library')

        for i, library in enumerate(['pyfeat', 'deepface', 'mediapipe']):
            if library in results and results[library]:
                emotions_data = []
                for result in results[library]:
                    if library in result:
                        emotions = result[library]
                        emotions_data.append(list(emotions.values()))

                if emotions_data:
                    emotions_df = pd.DataFrame(emotions_data, columns=EKMAN_EMOTIONS)
                    emotions_df.boxplot(ax=axes[i])
                    axes[i].set_title(f'{library.upper()} Emotion Scores')
                    axes[i].set_ylabel('Emotion Intensity')
                    axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('emotion_detection_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Analyze correlation with facial expression scores
        correlations = {}
        for library in results:
            if results[library]:
                facial_scores = [r['facial_expression_score'] for r in results[library]]
                emotion_scores = []

                for r in results[library]:
                    if library in r:
                        # Use happiness as proxy for positive expression
                        happiness_score = r[library].get('happiness', 0)
                        emotion_scores.append(happiness_score)

                if emotion_scores:
                    corr = np.corrcoef(facial_scores, emotion_scores)[0, 1]
                    correlations[library] = corr
                    print(".3f")

        return correlations

def main():
    # Initialize comparison system
    comparator = EmotionDetectionComparison()

    # Load metadata
    df = comparator.load_metadata()

    # Extract frames from videos
    frames, labels = comparator.extract_frames_batch(num_videos=20, frame_interval=10)

    if not frames:
        print("No frames extracted. Please check video files.")
        return

    # Compare emotion detection libraries
    results = comparator.compare_libraries(frames, labels, max_frames=50)

    # Analyze results
    correlations = comparator.analyze_results(results)

    print("\n=== Summary ===")
    print(f"Available libraries:")
    print(f"  Py-Feat: {'✓' if self.pyfeat_available else '✗'}")
    print(f"  DeepFace: {'✓' if self.deepface_available else '✗'}")
    print(f"  MediaPipe: {'✓' if self.mediapipe_available else '✗'}")

    print("\nEmotion detection comparison completed!")
    print("Results saved to: emotion_detection_comparison.png")

if __name__ == "__main__":
    main()