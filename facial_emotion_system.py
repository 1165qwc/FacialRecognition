import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ekman's 7 basic emotions
EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'contempt']

class FacialEmotionSystem:
    def __init__(self, data_path="RecruitView_Data", input_shape=(64, 64, 3)):
        self.data_path = data_path
        self.input_shape = input_shape
        self.metadata_file = os.path.join(data_path, "analyzed_data_with_categories.csv")
        self.model_path = "facial_emotion_model.keras"

        # Load metadata
        self.df = pd.read_csv(self.metadata_file)
        print(f"Loaded {len(self.df)} samples from dataset")

        # Initialize label encoder for expressiveness categories
        self.label_encoder = LabelEncoder()
        self.df['expressiveness_encoded'] = self.label_encoder.fit_transform(self.df['expressiveness_category'])

    def create_emotion_model(self):
        """Create a CNN model for emotion recognition"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=self.input_shape),

            # Convolutional layers
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            # Dense layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),

            # Output layer - predict expressiveness category
            keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        return model

    def preprocess_frame(self, frame):
        """Preprocess a frame for model input"""
        # Resize to input shape
        resized = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))

        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def extract_faces_from_frame(self, frame):
        """Extract faces from a frame using Haar cascades"""
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        face_images = []
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = int(0.1 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                face_images.append(self.preprocess_frame(face_img))

        return face_images

    def create_training_dataset(self, num_videos=100, frames_per_video=5, frame_interval=10):
        """Create training dataset from videos"""
        print(f"Creating training dataset from {num_videos} videos...")

        videos_dir = os.path.join(self.data_path, "videos")
        video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')][:num_videos]

        X = []  # Face images
        y = []  # Expressiveness labels

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(videos_dir, video_file)
            video_id = video_file.replace('vid_', '').replace('.mp4', '')

            # Get metadata
            video_data = self.df[self.df['id'] == int(video_id)]
            if len(video_data) == 0:
                continue

            expressiveness = video_data['expressiveness_encoded'].iloc[0]
            expressiveness_label = video_data['expressiveness_category'].iloc[0]

            # Extract frames from video
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            frames_extracted = 0

            while frames_extracted < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Extract faces
                    faces = self.extract_faces_from_frame(frame_rgb)

                    # Add faces to dataset
                    for face in faces:
                        X.append(face)
                        y.append(expressiveness)
                        frames_extracted += 1

                        if frames_extracted >= frames_per_video:
                            break

                frame_count += 1

            cap.release()

        X = np.array(X)
        y = np.array(y)

        print(f"Created dataset with {len(X)} face images")
        print(f"Class distribution: {pd.Series(y).value_counts()}")

        return X, y

    def train_model(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the emotion recognition model"""
        print("Training emotion recognition model...")

        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, num_classes=len(self.label_encoder.classes_))

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y
        )

        # Create model
        model = self.create_emotion_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                print(f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")
            )
        ]

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0  # Reduce verbosity to avoid encoding issues
        )

        # Save final model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

        return model, history

    def evaluate_model(self, model, X, y):
        """Evaluate the trained model"""
        print("Evaluating model...")

        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, num_classes=len(self.label_encoder.classes_))

        # Predictions
        y_pred_probs = model.predict(X)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_categorical, axis=1)

        # Classification report
        target_names = self.label_encoder.classes_
        try:
            report = classification_report(y_true, y_pred, target_names=target_names)
            print("\nClassification Report:")
            print(report)
        except UnicodeEncodeError:
            # Fallback for encoding issues
            print("\nClassification Report (simplified):")
            accuracy = (y_pred == y_true).mean()
            print(f"Accuracy: {accuracy:.4f}")
            for i, class_name in enumerate(target_names):
                mask = (y_true == i)
                if mask.sum() > 0:
                    class_acc = (y_pred[mask] == i).mean()
                    print(f"{class_name}: {class_acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return report, cm

    def predict_emotion(self, frame):
        """Predict emotion from a single frame"""
        if not hasattr(self, 'model'):
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
            else:
                raise ValueError("No trained model found. Please train the model first.")

        # Extract faces
        faces = self.extract_faces_from_frame(frame)

        if not faces:
            return None, "No face detected"

        # Get prediction for first face
        face = faces[0]
        face_batch = np.expand_dims(face, axis=0)

        predictions = self.model.predict(face_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = predictions[predicted_class_idx]

        return predicted_class, confidence

    def create_emotion_ekman_mapping(self):
        """Create mapping from expressiveness categories to Ekman's emotions"""
        # This is a simplified mapping - in practice, you'd need more sophisticated analysis
        emotion_mapping = {
            'Reserved Expression': ['sadness', 'contempt'],  # Low expressiveness
            'Balanced Expression': ['surprise', 'happiness'],  # Moderate expressiveness
            'Expressive': ['anger', 'fear', 'disgust', 'happiness']  # High expressiveness
        }

        return emotion_mapping

    def real_time_emotion_detection(self, video_path=None):
        """Demonstrate real-time emotion detection"""
        print("Starting real-time emotion detection...")

        # Load model if available
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            print("No trained model found. Using face detection only.")
            self.model = None

        # Open video capture
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)  # Webcam

        if not cap.isOpened():
            print("Could not open video source")
            return

        emotion_mapping = self.create_emotion_ekman_mapping()

        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            # Process each face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                if self.model:
                    # Predict expressiveness
                    expressiveness, confidence = self.predict_emotion(frame_rgb)

                    # Map to Ekman's emotions
                    possible_emotions = emotion_mapping.get(expressiveness, EKMAN_EMOTIONS)

                    # Display results
                    label = f"{expressiveness} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Show possible Ekman emotions
                    emotions_text = f"Emotions: {', '.join(possible_emotions[:3])}"
                    cv2.putText(frame, emotions_text, (x, y+h+30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Facial Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize emotion system
    emotion_system = FacialEmotionSystem()

    # Create training dataset
    print("=== Creating Training Dataset ===")
    X, y = emotion_system.create_training_dataset(num_videos=50, frames_per_video=3)

    if len(X) == 0:
        print("No training data created. Exiting.")
        return

    # Train model
    print("\n=== Training Model ===")
    model, history = emotion_system.train_model(X, y, epochs=30)

    # Evaluate model
    print("\n=== Evaluating Model ===")
    report, cm = emotion_system.evaluate_model(model, X, y)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== Summary ===")
    print(f"Training completed with {len(X)} samples")
    print(f"Model saved to: {emotion_system.model_path}")
    print("Results saved to: training_history.png, confusion_matrix.png")

    # Optional: Run real-time detection (commented out for automated testing)
    # print("\n=== Real-time Emotion Detection ===")
    # emotion_system.real_time_emotion_detection()

if __name__ == "__main__":
    main()