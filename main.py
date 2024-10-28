# main.py
import cv2
from fer import FER

def set_camera_resolution(cap, width, height):
    """ Set the resolution of the camera. """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def adjust_camera_settings(cap):
    """ Adjust camera settings for optimal quality. """
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness (depends on camera driver support)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)     # Adjust contrast (depends on camera driver support)
    cap.set(cv2.CAP_PROP_FPS, 30)          # Set FPS to 30 if the camera supports it

def main():
    # Load the webcam feed
    cap = cv2.VideoCapture(0)

    # Set camera resolution to 1280x720 (you can change it based on your requirements)
    set_camera_resolution(cap, 1280, 720)
    
    # Adjust camera settings to improve quality
    adjust_camera_settings(cap)

    # Initialize the emotion detector
    emotion_detector = FER()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        # Detect emotions in the frame
        emotion_data = emotion_detector.detect_emotions(frame)
        
        for face in emotion_data:
            bounding_box = face['box']
            emotions = face['emotions']

            # Draw bounding box around the detected face
            x, y, w, h = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display emotions with percentages
            total_score = sum(emotions.values())
            for idx, (emotion, score) in enumerate(emotions.items()):
                percentage = (score / total_score) * 100
                text = f"{emotion}: {percentage:.2f}%"
                cv2.putText(frame, text, (x, y - 10 - idx * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
