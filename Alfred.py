import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    hand_results = hands.process(image)
    face_results = face_detection.process(image)

    # Draw the hand tracking annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if hand_results.multi_hand_landmarks:
      for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw face detection annotations on the image.
    if face_results.detections:
      for detection in face_results.detections:
        mp_drawing.draw_detection(image, detection)
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands & Face', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
