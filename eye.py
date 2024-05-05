import cv2
import mediapipe
import pyautogui

# Initialize webcam and Face Mesh
cam = cv2.VideoCapture(0)
face_mesh_landmark = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen dimensions
screen_w, screen_h = pyautogui.size()

while True:
    # Capture frame-by-frame from the webcam
    success, image = cam.read()

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # If capture failed, exit the loop
    if not success:
        print("Failed to capture image")
        break

    # Get the dimensions (height, width, channels) of the frame
    window_h, window_w, _ = image.shape

    # Convert BGR to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect face landmarks
    processed_image = face_mesh_landmark.process(rgb_image)

    # Check if any face landmarks were detected
    if processed_image.multi_face_landmarks:
        # Get the first set of face landmarks
        one_face = processed_image.multi_face_landmarks[0]

        # Draw landmarks on the image and control the mouse
        for id, landmark_point in enumerate(one_face.landmark[474:478]):
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)

            # Draw a circle at each landmark
            cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=-1)

            if id == 1:
                # Calculate mouse coordinates relative to the screen
                mouse_x = int(landmark_point.x * screen_w)
                mouse_y = int(landmark_point.y * screen_h)
                # Move the mouse to the calculated coordinates
                pyautogui.moveTo(mouse_x, mouse_y)

        # Left eye landmarks
        left_eye = [one_face.landmark[145], one_face.landmark[159]]

        # Draw circles for left eye landmarks
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=-1)

        # Check for eye blink to simulate a mouse click
        # If the y-difference between landmarks is small, consider it a blink
        if abs(left_eye[0].y - left_eye[1].y) < 0.01:
            pyautogui.click()  # Simulate a mouse click
            pyautogui.sleep(1)  # Small delay to prevent multiple clicks
            print("Click successful")

    # Display the resulting frame
    cv2.imshow("Eye Control", image)

    # Exit on 'Esc' key
    key = cv2.waitKey(1)  # Wait briefly for user input
    if key == 27:  # ASCII code for 'Esc'
        break

# Release webcam and close OpenCV windows
cam.release()
cv2.destroyAllWindows()