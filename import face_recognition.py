import face_recognition
import cv2
# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("known_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Create array of known face encodings and names
known_face_encodings = [known_encoding]
known_face_names = ["Your Name"]

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert image from BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find faces and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through faces in this frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if face matches known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
video_capture.release()
cv2.destroyAllWindows()
