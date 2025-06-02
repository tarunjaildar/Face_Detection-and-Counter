import cv2
import imutils
import numpy as np
import threading
import pyodbc
from datetime import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def connect_db():
    try:
        conn = pyodbc.connect(
            "DRIVER={SQL Server};"
            "SERVER=NITRO\\SQLEXPRESS01;"  
            "DATABASE=FaceDetectionDB;"
            "Trusted_Connection=yes;"
        )
        return conn
    except Exception as err:
        print(f"Database connection failed: {err}")
        return None


def create_table():
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='face_detection_log' AND xtype='U')
            CREATE TABLE face_detection_log (
                id INT IDENTITY(1,1) PRIMARY KEY,
                source NVARCHAR(255),
                face_count INT,
                detection_time DATETIME DEFAULT GETDATE()
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()

def insert_face_data(face_count, source="webcam"):
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        query = "INSERT INTO face_detection_log (source, face_count) VALUES (?, ?)"
        cursor.execute(query, (source, face_count))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Data inserted: {face_count} faces detected in {source}")


def detect_faces(frame):
    if frame.shape[1] > 800:
        frame = imutils.resize(frame, width=800)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.07,
        minNeighbors=5,
        minSize=(3, 3),
        maxSize=(300, 300)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    total_faces = len(faces)
    cv2.putText(frame, 'Status: Detecting', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Faces: {total_faces}', (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    return frame, total_faces

def detect_video(video_source=0, output_path=None):
    video = cv2.VideoCapture(video_source)
    
    if not video.isOpened():
        print('Video source not found. Please check the path.')
        return

    print('Detecting faces...')
    writer = None
    if output_path:
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))
    
    while True:
        ret, frame = video.read()
        if not ret:
            print('End of video or failed to capture.')
            break

        processed_frame, face_count = detect_faces(frame)
        
        if writer:
            writer.write(processed_frame)

        cv2.imshow('Face Detection Output', processed_frame)
        
        insert_face_data(face_count, video_source if isinstance(video_source, str) else "webcam")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def detect_image(image_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image file at {image_path}.")
        return

    processed_image, face_count = detect_faces(image)

    cv2.imshow('Detected Image', processed_image)

    if output_path:
        cv2.imwrite(output_path, processed_image)
        print(f"Image saved at {output_path}")

    insert_face_data(face_count, image_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def start_webcam_detection():
    detection_thread = threading.Thread(target=detect_video, args=(0,))
    detection_thread.start()


def face_detector(inp_type, inp_path='', output_path=''):
    create_table()  

    if inp_type == 'camera':
        print('Starting webcam detection...')
        start_webcam_detection()
    elif inp_type == 'image':
        print('Processing image...')
        detect_image(inp_path, output_path)
    elif inp_type == 'video':
        print('Processing video...')
        detect_video(inp_path, output_path)

# Example usage
# face_detector('image', r"C:\Users\tarun\Desktop\python\Screenshot 2024-09-10 175921.png")
# face_detector('image', r"C:\Users\tarun\Desktop\python\Screenshot 2024-09-10 175938.png")
# face_detector('image', r"C:\Users\tarun\Desktop\python\Screenshot 2024-09-10 181600.png")
# face_detector('image', r"C:\Users\tarun\Desktop\python\Screenshot 2024-09-10 181659.png")
# face_detector('image', r"C:\Users\tarun\Desktop\python\Screenshot 2024-11-27 152056.png")
# face_detector('image', r"C:\Users\tarun\Desktop\python\WhatsApp Image 2025-05-22 at 10.01.29_1782dd98.jpg") 

# face_detector('video', r"C:\Users\tarun\Desktop\python\face-demographics-walking.mp4")

face_detector('camera')