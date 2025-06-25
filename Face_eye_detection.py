import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


class FaceEyeDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.geometry("800x600")

        # Label to display images
        self.image_label = Label(root)
        self.image_label.pack(pady=20)

        # Upload image button
        self.upload_button = Button(
            root, text="Upload Image", command=self.upload_image, width=20, height=2, bg="lightblue"
        )
        self.upload_button.pack(pady=10)

        # Use webcam button
        self.webcam_button = Button(
            root, text="Use Webcam", command=self.use_webcam, width=20, height=2, bg="lightgreen"
        )
        self.webcam_button.pack(pady=10)

        # Quit button
        self.quit_button = Button(root, text="Quit", command=root.quit, width=20, height=2, bg="coral")
        self.quit_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            image = self.resize_image(image, max_width=800, max_height=600)  # Resize for consistency
            self.detect(image, face_cascade, eye_cascade, display=True)

    def use_webcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.detect(frame, face_cascade, eye_cascade, display=False)
            cv2.imshow("Webcam - Press Q to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def draw_boundary(self, img, cascade, scaleFactor, minNeighbors, color, label):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

        coords = []
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
            )
            coords = [x, y, w, h]

        return coords

    def detect(self, img, faceCascade, eyeCascade, display):
        colors = {"blue": (255, 0, 0), "green": (0, 255, 0)}
        coords = self.draw_boundary(img, faceCascade, 1.1, 10, colors["blue"], "Face")

        if len(coords) == 4:
            x, y, w, h = coords
            roi_img = img[y: y + h, x: x + w]
            self.draw_boundary(roi_img, eyeCascade, 1.1, 14, colors["green"], "Eyes")

        if display:
            # Convert image to RGB format for Tkinter
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image

    def resize_image(self, img, max_width, max_height):
        """Resize the image to fit within max_width and max_height while maintaining aspect ratio."""
        height, width = img.shape[:2]
        aspect_ratio = width / height

        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)

            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return img


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetection(root)
    root.mainloop()
