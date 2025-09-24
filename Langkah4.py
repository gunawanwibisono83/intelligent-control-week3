import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load model yang telah dilatih ===
model = load_model("cnn_model.h5")

# === Load label kelas dari file ===
class_labels = np.load("class_labels.npy")

# === Akses kamera ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mode Night Vision dengan konversi ke skala abu-abu
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi kelas
    pred = model.predict(img)
    class_index = np.argmax(pred)
    label = class_labels[class_index]
    confidence = pred[0][class_index] * 100  # persen

    # Tampilkan hasil di frame utama (warna merah)
    cv2.putText(frame, f"Class: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # merah
    cv2.putText(frame, f"Accuracy: {confidence:.2f}%", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # merah

    # Tampilkan hasil juga di night vision (warna merah)
    cv2.putText(night_vision, f"Class: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(night_vision, f"Accuracy: {confidence:.2f}%", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Tampilkan jendela
    cv2.imshow("Frame", frame)
    cv2.imshow("Night Vision", night_vision)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
