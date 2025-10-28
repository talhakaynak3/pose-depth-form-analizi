import cv2
import os
import sys

def main(path):
    # Yol mevcut mu?
    if not os.path.isfile(path):
        print(f"[HATA] Dosya bulunamadı: {path}")
        sys.exit(1)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = max(1, int(1000 // fps))  # ms

    print(f"[OK] Açıldı: {path} | FPS≈{fps:.1f} | ESC ile çık.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[BİTTİ] Video sonu.")
            break
        cv2.imshow("Video", frame)
        if (cv2.waitKey(delay) & 0xFF) == 27:  # ESC
            print("[ÇIKIŞ] ESC basıldı.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1) Aşağıdaki yolu videonun tam yolu veya aynı klasördeyse adını yaz
    VIDEO_PATH = r"doğru form ön açı.mp4"
    # Örnek tam yol:
    # VIDEO_PATH = r"C:\Users\talha\PyCharmMiscProject\annotated_correct.mp4"

    main(VIDEO_PATH)
