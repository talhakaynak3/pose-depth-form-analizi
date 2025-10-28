import cv2
import math
import mediapipe as mp

# >>> VİDEO YOLUNU BURAYA YAZ (aynı klasördeyse adı yeter)
VIDEO_PATH = r"hatalı form 1 ön açı.mp4"
# Örnek tam yol: r"C:\Users\talha\PyCharmMiscProject\annotated_correct.mp4"

# ---- yardımcılar ----
def angle_3pts(a, b, c):
    # a,b,c: (x,y) ve açı B'de (derece)
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    dot = bax*bcx + bay*bcy
    den = (math.hypot(bax, bay) * math.hypot(bcx, bcy)) + 1e-6
    v = max(-1.0, min(1.0, dot/den))
    return math.degrees(math.acos(v))

def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"[HATA] Video açılamadı: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
delay = max(1, int(1000//fps))
drawer = mp.solutions.drawing_utils
style  = mp.solutions.drawing_styles

print("[OK] Açıldı. ESC ile çık.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[BİTTİ] Video sonu.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        # iskeleti çiz
        drawer.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=style.get_default_pose_landmarks_style()
        )

        lm = res.pose_landmarks.landmark
        # sol taraf indeksleri
        L_SH, L_EL, L_WR, L_HIP = 11, 13, 15, 23
        # sağ taraf indeksleri
        R_SH, R_EL, R_WR, R_HIP = 12, 14, 16, 24

        # piksele çevir
        l_sh, l_el, l_wr = to_px(lm[L_SH], w, h), to_px(lm[L_EL], w, h), to_px(lm[L_WR], w, h)
        r_sh, r_el, r_wr = to_px(lm[R_SH], w, h), to_px(lm[R_EL], w, h), to_px(lm[R_WR], w, h)
        l_hip, r_hip     = to_px(lm[L_HIP], w, h), to_px(lm[R_HIP], w, h)

        # açılar
        left_elbow  = angle_3pts(l_sh, l_el, l_wr)
        right_elbow = angle_3pts(r_sh, r_el, r_wr)
        left_shldr  = angle_3pts(l_hip, l_sh, l_el)
        right_shldr = angle_3pts(r_hip, r_sh, r_el)

        # ekrana yaz
        cv2.putText(frame, f"L-ELB: {left_elbow:5.1f}",  (20,40), 0, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"R-ELB: {right_elbow:5.1f}", (20,70), 0, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"L-SHD: {left_shldr:5.1f}",  (20,100),0, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"R-SHD: {right_shldr:5.1f}", (20,130),0, 0.7, (255,255,255), 2)

    cv2.namedWindow("Pose (ESC=exit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose (ESC=exit)", 960, 540)

    cv2.imshow("Pose (ESC=exit)", frame)
    if (cv2.waitKey(delay) & 0xFF) == 27:
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
