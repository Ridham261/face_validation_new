import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh




def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)




def compute_yaw_angle(left_eye, right_eye, nose):
    left_dist = abs(nose.x - left_eye.x)
    right_dist = abs(right_eye.x - nose.x)
    total_dist = left_dist + right_dist

    if total_dist == 0:
        return 0.0

    ratio = (right_dist - left_dist) / total_dist
    ratio = max(-1.0, min(1.0, ratio))

    return abs(math.degrees(math.asin(ratio)))


def compute_eye_ratio(left_eye, right_eye, nose):
    left_dist = abs(nose.x - left_eye.x)
    right_dist = abs(right_eye.x - nose.x)

    max_dist = max(left_dist, right_dist)
    if max_dist == 0:
        return 1.0

    return min(left_dist, right_dist) / max_dist


def check_side_profile(yaw_angle, eye_ratio):
    if eye_ratio < 0.15:
        return True, "Not acceptable side facing photo"

    if yaw_angle > 45:
        return True, "Not acceptable side facing photo"

    if yaw_angle > 35 and eye_ratio < 0.35:
        return True, "Not acceptable side facing photo"

    return False, ""



def compute_eye_aspect_ratio(landmarks):
    LEFT = [33, 133, 159, 145]
    RIGHT = [263, 362, 386, 374]

    # Left Eye
    left_width = distance(landmarks[LEFT[0]], landmarks[LEFT[1]])
    left_height = distance(landmarks[LEFT[2]], landmarks[LEFT[3]])
    left_EAR = left_height / left_width if left_width != 0 else 0

    # Right Eye
    right_width = distance(landmarks[RIGHT[0]], landmarks[RIGHT[1]])
    right_height = distance(landmarks[RIGHT[2]], landmarks[RIGHT[3]])
    right_EAR = right_height / right_width if right_width != 0 else 0

    return (left_EAR + right_EAR) / 2


def detect_eye_closed(landmarks):
    ear = compute_eye_aspect_ratio(landmarks)
    if ear < 0.16:
        return True, ear
    return False, ear



def get_eye_roi(eye, image, w, h):
    x = int(eye.x * w)
    y = int(eye.y * h)
    size = int(0.04 * w)

    roi = image[max(0, y-size):min(h, y+size),
                max(0, x-size):min(w, x+size)]
    return roi


def histogram_spread(gray_roi):
    hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
    return np.std(hist)


def detect_sunglasses(left_eye, right_eye, image, w, h):
    left_roi = get_eye_roi(left_eye, image, w, h)
    right_roi = get_eye_roi(right_eye, image, w, h)

    if left_roi.size == 0 or right_roi.size == 0:
        return False

    left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    left_brightness = np.mean(left_gray)
    right_brightness = np.mean(right_gray)

    left_contrast = np.std(left_gray)
    right_contrast = np.std(right_gray)

    left_spread = histogram_spread(left_gray)
    right_spread = histogram_spread(right_gray)

    return (
        left_brightness < 60 and right_brightness < 60 and
        left_contrast < 25 and right_contrast < 25 and
        left_spread < 500 and right_spread < 500
    )


# =====================================================
# ================= MASK DETECTION ====================
# =====================================================

def detect_mask(nose, chin, left_mouth, right_mouth, image, w, h):
    lower_face_height = abs(chin.y - nose.y)
    mouth_width = abs(right_mouth.x - left_mouth.x)

    geometry_flag = lower_face_height < 0.09 or mouth_width < 0.05

    x1 = int(left_mouth.x * w)
    x2 = int(right_mouth.x * w)
    y1 = int(nose.y * h)
    y2 = int(chin.y * h)

    roi = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)

    texture_flag = texture_variance < 300

    return geometry_flag and texture_flag




def detect_cap(nose, chin, forehead, image, w, h):
    forehead_height = abs(nose.y - forehead.y)
    face_height = abs(chin.y - forehead.y)

    geometry_flag = forehead_height < 0.05 and face_height < 0.42

    x = int(forehead.x * w)
    y = int(forehead.y * h)
    size = int(0.06 * w)

    roi = image[max(0, y-size):y,
                max(0, x-size):min(w, x+size)]

    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)

    texture_flag = texture_variance < 250

    return geometry_flag and texture_flag




def compute_confidence_score(yaw, eye_ratio, ear):
    score = 100

    if yaw > 30:
        score -= 30
    elif yaw > 20:
        score -= 15

    if eye_ratio < 0.4:
        score -= 10

    if ear < 0.2:
        score -= 20

    return max(0, score)




def analyze_face(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return False, "Invalid image", 0, 0

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as detector:
        result = detector.process(rgb)
        if not result.detections:
            return False, "No face detected", 0, 0
        if len(result.detections) > 1:
            return False, "Multiple faces detected", 0, 0

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.7) as mesh:

        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return False, "Face not clearly visible. Please ensure good lighting and a front-facing position.", 0, 0


        landmarks = result.multi_face_landmarks[0].landmark

        LEFT_EYE = landmarks[33]
        RIGHT_EYE = landmarks[263]
        NOSE = landmarks[1]
        CHIN = landmarks[152]
        LEFT_MOUTH = landmarks[61]
        RIGHT_MOUTH = landmarks[291]
        FOREHEAD = landmarks[10]
        LEFT_FACE = landmarks[234]
        RIGHT_FACE = landmarks[454]

        yaw = compute_yaw_angle(LEFT_EYE, RIGHT_EYE, NOSE)
        eye_ratio = compute_eye_ratio(LEFT_EYE, RIGHT_EYE, NOSE)

        side_flag, msg = check_side_profile(yaw, eye_ratio)
        if side_flag:
            return False, msg, yaw, 0

        eye_closed, ear = detect_eye_closed(landmarks)
        if eye_closed:
            return False, "Eyes are closed", yaw, 0

        if detect_sunglasses(LEFT_EYE, RIGHT_EYE, image, w, h):
            return False, "Black sunglasses detected", yaw, 0

        if detect_mask(NOISE:=NOSE, chin=CHIN, left_mouth=LEFT_MOUTH,
                       right_mouth=RIGHT_MOUTH, image=image, w=w, h=h):
            return False, "Face mask detected", yaw, 0

        if detect_cap(NOSE, CHIN, FOREHEAD, image, w, h):
            return False, "Cap or helmet detected", yaw, 0

        confidence = compute_confidence_score(yaw, eye_ratio, ear)

    return True, "Your photo is front facing", yaw, confidence
