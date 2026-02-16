import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


# =====================================================
# ================= BASIC UTILITIES ===================
# =====================================================

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# =====================================================
# ================= FULL FACE CHECK ===================
# =====================================================

def check_full_face_visible(left_face, right_face, forehead, chin):
    face_width = abs(right_face.x - left_face.x)
    face_height = abs(chin.y - forehead.y)

    # Minimum face size requirement
    if face_width < 0.25:
        return False

    if face_height < 0.30:
        return False

    # Face should not touch borders
    if left_face.x < 0.05 or right_face.x > 0.95:
        return False

    if forehead.y < 0.05 or chin.y > 0.95:
        return False

    return True


# =====================================================
# ================= IMPROVED YAW ======================
# =====================================================

def compute_yaw_improved(left_face, right_face, nose):
    left_width = abs(nose.x - left_face.x)
    right_width = abs(right_face.x - nose.x)

    total = left_width + right_width
    if total == 0:
        return 0

    symmetry = abs(left_width - right_width) / total
    yaw_score = symmetry * 90  # scale to degree-like value

    return yaw_score


def compute_eye_ratio(left_eye, right_eye, nose):
    left_dist = abs(nose.x - left_eye.x)
    right_dist = abs(right_eye.x - nose.x)

    max_dist = max(left_dist, right_dist)
    if max_dist == 0:
        return 1.0

    return min(left_dist, right_dist) / max_dist


def check_side_profile(yaw_score, eye_ratio):
    if yaw_score > 35:
        return True, "Side facing photo not allowed"

    if eye_ratio < 0.45:
        return True, "Side facing photo not allowed"

    return False, ""


# =====================================================
# ================= EYE CLOSED ========================
# =====================================================

def compute_eye_aspect_ratio(landmarks):
    LEFT = [33, 133, 159, 145]
    RIGHT = [263, 362, 386, 374]

    left_width = distance(landmarks[LEFT[0]], landmarks[LEFT[1]])
    left_height = distance(landmarks[LEFT[2]], landmarks[LEFT[3]])
    left_EAR = left_height / left_width if left_width != 0 else 0

    right_width = distance(landmarks[RIGHT[0]], landmarks[RIGHT[1]])
    right_height = distance(landmarks[RIGHT[2]], landmarks[RIGHT[3]])
    right_EAR = right_height / right_width if right_width != 0 else 0

    return (left_EAR + right_EAR) / 2


def detect_eye_closed(landmarks):
    ear = compute_eye_aspect_ratio(landmarks)
    if ear < 0.18:
        return True, ear
    return False, ear


# =====================================================
# ================= SUNGLASSES ========================
# =====================================================

def detect_sunglasses(left_eye, right_eye, image, w, h):

    def get_eye_roi(eye):
        x = int(eye.x * w)
        y = int(eye.y * h)
        size = int(0.04 * w)
        return image[max(0, y-size):min(h, y+size),
                     max(0, x-size):min(w, x+size)]

    left_roi = get_eye_roi(left_eye)
    right_roi = get_eye_roi(right_eye)

    if left_roi.size == 0 or right_roi.size == 0:
        return False

    left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    left_brightness = np.mean(left_gray)
    right_brightness = np.mean(right_gray)

    left_contrast = np.std(left_gray)
    right_contrast = np.std(right_gray)

    return (
        left_brightness < 60 and right_brightness < 60 and
        left_contrast < 25 and right_contrast < 25
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

    roi = image[min(y1, y2):max(y1, y2),
                min(x1, x2):max(x1, x2)]

    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)

    texture_flag = texture_variance < 300

    return geometry_flag and texture_flag


# =====================================================
# ================= CAP DETECTION =====================
# =====================================================

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


# =====================================================
# ================= CONFIDENCE SCORE ==================
# =====================================================

def compute_confidence_score(yaw, eye_ratio, ear):
    score = 100

    if yaw > 25:
        score -= 25
    elif yaw > 15:
        score -= 10

    if eye_ratio < 0.5:
        score -= 10

    if ear < 0.22:
        score -= 15

    return max(0, score)


# =====================================================
# ================= MAIN ANALYSIS =====================
# =====================================================

def analyze_face(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return False, "Invalid image", 0, 0

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face detection
    with mp_face_detection.FaceDetection(model_selection=1,
                                         min_detection_confidence=0.7) as detector:
        result = detector.process(rgb)

        if not result.detections:
            return False, "No face detected", 0, 0

        if len(result.detections) > 1:
            return False, "Multiple faces detected", 0, 0

    # Face mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.7) as mesh:

        result = mesh.process(rgb)

        if not result.multi_face_landmarks:
            return False, "Face not clearly visible", 0, 0

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

        # 🔥 Full face visibility check
        if not check_full_face_visible(LEFT_FACE, RIGHT_FACE, FOREHEAD, CHIN):
            return False, "Full face not clearly visible. Keep entire face inside frame.", 0, 0

        # 🔥 Improved yaw
        yaw = compute_yaw_improved(LEFT_FACE, RIGHT_FACE, NOSE)
        eye_ratio = compute_eye_ratio(LEFT_EYE, RIGHT_EYE, NOSE)

        side_flag, msg = check_side_profile(yaw, eye_ratio)
        if side_flag:
            return False, msg, yaw, 0

        # Eyes
        eye_closed, ear = detect_eye_closed(landmarks)
        if eye_closed:
            return False, "Eyes are closed", yaw, 0

        # Sunglasses
        if detect_sunglasses(LEFT_EYE, RIGHT_EYE, image, w, h):
            return False, "Black sunglasses detected", yaw, 0

        # Mask
        if detect_mask(NOSE, CHIN, LEFT_MOUTH, RIGHT_MOUTH, image, w, h):
            return False, "Face mask detected", yaw, 0

        # Cap
        if detect_cap(NOSE, CHIN, FOREHEAD, image, w, h):
            return False, "Cap or helmet detected", yaw, 0

        confidence = compute_confidence_score(yaw, eye_ratio, ear)

    return True, "Your photo is front facing", yaw, confidence
