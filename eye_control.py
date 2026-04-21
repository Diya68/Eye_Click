import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import collections

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

screen_w, screen_h = pyautogui.size()

LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144]

# Calibration data - 30 second exercise
CALIBRATION_DURATION = 30  # seconds
calibrated = False
baseline_ear_left = 0
baseline_ear_right = 0
baseline_iris_pos = None

# Dynamic thresholds (set during calibration)
BLINK_THRESHOLD_LEFT = 0.22
BLINK_THRESHOLD_RIGHT = 0.22
EYE_ROLL_THRESHOLD = 0.4

# Cooldowns
ROLL_COOLDOWN = 0.35
CLICK_COOLDOWN = 0.4
RIGHT_CLICK_COOLDOWN = 0.5
ROLL_AMOUNT = 40  # Reduced for more controlled roll movement

# State variables
last_roll_time = 0
last_click_time = 0
last_right_click_time = 0
blink_state = 'open'
blink_start_time = 0
first_blink_time = 0
blink_count = 0

# Frame counters for stable detection
closed_frames = 0
open_frames = 0
BLINK_MIN_FRAMES = 3  # Increased to 3 frames for more stability
BLINK_MAX_FRAMES = 8  # More than this is a long blink, not a click
REQUIRED_OPEN_FRAMES = 4  # Eyes must be open this many frames before valid blink
consecutive_open_frames = 0  # Track consecutive open frames

# Iris position history to detect movement during blink
iris_position_history = collections.deque(maxlen=10)
BLINK_MAX_IRIS_MOVEMENT = 15  # Max pixels iris can move during valid blink

# Real-time rendering settings
FPS_HISTORY_SIZE = 30
SHOW_FPS = True
RENDER_QUALITY = 2  # Higher = more smoothing frames

# Movement history for smoothing
roll_history = collections.deque(maxlen=5)

# FPS tracking for real-time rendering
fps_history = collections.deque(maxlen=FPS_HISTORY_SIZE)
last_frame_time = time.time()

# Velocity smoothing
velocity_x = 0
velocity_y = 0
SMOOTHING_FACTOR = 0.03  # Reduced for smoother, slower movement

# Cursor speed control (lower = slower cursor)
CURSOR_SPEED = 0.15  # Reduced to 15% for much slower, controlled cursor

# Dead zone - cursor stops when movement is smaller than this (in pixels)
DEAD_ZONE_THRESHOLD = 8  # Ignore movements smaller than 8 pixels

# Position history for stability detection
POSITION_HISTORY_SIZE = 8
position_history = collections.deque(maxlen=POSITION_HISTORY_SIZE)

# Stability threshold - if position stays within this range, cursor stops
STABLE_THRESHOLD = 15  # pixels - if iris stays within 15px, cursor is stable
STABLE_FRAMES_REQUIRED = 5  # need 5 frames of stability to stop

# Roll mode - when True, cursor pauses and only scrolls
in_roll_mode = False
ROLL_MODE_COOLDOWN = 0.1

# Double-click timing (increased window for easier double-click)
DBL_CLICK_WINDOW = 0.45  # Was 0.35, now 450ms window


def eye_aspect_ratio(landmarks, eye_indices):
    v1 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    v2 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    h = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h)


def get_iris_center(points, iris_indices):
    iris = points[iris_indices]
    return np.mean(iris, axis=0)


def get_eye_center(points, eye_contour_indices):
    eye_contour = points[eye_contour_indices]
    return np.mean(eye_contour, axis=0)


def calibrate(frame, mesh):
    """Calibrate thresholds based on user's neutral eye state"""
    global baseline_ear_left, baseline_ear_right, baseline_iris_pos
    global BLINK_THRESHOLD_LEFT, BLINK_THRESHOLD_RIGHT, EYE_ROLL_THRESHOLD

    left_ear = eye_aspect_ratio(mesh, LEFT_EYE)
    right_ear = eye_aspect_ratio(mesh, RIGHT_EYE)
    iris_center = get_iris_center(mesh, LEFT_IRIS)
    eye_center = get_eye_center(mesh, LEFT_EYE_CONTOUR)

    baseline_ear_left = left_ear
    baseline_ear_right = right_ear
    baseline_iris_pos = iris_center - eye_center

    # Set thresholds based on calibration
    # Blink threshold is 70% of open eye EAR
    BLINK_THRESHOLD_LEFT = left_ear * 0.70
    BLINK_THRESHOLD_RIGHT = right_ear * 0.70
    # Roll threshold is based on eye size
    EYE_ROLL_THRESHOLD = 0.35

    return True


def detect_eye_roll(mesh, iris_indices, eye_contour_indices, baseline_offset):
    """Detect eye roll relative to calibrated neutral position"""
    iris_center = get_iris_center(mesh, iris_indices)
    eye_center = get_eye_center(mesh, eye_contour_indices)

    # Calculate current position relative to eye center
    current_offset = iris_center - eye_center

    # Subtract baseline to get relative movement
    dx = (current_offset[0] - baseline_offset[0]) / 22
    dy = (current_offset[1] - baseline_offset[1]) / 14

    # Require stronger movement to trigger
    if abs(dx) > abs(dy):
        if abs(dx) > EYE_ROLL_THRESHOLD:
            return 'left' if dx < 0 else 'right'
    else:
        if abs(dy) > EYE_ROLL_THRESHOLD:
            return 'up' if dy < 0 else 'down'
    return None


def draw_controls(frame):
    """Draw all controls/info on screen"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent panel on right side
    panel_w = 280
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "EYE CONTROL", (w - panel_w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(frame, (w - panel_w + 10, 40), (w - 10, 40), (0, 255, 255), 1)

    # Controls list
    y_pos = 70
    controls = [
        ("MOVEMENTS:", (255, 255, 255)),
        ("", (255, 255, 255)),
        ("  Eye Roll UP", (0, 255, 0)),
        ("  -> Cursor UP", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("  Eye Roll DOWN", (0, 255, 0)),
        ("  -> Cursor DOWN", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("  Eye Roll LEFT", (0, 255, 0)),
        ("  -> Cursor LEFT", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("  Eye Roll RIGHT", (0, 255, 0)),
        ("  -> Cursor RIGHT", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("CLICKS:", (255, 255, 255)),
        ("", (255, 255, 255)),
        ("  Wink Left Eye (short)", (0, 255, 255)),
        ("  -> Left Click", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("  Wink Left Eye (hold 0.5s)", (0, 255, 255)),
        ("  -> Double Click", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("  Wink Right Eye", (0, 255, 255)),
        ("  -> Right Click", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("EXIT:", (255, 255, 255)),
        ("  Press 'Q'", (0, 0, 255)),
    ]

    for text, color in controls:
        cv2.putText(frame, text, (w - panel_w + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 20

    return frame


def draw_status(frame, left_ear, right_ear, roll_dir, blink_st, closed_fr=0, in_roll=False, dbl_ready=False, fps=0, is_stable=False):
    """Draw current status at bottom with real-time metrics"""
    h, w = frame.shape[:2]

    # Status bar at bottom (taller for more info)
    cv2.rectangle(frame, (0, h - 50), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 50), (w, h), (100, 100, 100), 1)

    # Main status line
    status_text = f"L:{left_ear:.2f} R:{right_ear:.2f} | State:{blink_st}"
    if closed_fr > 0:
        status_text += f" | frm:{closed_fr}"
    if roll_dir:
        status_text += f" | ROLL:{roll_dir.upper()}"
    if in_roll:
        status_text += " [SCROLL]"
    if is_stable:
        status_text += " [STABLE]"

    # Choose color based on stability
    status_color = (0, 255, 0) if is_stable else (255, 255, 255)

    cv2.putText(frame, status_text, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # Second line with FPS and double-click status
    status_text2 = f""
    if fps > 0:
        status_text2 += f"FPS:{fps:.1f} | "
    if dbl_ready:
        # Visual countdown for double-click window
        cv2.rectangle(frame, (w//2 - 100, h - 25), (w//2 + 100, h - 5), (0, 255, 0), -1)
        cv2.putText(frame, "DBL-CLICK READY!", (w//2 - 80, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        status_text2 += "Blink x2 for double-click"
        cv2.putText(frame, status_text2, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


def draw_click_feedback(frame, click_type="left"):
    """Draw visual feedback when click occurs"""
    h, w = frame.shape[:2]

    # Flash effect
    overlay = frame.copy()

    if click_type == "left":
        color = (0, 255, 0)  # Green for left click
        text = "LEFT CLICK!"
    elif click_type == "double":
        color = (255, 255, 0)  # Cyan for double click
        text = "DOUBLE CLICK!"
    elif click_type == "right":
        color = (0, 150, 255)  # Orange for right click
        text = "RIGHT CLICK!"
    else:
        color = (255, 255, 255)
        text = "CLICK!"

    # Draw flash circle at bottom center
    center = (w // 2, h - 80)
    cv2.circle(overlay, center, 40, color, -1)
    cv2.circle(overlay, center, 45, (255, 255, 255), 3)

    # Add text
    cv2.putText(overlay, text, (center[0] - 80, center[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Blend with frame
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    return frame


def draw_calibration_exercise(frame, elapsed_time, phase, instruction, progress_pct,
                              roll_detected=None, blink_detected=False):
    """Draw the 30-second calibration exercise with phases"""
    h, w = frame.shape[:2]

    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "CALIBRATION EXERCISE", (w//2 - 180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Phase indicator
    cv2.putText(frame, f"Phase {phase}/5", (w//2 - 60, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Progress bar at top
    bar_w = 500
    bar_h = 20
    x = (w - bar_w) // 2
    y = 130

    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), -1)
    fill_w = int((elapsed_time / CALIBRATION_DURATION) * bar_w)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), (0, 255, 0), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 2)

    # Time remaining
    remaining = CALIBRATION_DURATION - int(elapsed_time)
    cv2.putText(frame, f"Time: {remaining}s", (w//2 - 50, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Main instruction box
    box_h = 120
    cv2.rectangle(frame, (50, h//2 - box_h//2), (w - 50, h//2 + box_h//2), (50, 50, 50), -1)
    cv2.rectangle(frame, (50, h//2 - box_h//2), (w - 50, h//2 + box_h//2), (0, 255, 255), 2)

    # Instruction text
    y_offset = h//2 - 20
    for line in instruction:
        cv2.putText(frame, line, (w//2 - 200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 35

    # Detection feedback
    if roll_detected:
        cv2.putText(frame, f"Detected: {roll_detected.upper()}!", (w//2 - 100, h//2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif blink_detected:
        cv2.putText(frame, "Blink Detected!", (w//2 - 100, h//2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Phase descriptions at bottom
    phases_y = h - 120
    phase_texts = [
        "1. Eye Roll Practice",
        "2. Blink Practice",
        "3. Wink Practice",
        "4. Final Setup"
    ]

    for i, text in enumerate(phase_texts):
        color = (0, 255, 0) if i + 1 == phase else (100, 100, 100)
        cv2.putText(frame, text, (50 + i * 180, phases_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def eye_control_loop():
    global calibrated, baseline_iris_pos
    global last_roll_time, last_click_time, last_right_click_time
    global blink_state, blink_start_time, blink_count, first_blink_time
    global roll_history, calibration_start_time
    global closed_frames, open_frames, velocity_x, velocity_y, in_roll_mode, dbl_click_ready
    global fps_history, last_frame_time, click_feedback, click_feedback_time
    global consecutive_open_frames, iris_position_history, position_history

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera not working")
        return

    print("Eye Control Starting...")
    print("30-second calibration exercise will begin.")
    print("Follow the on-screen instructions to practice each movement.")
    print("")

    calibration_start_time = None
    calibration_samples_left = []
    calibration_samples_right = []
    calibration_iris_offset = []

    # Track detected movements during calibration
    roll_up_detected = False
    roll_down_detected = False
    roll_left_detected = False
    roll_right_detected = False
    blink_detected_count = 0
    wink_detected_count = 0

    current_x, current_y = pyautogui.position()
    pyautogui.FAILSAFE = False

    # Visual feedback tracking
    click_feedback = None
    click_feedback_time = 0

    # Reset anti-fake-click tracking
    consecutive_open_frames = 0
    iris_position_history.clear()

    # Reset position history for stability detection
    position_history.clear()

    while True:
        # Calculate FPS
        current_frame_time = time.time()
        frame_time = current_frame_time - last_frame_time
        last_frame_time = current_frame_time
        if frame_time > 0:
            fps_history.append(1.0 / frame_time)
        avg_fps = np.mean(fps_history) if fps_history else 0

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            mesh = np.array([
                (int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
                for p in landmarks
            ])

            left_ear = eye_aspect_ratio(mesh, LEFT_EYE)
            right_ear = eye_aspect_ratio(mesh, RIGHT_EYE)
            iris_center = get_iris_center(mesh, LEFT_IRIS)
            eye_center = get_eye_center(mesh, LEFT_EYE_CONTOUR)
            current_offset = iris_center - eye_center

            # Calibration phase - 30 second exercise
            if not calibrated:
                if calibration_start_time is None:
                    calibration_start_time = time.time()
                    print("Starting 30-second calibration exercise...")

                elapsed = time.time() - calibration_start_time

                # Collect baseline data throughout
                calibration_samples_left.append(left_ear)
                calibration_samples_right.append(right_ear)
                calibration_iris_offset.append(current_offset)

                # Determine phase and instruction
                if elapsed < 6:  # Phase 1: Eye rolls (0-6s)
                    phase = 1
                    instruction = ["PRACTICE: Roll your eyes", "UP, DOWN, LEFT, RIGHT", "Try different directions!"]

                    # Detect rolls during practice
                    dx = (current_offset[0] - np.mean([o[0] for o in calibration_iris_offset[-10:]])) / 22 if len(calibration_iris_offset) >= 10 else 0
                    dy = (current_offset[1] - np.mean([o[1] for o in calibration_iris_offset[-10:]])) / 14 if len(calibration_iris_offset) >= 10 else 0

                    roll_detected = None
                    if abs(dy) > abs(dx) and abs(dy) > 0.3:
                        roll_detected = 'up' if dy < 0 else 'down'
                        if roll_detected == 'up': roll_up_detected = True
                        if roll_detected == 'down': roll_down_detected = True
                    elif abs(dx) > abs(dy) and abs(dx) > 0.3:
                        roll_detected = 'left' if dx < 0 else 'right'
                        if roll_detected == 'left': roll_left_detected = True
                        if roll_detected == 'right': roll_right_detected = True

                    frame = draw_calibration_exercise(frame, elapsed, phase, instruction,
                                                       elapsed/CALIBRATION_DURATION, roll_detected)

                elif elapsed < 12:  # Phase 2: Blink practice (6-12s)
                    phase = 2
                    instruction = ["PRACTICE: Blink both eyes", "Close and open quickly", f"Blinks detected: {blink_detected_count}"]

                    # Simple blink detection during practice
                    left_closed = left_ear < (sum(calibration_samples_left[-20:]) / len(calibration_samples_left[-20:]) if len(calibration_samples_left) >= 20 else 0.25) * 0.7 if calibration_samples_left else 0.25
                    right_closed = right_ear < (sum(calibration_samples_right[-20:]) / len(calibration_samples_right[-20:]) if len(calibration_samples_right) >= 20 else 0.25) * 0.7 if calibration_samples_right else 0.25

                    blink_now = False
                    if hasattr(eye_control_loop, 'was_closed'):
                        if eye_control_loop.was_closed and not (left_closed and right_closed):
                            blink_detected_count += 1
                            blink_now = True
                    eye_control_loop.was_closed = left_closed and right_closed

                    frame = draw_calibration_exercise(frame, elapsed, phase, instruction,
                                                       elapsed/CALIBRATION_DURATION, blink_detected=blink_now)

                elif elapsed < 18:  # Phase 3: Wink practice (12-18s)
                    phase = 3
                    instruction = ["PRACTICE: Wink either eye", "Close only left or right", f"Winks detected: {wink_detected_count}"]

                    # Simple wink detection (either eye)
                    left_closed = left_ear < (sum(calibration_samples_left[-20:]) / len(calibration_samples_left[-20:]) if len(calibration_samples_left) >= 20 else 0.25) * 0.7 if calibration_samples_left else 0.25
                    right_closed = right_ear < (sum(calibration_samples_right[-20:]) / len(calibration_samples_right[-20:]) if len(calibration_samples_right) >= 20 else 0.25) * 0.7 if calibration_samples_right else 0.25

                    wink_now = False
                    wink_detected = right_closed and not left_closed or left_closed and not right_closed
                    if hasattr(eye_control_loop, 'was_winking'):
                        if eye_control_loop.was_winking and not wink_detected:
                            wink_detected_count += 1
                            wink_now = True
                    eye_control_loop.was_winking = wink_detected

                    frame = draw_calibration_exercise(frame, elapsed, phase, instruction,
                                                       elapsed/CALIBRATION_DURATION, blink_detected=wink_now)

                elif elapsed < 24:  # Phase 4: Neutral position (18-24s)
                    phase = 4
                    instruction = ["Keep eyes OPEN", "Look straight ahead", "Final calibration..."]
                    frame = draw_calibration_exercise(frame, elapsed, phase, instruction,
                                                       elapsed/CALIBRATION_DURATION)

                else:  # Phase 5: Ready (24-30s)
                    phase = 5
                    up_check = "OK" if roll_up_detected else "X"
                    down_check = "OK" if roll_down_detected else "X"
                    left_check = "OK" if roll_left_detected else "X"
                    right_check = "OK" if roll_right_detected else "X"
                    instruction = ["CALIBRATION COMPLETE!", "Get ready to control...", f"Rolls: U:{up_check} D:{down_check} L:{left_check} R:{right_check}"]
                    frame = draw_calibration_exercise(frame, elapsed, phase, instruction,
                                                       elapsed/CALIBRATION_DURATION)

                # Show frame
                cv2.imshow("Eye Control", frame)

                # Check if calibration complete
                if elapsed >= CALIBRATION_DURATION:
                    # Calculate final thresholds
                    baseline_ear_left = np.mean(calibration_samples_left)
                    baseline_ear_right = np.mean(calibration_samples_right)
                    baseline_iris_pos = np.mean(calibration_iris_offset, axis=0)

                    # Set calibrated thresholds
                    global BLINK_THRESHOLD_LEFT, BLINK_THRESHOLD_RIGHT
                    BLINK_THRESHOLD_LEFT = baseline_ear_left * 0.65
                    BLINK_THRESHOLD_RIGHT = baseline_ear_right * 0.65

                    calibrated = True
                    print(f"\nCalibration complete!")
                    print(f"  Baseline L-EAR: {baseline_ear_left:.3f}")
                    print(f"  Baseline R-EAR: {baseline_ear_right:.3f}")
                    print(f"  Blink threshold L: {BLINK_THRESHOLD_LEFT:.3f}")
                    print(f"  Blink threshold R: {BLINK_THRESHOLD_RIGHT:.3f}")
                    print(f"  Eye roll threshold: {EYE_ROLL_THRESHOLD:.3f}")
                    print(f"\nMovements practiced:")
                    print(f"  Rolls: Up={roll_up_detected}, Down={roll_down_detected}, Left={roll_left_detected}, Right={roll_right_detected}")
                    print(f"  Blinks: {blink_detected_count}, Winks: {wink_detected_count}")
                    print("")
                    print("Controls:")
                    print("  Eye Roll UP    -> Cursor UP")
                    print("  Eye Roll DOWN  -> Cursor DOWN")
                    print("  Eye Roll LEFT  -> Cursor LEFT")
                    print("  Eye Roll RIGHT -> Cursor RIGHT")
                    print("  Wink Left (short)      -> Left Click")
                    print("  Wink Left (hold 0.5s)  -> Double Click")
                    print("  Wink Right Eye         -> Right Click")
                    print("  Press 'Q' to quit, 'R' to recalibrate")
                    print("")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
                continue

            # Main control phase
            if calibrated:
                # Get iris for tracking
                iris = get_iris_center(mesh, LEFT_IRIS)
                x, y = iris

                # Detect eye roll using calibrated baseline
                roll_direction = detect_eye_roll(mesh, LEFT_IRIS, LEFT_EYE_CONTOUR, baseline_iris_pos)

                # Add to history for smoothing roll detection
                if roll_direction:
                    roll_history.append(roll_direction)
                    # Require consistent detection (5 consecutive frames)
                    if len(roll_history) >= 5:
                        last_five = list(roll_history)[-5:]
                        if all(r == roll_history[-1] for r in last_five):
                            roll_direction = roll_history[-1]
                        else:
                            roll_direction = None
                else:
                    # Require several None frames before clearing roll mode
                    roll_history.append(None)

                current_time = time.time()
                status_text = ""

                # Handle eye roll with improved cooldown and smoothness
                roll_triggered = False
                if roll_direction and current_time - last_roll_time > ROLL_COOLDOWN:
                    last_roll_time = current_time
                    roll_history.clear()
                    roll_triggered = True
                    in_roll_mode = True

                    # Calculate target position with roll amount
                    if roll_direction == 'up':
                        target_y = max(0, current_y - ROLL_AMOUNT)
                        current_y = current_y + (target_y - current_y) * 0.5  # Smooth transition
                        pyautogui.moveTo(current_x, current_y)
                        status_text = "ROLL UP"
                    elif roll_direction == 'down':
                        target_y = min(screen_h, current_y + ROLL_AMOUNT)
                        current_y = current_y + (target_y - current_y) * 0.5
                        pyautogui.moveTo(current_x, current_y)
                        status_text = "ROLL DOWN"
                    elif roll_direction == 'left':
                        target_x = max(0, current_x - ROLL_AMOUNT)
                        current_x = current_x + (target_x - current_x) * 0.5
                        pyautogui.moveTo(current_x, current_y)
                        status_text = "ROLL LEFT"
                    elif roll_direction == 'right':
                        target_x = min(screen_w, current_x + ROLL_AMOUNT)
                        current_x = current_x + (target_x - current_x) * 0.5
                        pyautogui.moveTo(current_x, current_y)
                        status_text = "ROLL RIGHT"

                # Exit roll mode after cooldown
                if in_roll_mode and current_time - last_roll_time > ROLL_MODE_COOLDOWN:
                    in_roll_mode = False

                # Map iris to screen
                screen_x = np.interp(x, [0, frame.shape[1]], [0, screen_w])
                screen_y = np.interp(y, [0, frame.shape[0]], [0, screen_h])

                # Add current position to history for stability detection
                position_history.append((screen_x, screen_y))

                # Check if position is stable (looking at one point)
                is_stable = False
                if len(position_history) >= STABLE_FRAMES_REQUIRED:
                    recent_positions = list(position_history)[-STABLE_FRAMES_REQUIRED:]
                    # Calculate spread of recent positions
                    xs = [p[0] for p in recent_positions]
                    ys = [p[1] for p in recent_positions]
                    x_spread = max(xs) - min(xs)
                    y_spread = max(ys) - min(ys)
                    # Stable if both x and y are within threshold
                    if x_spread < STABLE_THRESHOLD and y_spread < STABLE_THRESHOLD:
                        is_stable = True

                # Enhanced smooth movement with velocity-based easing and dead zone
                # Only update position if not in roll mode and not stable
                if not in_roll_mode and not roll_triggered and not is_stable:
                    # Calculate target velocity (scaled by CURSOR_SPEED)
                    target_vx = (screen_x - current_x) * SMOOTHING_FACTOR * CURSOR_SPEED
                    target_vy = (screen_y - current_y) * SMOOTHING_FACTOR * CURSOR_SPEED

                    # Apply dead zone - ignore small movements
                    if abs(target_vx) < DEAD_ZONE_THRESHOLD * SMOOTHING_FACTOR:
                        target_vx = 0
                    if abs(target_vy) < DEAD_ZONE_THRESHOLD * SMOOTHING_FACTOR:
                        target_vy = 0

                    # Smooth velocity changes (inertia) - lower factor = more damping
                    velocity_x = velocity_x + (target_vx - velocity_x) * 0.15  # Slower response
                    velocity_y = velocity_y + (target_vy - velocity_y) * 0.15

                    # Apply movement only if velocity is significant
                    if abs(velocity_x) > 0.5 or abs(velocity_y) > 0.5:
                        current_x = current_x + velocity_x
                        current_y = current_y + velocity_y

                        # Clamp to screen bounds
                        current_x = max(0, min(screen_w, current_x))
                        current_y = max(0, min(screen_h, current_y))

                        pyautogui.moveTo(current_x, current_y)
                elif is_stable:
                    # When stable, gradually reduce velocity to stop cursor
                    velocity_x *= 0.5
                    velocity_y *= 0.5

                # Blink detection with frame consistency check
                left_closed = left_ear < BLINK_THRESHOLD_LEFT
                right_closed = right_ear < BLINK_THRESHOLD_RIGHT
                both_closed = left_closed and right_closed

                # Frame-based stability check to prevent fake clicks
                if left_closed or right_closed:
                    closed_frames += 1
                    open_frames = 0
                else:
                    open_frames += 1
                    if open_frames >= 2:  # Need 2 open frames to reset
                        closed_frames = 0

                # Stable blink requires minimum consecutive closed frames
                stable_blink = closed_frames >= BLINK_MIN_FRAMES and closed_frames <= BLINK_MAX_FRAMES

                # State machine with stable wink requirement
                if blink_state == 'open':
                    if left_closed and not right_closed:
                        # Left wink detected - start left click sequence
                        if current_time - last_click_time > 0.3:
                            blink_state = 'wink_left'
                            blink_start_time = current_time
                    elif right_closed and not left_closed:
                        # Right wink detected - start right click sequence
                        if current_time - last_right_click_time > 0.3:
                            blink_state = 'wink_right'
                            blink_start_time = current_time

                elif blink_state == 'wink_left':
                    if not left_closed:
                        # Calculate wink duration
                        wink_duration = current_time - blink_start_time

                        # Single long wink (0.5s - 1.0s) = Double Click
                        # Short wink (0.08s - 0.5s) = Single Click
                        if current_time - last_click_time > CLICK_COOLDOWN:
                            if 0.5 <= wink_duration < 1.0:
                                pyautogui.doubleClick()
                                print("Double Click!")
                                last_click_time = current_time
                                click_feedback = "double"
                                click_feedback_time = current_time
                            elif 0.08 <= wink_duration < 0.5:
                                pyautogui.click()
                                print("Left Click!")
                                last_click_time = current_time
                                click_feedback = "left"
                                click_feedback_time = current_time

                        blink_state = 'open'
                        closed_frames = 0
                    elif current_time - blink_start_time > 1.0:
                        # Timeout - too long holding wink
                        blink_state = 'open'
                        closed_frames = 0

                elif blink_state == 'wink_right':
                    if not right_closed:
                        wink_duration = current_time - blink_start_time
                        if 0.12 < wink_duration < 0.6:
                            if current_time - last_right_click_time > RIGHT_CLICK_COOLDOWN:
                                pyautogui.rightClick()
                                print("Right Click!")
                                last_right_click_time = current_time
                                click_feedback = "right"
                                click_feedback_time = current_time
                        blink_state = 'open'
                    elif current_time - blink_start_time > 0.8:
                        blink_state = 'open'

                # Handle single click after double-click window expires
                dbl_click_ready = blink_count == 1
                if dbl_click_ready and current_time - first_blink_time > DBL_CLICK_WINDOW:
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        print("Left Click!")
                        last_click_time = current_time
                        click_feedback = "left"
                        click_feedback_time = current_time
                    blink_count = 0
                    dbl_click_ready = False

                # Draw UI with FPS
                frame = draw_controls(frame)
                frame = draw_status(frame, left_ear, right_ear, roll_direction, blink_state, closed_frames, in_roll_mode, dbl_click_ready, avg_fps, is_stable)

                # Show click feedback if recent
                if click_feedback and current_time - click_feedback_time < 0.3:
                    frame = draw_click_feedback(frame, click_feedback)

                # Draw iris tracking point
                cv2.circle(frame, (int(iris[0]), int(iris[1])), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(iris[0]), int(iris[1])), 8, (255, 255, 255), 2)

        # Show frame
        display_scale = 1.2 if calibrated else 1.0
        display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Eye Control", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Quitting...")
            break
        if key == ord('r') or key == ord('R'):
            print("Recalibrating...")
            calibrated = False
            calibration_start_time = None
            calibration_samples_left.clear()
            calibration_samples_right.clear()
            calibration_iris_offset.clear()
            blink_detected_count = 0
            wink_detected_count = 0
            roll_up_detected = False
            roll_down_detected = False
            roll_left_detected = False
            roll_right_detected = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    eye_control_loop()
