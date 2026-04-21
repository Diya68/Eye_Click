# Eye_Controlled_Mouse_Interface

A Python application that enables hands-free computer control using eye tracking and facial gestures. Control your mouse cursor, click, and scroll using only your eyes and simple eye movements.

This Eye-Controlled Mouse Interface uses MediaPipe Face Mesh to track eye movements and control the cursor in real time. It supports gestures, adaptive calibration, and smooth control, offering a low-cost accessibility solution using a webcam.

## Features

- **Eye Tracking Cursor Control**: Move the mouse cursor by looking in different directions
- **Eye Roll Navigation**: Roll your eyes up/down/left/right for precise cursor movement
- **Blink & Wink Gestures**:
  - Left eye wink (short): Left click
  - Left eye wink (hold 0.5s): Double click
  - Right eye wink: Right click
- **Auto-Calibration**: 30-second guided calibration exercise for personalized settings
- **Stability Detection**: Cursor pauses when you focus on a specific point
- **Real-time Feedback**: Visual overlays showing controls, status, and FPS

## Requirements

- Python 3.7+
- Webcam
- Windows/Mac/Linux

## Installation

1. Clone or download this repository
2. Navigate to the Code_bases directory:
   ```bash
   cd Code_bases
   ```
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python eye_control.py
```

### Calibration

When you start the application, a 30-second calibration exercise will guide you through:
1. **Phase 1 (0-6s)**: Practice eye rolls (up, down, left, right)
2. **Phase 2 (6-12s)**: Practice blinking
3. **Phase 3 (12-18s)**: Practice winking
4. **Phase 4 (18-24s)**: Neutral position capture
5. **Phase 5 (24-30s)**: Final setup

### Controls

| Eye Movement | Action |
|-------------|--------|
| Look around | Move cursor smoothly |
| Eye roll UP | Move cursor UP |
| Eye roll DOWN | Move cursor DOWN |
| Eye roll LEFT | Move cursor LEFT |
| Eye roll RIGHT | Move cursor RIGHT |
| Wink Left Eye (short) | Left Click |
| Wink Left Eye (hold 0.5s) | Double Click |
| Wink Right Eye | Right Click |


### Keyboard Shortcuts

- **Q**: Quit application
- **R**: Restart calibration

## How It Works

The application uses computer vision and machine learning:

1. **Face Detection**: Uses MediaPipe Face Mesh to detect 468 facial landmarks
2. **Eye Tracking**: Tracks iris position relative to eye contours
3. **Gesture Recognition**: Detects eye aspect ratio changes for blinks and winks
4. **Smooth Movement**: Velocity-based cursor smoothing with dead zone for precision
5. **Calibration**: Learns your neutral eye position for personalized thresholds

## Project Structure

```
Eye_Controlled_Mouse_Interface/
├── Code_bases/
│   ├── eye_control.py      # Main application
│   ├── requirements.txt    # Python dependencies
│   └── venv/               # Virtual environment
└── README.md              # This file
```

## Dependencies

- `opencv-python` - Video capture and image processing
- `mediapipe` - Face mesh detection
- `pyautogui` - Mouse control
- `numpy` - Numerical operations
- `fastapi` & `uvicorn` - Web server (for future extensions)

## Troubleshooting

### Camera not detected
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index in the code

### Cursor movement is erratic
- Ensure good lighting conditions
- Re-run calibration by pressing 'R'
- Adjust distance from camera (recommended: 1-2 feet)

### Clicks not registering
- Blink/wink more deliberately during calibration
- Check that your eyes are clearly visible to the camera

### Performance issues
- Lower camera resolution
- Close other applications using the camera

## Future Enhancements

- Web-based interface for settings
- Custom gesture mapping
- Scroll functionality
- Keyboard input via eye gestures
- Multiple user profiles





