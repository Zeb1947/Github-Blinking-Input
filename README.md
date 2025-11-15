# GDATHCI Eye Tracker Project Documentation

## GDATHCI (Geometry Dash Assistive Technology and Human-Computer Interaction)

The **GDATHCI** project demonstrates how **assistive technology** and **human-computer interaction (HCI)** allow humans to control computers naturally and inclusively. It focuses on using **blinking as an input method**, replacing traditional devices like keyboards or controllers. Using a simple webcam and an advanced computer program developed in Python with a few libraries, the system detects eye blinks to control a game, such as **Geometry Dash**.

Inspired by accessibility systems for people with limited mobility, GDATHCI uses **computer vision** to track the face and detect eye blinks. A blink triggers a jump (simulating a spacebar press), while eyes remaining closed simulate holding the jump button (similar to holding the spacebar).

The program combines **artificial intelligence**, **pattern recognition**, and **real-time video analysis** for efficiency. Even low-quality webcams work, as the system analyzes eye structure and movement patterns to improve precision, adapting to varying conditions and hardware.

GDATHCI is a creative application combining **physics** (light/colour detection), **biology** (eye behaviour), and **computer science** (image processing). Future applications could include hands-free interfaces for education, games, or assistive devices.

---

## Technical Overview: How It Works

The core of the GDATHCI tracker relies on the **Eye Aspect Ratio (EAR)**, derived from facial landmarks provided by **MediaPipe**.

### Face and Eye Tracking (MediaPipe)

* The system uses the **MediaPipe Face Mesh** library to identify **468 specific landmarks** on the face in real-time.
* It extracts **6 key landmarks** around each eye to monitor their state (open or closed).

### Eye Aspect Ratio (EAR) Calculation

The EAR is a mathematical measure of the ratio between the vertical and horizontal distances of the eye landmarks.

$$
EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}
$$

Where $p_1$ to $p_6$ are the six eye coordinates.

- When the eye is open, the EAR is high (e.g., 0.30 – 0.40).  
- When the eye is closed (blinking), the EAR drops sharply (e.g., below 0.15).

### Calibration and Blink Detection

* **Calibration:** A brief **3-second calibration** establishes the user's **Baseline EAR**, accounting for individual differences and environment.
* **Blink Detection:** The program calculates a **Threshold**:  

$$
Threshold = Baseline\ EAR \times Sensitivity
$$

If the current average EAR falls below this threshold, a closed eye is detected.

* **Control:** It uses the **`pynput`** library to simulate keyboard actions:
    * Eye closes → `Key.space` is **pressed**.
    * Eye opens → `Key.space` is **released**.

---

## Requirements and Installation

The GDATHCI program is written in Python 3.9+ and requires the following libraries:

```bash
pip install opencv-python mediapipe numpy pynput
```

To run on Windows, just double click the file or run:
```bash
python gmBlink_windows.py
```

For Linux, it is almost the same:
```bash
python3 gmBlink_linux.py
```
