import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.keyboard import Key, Controller, Listener
import tkinter as tk
from tkinter import ttk
import ctypes
import os

# ------------------ SETTINGS ------------------
default_settings = {"sensitivity": 0.65, "fps": 30, "baseline": 0.291}

# ------------------ WINDOWS TWEAKS ------------------
def optimize_for_windows():
    try:
        os.system("")  # Enable ANSI colors in Windows Terminal
        ctypes.windll.kernel32.SetConsoleTitleW("Ultimate Eye Tracker")
        print("\033[96m[INFO]\033[0m Optimized for Windows environment.")
    except Exception:
        print("[WARN] Windows optimization skipped.")

# ------------------ DIAGNOSTIC ------------------
def run_diagnostic():
    print("\n--- SYSTEM DIAGNOSTIC ---")
    try:
        print(f"OpenCV Version: {cv2.__version__}")
        print("Mediapipe Available: OK")
        cv2.setUseOptimized(True)
    except Exception as e:
        print(f"Diagnostic Error: {e}")
    print("--------------------------\n")

# ------------------ TRACKER ------------------
def run_tracker(sensitivity, fps, baseline):
    keyboard = Controller()
    blink_enabled = True
    last_state = False
    blink_count = 0

    LEFT_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_IDXS = [362, 385, 387, 263, 373, 380]
    threshold = baseline * sensitivity

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                      min_detection_confidence=0.6,
                                      min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-friendly DirectShow backend
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check privacy settings.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, fps)

    window_name = "Ultimate Eye Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    prev_time = time.time()

    def eye_aspect_ratio(pts):
        A = np.linalg.norm(pts[1]-pts[5])
        B = np.linalg.norm(pts[2]-pts[4])
        C = np.linalg.norm(pts[0]-pts[3])
        return (A+B)/(2.0*C)

    def on_press(key):
        nonlocal blink_enabled
        try:
            if key.char == 'p':
                blink_enabled = not blink_enabled
                print(f"[TOGGLE] Tracker {'ON' if blink_enabled else 'OFF'}")
        except AttributeError:
            pass

    listener = Listener(on_press=on_press, daemon=True)
    listener.start()

    def put_text(img, text, pos, color=(255,255,255), scale=0.7):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        eyes_found = False
        eye_closed = False

        if results.multi_face_landmarks:
            mesh = np.array([(p.x*w, p.y*h) for p in results.multi_face_landmarks[0].landmark])
            left_eye = mesh[LEFT_IDXS]
            right_eye = mesh[RIGHT_IDXS]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear)/2
            eyes_found = True
            eye_closed = ear < threshold

            if eye_closed and not last_state:
                blink_count += 1

            # Draw overlays
            for pts in [left_eye, right_eye]:
                cv2.polylines(overlay, [np.int32(pts)], True, (0,255,255), 1, cv2.LINE_AA)

            color = (0,0,255) if eye_closed else (0,255,0)
            status = "EYES CLOSED" if eye_closed else "EYES OPEN"
            put_text(overlay, f"EAR L:{left_ear:.3f} R:{right_ear:.3f} AVG:{ear:.3f}", (20,30))
            put_text(overlay, f"Threshold: {threshold:.3f}", (20,60), (0,255,255))
            put_text(overlay, f"STATUS: {status}", (20,90), color, 0.8)
            put_text(overlay, f"BLINKS: {blink_count}", (20,120), (255,255,0))

        # FPS
        curr_time = time.time()
        fps_calc = 1/(curr_time-prev_time)
        prev_time = curr_time
        put_text(overlay, f"FPS: {fps_calc:.1f}", (20,h-20), (200,200,200))

        toggle_color = (0,255,0) if blink_enabled else (0,0,255)
        put_text(overlay, f"TRACKER {'ON' if blink_enabled else 'OFF'}", (w-220, 30), toggle_color, 0.8)

        if blink_enabled and eyes_found:
            if eye_closed and not last_state:
                try: keyboard.press(Key.space)
                except: pass
                last_state=True
            elif not eye_closed and last_state:
                try: keyboard.release(Key.space)
                except: pass
                last_state=False
        else:
            if last_state:
                try: keyboard.release(Key.space)
                except: pass
                last_state=False

        cv2.imshow(window_name, overlay)
        if cv2.waitKey(1) & 0xFF==27:  # ESC
            if last_state: keyboard.release(Key.space)
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()
    print("[EXIT] Tracker closed successfully.")

# ------------------ CALIBRATION ------------------
def calibrate_baseline():
    print("[INFO] Starting Calibration: Keep eyes open for 3 seconds.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Using default baseline.")
        return default_settings["baseline"]

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    LEFT_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_IDXS = [362, 385, 387, 263, 373, 380]

    calibration_ears = []
    start = time.time()

    def ear(pts):
        A = np.linalg.norm(pts[1]-pts[5])
        B = np.linalg.norm(pts[2]-pts[4])
        C = np.linalg.norm(pts[0]-pts[3])
        return (A+B)/(2.0*C)

    while time.time()-start < 3:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w = frame.shape[:2]
        if results.multi_face_landmarks:
            mesh = np.array([(p.x*w,p.y*h) for p in results.multi_face_landmarks[0].landmark])
            left_ear = ear(mesh[LEFT_IDXS])
            right_ear = ear(mesh[RIGHT_IDXS])
            calibration_ears.append((left_ear+right_ear)/2)

        cv2.putText(frame,f"Calibrating... {(3-int(time.time()-start))}s",
                    (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Calibration",frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.1)
    if calibration_ears:
        baseline=sum(calibration_ears)/len(calibration_ears)
        print(f"[INFO] Baseline EAR: {baseline:.3f}")
        return baseline
    else:
        print("[WARN] Calibration failed. Using default baseline.")
        return default_settings["baseline"]

# ------------------ GUI ------------------
def get_settings_from_gui(baseline):
    settings = {}
    def on_launch():
        nonlocal settings
        try:
            settings["sensitivity"]=float(sens_var.get())
            settings["fps"]=int(fps_var.get())
            settings["baseline"]=float(base_var.get())
        except:
            settings = default_settings
        root.destroy()

    root = tk.Tk()
    root.title("Eye Tracker Launcher (Windows)")
    root.geometry("400x260")
    root.resizable(False, False)

    tk.Label(root,text="Eye Tracker Configuration",font=("Segoe UI",14,"bold")).pack(pady=10)
    frame=tk.Frame(root); frame.pack(pady=5)

    tk.Label(frame,text="Sensitivity (0-1):").grid(row=0,column=0,sticky="w",padx=5,pady=2)
    sens_var=tk.StringVar(value=str(default_settings["sensitivity"]))
    tk.Entry(frame,textvariable=sens_var,width=10).grid(row=0,column=1,padx=5)

    tk.Label(frame,text="Camera FPS:").grid(row=1,column=0,sticky="w",padx=5,pady=2)
    fps_var=tk.StringVar(value=str(default_settings["fps"]))
    tk.Entry(frame,textvariable=fps_var,width=10).grid(row=1,column=1,padx=5)

    tk.Label(frame,text="Baseline EAR:").grid(row=2,column=0,sticky="w",padx=5,pady=2)
    base_var=tk.StringVar(value=f"{baseline:.3f}")
    tk.Entry(frame,textvariable=base_var,width=10).grid(row=2,column=1,padx=5)

    ttk.Button(root,text="Launch Tracker",command=on_launch).pack(pady=20)
    tk.Label(root,text="Toggle ON/OFF: Press 'p' key after launch",fg="gray").pack()
    root.mainloop()
    return settings or default_settings

# ------------------ MAIN ------------------
if __name__=="__main__":
    optimize_for_windows()
    run_diagnostic()
    baseline=calibrate_baseline()
    settings=get_settings_from_gui(baseline)
    run_tracker(settings["sensitivity"],settings["fps"],settings["baseline"])
