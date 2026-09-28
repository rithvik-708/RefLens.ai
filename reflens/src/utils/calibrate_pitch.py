import cv2
import json
import os
import argparse
import sys

def calibrate_pitch(video_path: str, frame_num: int):
    print("="*50)
    print("RefLens Interactive Pitch Calibration")
    print("="*50)

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_num} from video.")
        return

    # Pitch configuration
    pitch_length = float(input("Enter pitch length in meters (default 105.0): ") or 105.0)
    pitch_width = float(input("Enter pitch width in meters (default 68.0): ") or 68.0)

    print("\nCoordinate System Reference:")
    print(f"X (Length): 0 to {pitch_length}")
    print(f"Y (Width) : 0 to {pitch_width}")
    print("Typically, X=0 is the left goal line, X=105 is the right goal line.")
    print("Y=0 is the top touchline, Y=68 is the bottom touchline.\n")

    calibration_data = {
        "video_path": video_path,
        "pitch_length_m": pitch_length,
        "pitch_width_m": pitch_width,
        "calibration_frame": frame_num,
        "points": []
    }

    window_name = "RefLens Pitch Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    click_queue = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.append((x, y))
            # Draw a temporary circle
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, frame)

    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("1. Click a known landmark on the pitch in the image window.")
    print("2. Return to this terminal and enter its X Y pitch coordinates (in meters).")
    print("3. Repeat for at least 4 points (preferably 6-8 spread out).")
    print("4. Press 'q' or 'ESC' in the image window to finish and save.\n")

    while True:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(10) & 0xFF

        if key == 27 or key == ord('q'):
            break

        if click_queue:
            img_x, img_y = click_queue.pop(0)
            print(f"\n=> Clicked image point: ({img_x}, {img_y})")
            
            while True:
                try:
                    coords = input(f"Enter pitch X and Y for this point (e.g., '0 34' or 's' to skip): ").strip()
                    if coords.lower() == 's':
                        print("Point skipped.")
                        break
                    
                    parts = coords.split()
                    if len(parts) != 2:
                        print("Please enter exactly two numbers separated by a space.")
                        continue
                        
                    pitch_x, pitch_y = float(parts[0]), float(parts[1])
                    
                    calibration_data["points"].append({
                        "image": [img_x, img_y],
                        "pitch": [pitch_x, pitch_y]
                    })
                    
                    # Draw a persistent confirmed point
                    cv2.circle(frame, (img_x, img_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"({pitch_x},{pitch_y})", (img_x + 10, img_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(window_name, frame)
                    print("Point saved.")
                    break
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")

    cv2.destroyAllWindows()
    
    if len(calibration_data["points"]) < 4:
        print(f"\nWarning: Only {len(calibration_data['points'])} points collected. Homography requires at least 4.")
    else:
        print(f"\nSuccess: Collected {len(calibration_data['points'])} points.")
        
    out_dir = "configs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "calibration.json")
    
    with open(out_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
        
    print(f"Calibration saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefLens Interactive Pitch Calibration")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to extract for calibration")
    args = parser.parse_args()
    
    calibrate_pitch(args.video, args.frame)
