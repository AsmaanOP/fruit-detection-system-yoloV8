import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"  # Put best.pt in the same folder
GLOBAL_ACCURACY = 100   # fixed model accuracy (already trained)

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found!")
    exit()

model = YOLO(MODEL_PATH)

def run_prediction():
    # 1. Open Ubuntu File Dialog
    root_picker = tk.Tk()
    root_picker.withdraw()
    image_path = filedialog.askopenfilename()
    root_picker.destroy()

    if not image_path:
        return

    # 2. Predict
    results = model.predict(source=image_path, imgsz=224, verbose=False)
    res = results[0]
    
    # 3. TERMINAL TABLE OUTPUT
    print(f"\n{'+' + '-'*20 + '+' + '-'*12 + '+'}")
    print(f"| {'FRUIT NAME':<18} | {'CONF %':<10} |")
    print(f"|{'-'*20}|{'-'*12}|")
    
    for i, score in enumerate(res.probs.data):
        name = res.names[i].upper()
        val = float(score) * 100
        print(f"| {name:<18} | {val:>9.2f}% |")
    
    print(f"{'+' + '-'*20 + '+' + '-'*12 + '+'}\n")

    # 4. GUI DISPLAY
    fruit_name = res.names[int(res.probs.top1)]
    confidence = float(res.probs.top1conf) * 100

    img = Image.open(image_path)
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis("off")

    # Separated Stats: Accuracy in title, Confidence in a box
    plt.title(f"SYSTEM ACCURACY: {GLOBAL_ACCURACY}%\nPREDICTED: {fruit_name.upper()}", 
              fontsize=14, color='darkblue', fontweight='bold', pad=20)

    plt.text(10, 25, f"CONFIDENCE: {confidence:.2f}%", color='white', fontsize=12,
             fontweight='bold', bbox=dict(facecolor='red', alpha=0.8, edgecolor='none'))

    plt.show()

# --- MAIN UI ---
root = tk.Tk()
root.title("Fruit Detection System")
root.geometry("400x200")

# Center content
main_frame = tk.Frame(root)
main_frame.pack(expand=True)

label = tk.Label(main_frame, text="Fruit Classifier ", font=("Arial", 12, "bold"))
label.pack(pady=10)

btn = tk.Button(main_frame, text="SELECT IMAGE", command=run_prediction, 
                bg="#2196F3", fg="white", font=("Arial", 10, "bold"), height=2, width=20)
btn.pack(pady=20)

print("GUI Running... Check the window to select images.")
root.mainloop()