#! python3.11
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import os
import face_core

class FaceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x600")
        
        # --- Variabler ---
        self.mode = "RECOGNIZE" # RECOGNIZE eller CAPTURE
        self.capture_count = 0
        self.capture_name = ""
        self.capture_role = ""
        self.user_roles = face_core.load_roles()
        
        # Ladda modell
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        if os.path.exists(face_core.MODEL_FILE):
            self.recognizer.read(face_core.MODEL_FILE)
            # Vi måste ladda om label_map genom att "simulera" träning eller spara den separat. 
            # För enkelhetens skull kör vi en snabb inläsning av datamappen för att bygga label_map
            self.label_map = self.build_label_map_fast()
        
        # Vi använder nu MediaPipe via face_core, så vi behöver ingen Cascade här
        # self.face_cascade = ...

        # --- GUI Layout ---
        # Vänster: Videoström
        self.video_frame = tk.Frame(window, width=640, height=480, bg="black")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack()

        # Höger: Kontroller
        self.control_frame = tk.Frame(window, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(self.control_frame, text="Kontrollpanel", font=("Arial", 16, "bold")).pack(pady=10)

        # Registrering
        self.reg_frame = tk.LabelFrame(self.control_frame, text="Ny Användare", padx=10, pady=10)
        self.reg_frame.pack(fill="x", pady=10)
        
        tk.Label(self.reg_frame, text="Namn:").pack(anchor="w")
        self.entry_name = tk.Entry(self.reg_frame)
        self.entry_name.pack(fill="x")

        tk.Label(self.reg_frame, text="Roll (Teacher/Student):").pack(anchor="w")
        self.combo_role = ttk.Combobox(self.reg_frame, values=["Teacher", "Student"])
        self.combo_role.current(1)
        self.combo_role.pack(fill="x", pady=5)

        self.btn_capture = tk.Button(self.reg_frame, text="Starta Registrering", command=self.start_capture, bg="#4CAF50", fg="white")
        self.btn_capture.pack(fill="x", pady=10)

        # Status
        self.status_label = tk.Label(self.reg_frame, text="Redo", fg="grey")
        self.status_label.pack()

        # Träning
        self.train_frame = tk.LabelFrame(self.control_frame, text="System", padx=10, pady=10)
        self.train_frame.pack(fill="x", pady=10)
        
        tk.Button(self.train_frame, text="Träna om modell manuellt", command=self.retrain_model).pack(fill="x")
        tk.Button(self.train_frame, text="Avsluta", command=window.quit, bg="#f44336", fg="white").pack(fill="x", pady=5)

        # --- Kamera Start ---
        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.update_video()

    def build_label_map_fast(self):
        # Hjälpfunktion för att snabbt återskapa ID->Namn mappning från mappen
        current_label = 0
        l_map = {}
        if not os.path.exists(face_core.DATASET_PATH): return {}
        for person in os.listdir(face_core.DATASET_PATH):
            path = os.path.join(face_core.DATASET_PATH, person)
            if os.path.isdir(path):
                l_map[current_label] = person
                current_label += 1
        return l_map

    def start_capture(self):
        name = self.entry_name.get().strip()
        role = self.combo_role.get().strip()
        
        if not name:
            messagebox.showwarning("Fel", "Du måste skriva ett namn!")
            return
            
        self.capture_name = name
        self.capture_role = role
        self.capture_count = 0
        self.mode = "CAPTURE"
        self.user_roles[name] = role
        face_core.save_role(name, role)
        
        # Skapa mapp
        path = f"{face_core.DATASET_PATH}/{name}"
        os.makedirs(path, exist_ok=True)
        
        self.btn_capture.config(state="disabled", text="Tar bilder...")

    def retrain_model(self):
        self.status_label.config(text="Tränar modell... Vänta.")
        self.window.update()
        self.label_map = face_core.train_model()
        # Skapa om recognizer för att vara säker på att den ladda om
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(face_core.MODEL_FILE)
        self.status_label.config(text="Träning klar!", fg="green")
        messagebox.showinfo("Klar", "Modellen är uppdaterad!")

    def update_video(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Använd Haar Cascade via face_core
            faces = face_core.detect_faces(gray)

            for (x, y, w, h) in faces:
                color = (0, 255, 0)
                
                if self.mode == "CAPTURE":
                    if self.capture_count < face_core.MAX_IMAGES:
                        self.capture_count += 1
                        save_path = f"{face_core.DATASET_PATH}/{self.capture_name}/{self.capture_count}.jpg"
                        cv2.imwrite(save_path, gray[y:y+h, x:x+w])
                        self.status_label.config(text=f"Bild: {self.capture_count}/{face_core.MAX_IMAGES}", fg="blue")
                    else:
                        self.mode = "RECOGNIZE"
                        self.btn_capture.config(state="normal", text="Starta Registrering")
                        self.status_label.config(text="Registrering klar! Tränar...", fg="orange")
                        # Träna automatiskt
                        self.window.after(100, self.retrain_model)
                
                elif self.mode == "RECOGNIZE":
                    text = "Unknown"
                    if self.label_map:
                        try:
                            lbl, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                            if conf < 70 and lbl in self.label_map:
                                name = self.label_map[lbl]
                                role = self.user_roles.get(name, "")
                                text = f"{name} ({role})"
                        except:
                            pass
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Konvertera för Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.window.imgtk = imgtk # Behåll referens för att undvika Garbage Collection

        self.window.after(10, self.update_video)

# --- Starta Appen ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root, "Face Recognition System PRO")
    root.mainloop()
