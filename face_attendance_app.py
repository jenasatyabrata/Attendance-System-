import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import PIL.Image, PIL.ImageTk
import os
import face_recognition
import numpy as np
import pandas as pd
import pickle
import threading
import time

class FaceAttendanceApp:
    def __init__(self, window_title):
        self.window = tk.Tk()
        self.window.title(window_title)

        self.dataset_path = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_loaded = False
        self.default_model_filename = "face_encodings.pkl"
        self.current_model_path = self.default_model_filename
        self.face_match_tolerance = 0.6

        self.detected_persons_session = set()

        self.vid = None
        self.video_running = False
        self.video_panel = None
        self.default_video_source = 0
        self.selected_video_file_path = None
        self.processing_video_file_flag = False

        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        controls_frame = ttk.Frame(main_frame, padding="10")
        controls_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        main_frame.columnconfigure(0, weight=0)

        train_frame = ttk.LabelFrame(controls_frame, text="1. Training", padding="10")
        train_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        train_frame.columnconfigure(0, weight=1)

        self.btn_select_dataset = ttk.Button(train_frame, text="Select Dataset Directory", command=self.select_dataset_directory)
        self.btn_select_dataset.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.dataset_label = ttk.Label(train_frame, text="Dataset: Not selected")
        self.dataset_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        
        self.btn_train_model = ttk.Button(train_frame, text="Train and Save Model", command=self.train_model_thread, state=tk.DISABLED)
        self.btn_train_model.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.train_status_label = ttk.Label(train_frame, text="Status: Idle")
        self.train_status_label.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        model_frame = ttk.LabelFrame(controls_frame, text="2. Load Model", padding="10")
        model_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        model_frame.columnconfigure(0, weight=1)

        self.btn_load_model = ttk.Button(model_frame, text="Load Trained Model File", command=self.load_model_from_file_dialog)
        self.btn_load_model.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.model_status_label = ttk.Label(model_frame, text=f"Model: {os.path.basename(self.current_model_path) if self.model_loaded else 'None loaded'}")
        self.model_status_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        attendance_frame = ttk.LabelFrame(controls_frame, text="3. Attendance", padding="10")
        attendance_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        attendance_frame.columnconfigure(0, weight=1)

        self.btn_select_video = ttk.Button(attendance_frame, text="Select Video File (Optional)", command=self.select_video_file)
        self.btn_select_video.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.video_file_label = ttk.Label(attendance_frame, text="Input: Webcam (Default)")
        self.video_file_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.btn_start_cam = ttk.Button(attendance_frame, text="Start Attendance Processing", command=self.start_attendance_feed, state=tk.DISABLED)
        self.btn_start_cam.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.btn_stop_cam = ttk.Button(attendance_frame, text="Stop Attendance Processing", command=self.stop_attendance_feed, state=tk.DISABLED)
        self.btn_stop_cam.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.btn_export_excel = ttk.Button(attendance_frame, text="Export Attendance to Excel", command=self.export_to_excel, state=tk.DISABLED)
        self.btn_export_excel.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        self.attendance_status_label = ttk.Label(attendance_frame, text="Status: Idle")
        self.attendance_status_label.grid(row=5, column=0, padx=5, pady=2, sticky="w")
        
        self.detected_names_listbox_label = ttk.Label(attendance_frame, text="Detected This Session:")
        self.detected_names_listbox_label.grid(row=6, column=0, padx=5, pady=(10,2), sticky="w")
        self.detected_names_listbox = tk.Listbox(attendance_frame, height=5)
        self.detected_names_listbox.grid(row=7, column=0, padx=5, pady=2, sticky="ew")

        self.progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=250, mode="determinate")
        self.progress_bar.grid(row=3, column=0, padx=5, pady=10, sticky=(tk.W, tk.E))
        
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed / Video Processing", padding="10")
        video_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.video_panel = ttk.Label(video_frame, text="Video feed or processing status will appear here.", anchor="center")
        self.video_panel.pack(expand=True, fill=tk.BOTH)

        self.try_load_default_model()
        self.update_button_states()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update_gui_from_thread(self, func, *args, **kwargs):
        self.window.after(0, lambda: func(*args, **kwargs))

    def select_dataset_directory(self):
        path = filedialog.askdirectory(title="Select Parent Directory of Named Image Folders")
        if path:
            self.dataset_path = path
            self.dataset_label.config(text=f"Dataset: {os.path.basename(path)}")
            self.train_status_label.config(text="Status: Dataset selected. Ready to train.")
        else:
            self.dataset_label.config(text="Dataset: Not selected")
            self.train_status_label.config(text="Status: Idle")
        self.update_button_states()

    def select_video_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File for Attendance",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.selected_video_file_path = filepath
            self.video_file_label.config(text=f"Input: {os.path.basename(filepath)}")
        else:
            self.selected_video_file_path = None
            self.video_file_label.config(text="Input: Webcam (Default)")
        self.update_button_states()

    def train_model_thread(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select a dataset directory first.")
            return
        
        self.btn_train_model.config(state=tk.DISABLED)
        self.btn_select_dataset.config(state=tk.DISABLED)
        self.btn_load_model.config(state=tk.DISABLED)
        self.btn_start_cam.config(state=tk.DISABLED)
        self.btn_select_video.config(state=tk.DISABLED)

        self.train_status_label.config(text="Status: Training in progress...")
        self.progress_bar["value"] = 0
        
        thread = threading.Thread(target=self._train_model, args=(self.dataset_path,))
        thread.daemon = True
        thread.start()

    def _train_model(self, dataset_path):
        known_encodings = []
        known_names = []
        
        person_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        total_folders = len(person_folders)
        if total_folders == 0:
            self.update_gui_from_thread(messagebox.showerror, "Training Error", "No person subdirectories found in the dataset path.")
            self.update_gui_from_thread(self.train_status_label.config, text="Status: Error - No subdirectories.")
            self.update_gui_from_thread(self.progress_bar.config, value=0)
            self.update_gui_from_thread(self.update_button_states)
            return

        folders_processed = 0

        for person_name in person_folders:
            person_folder_path = os.path.join(dataset_path, person_name)
            image_files = [f for f in os.listdir(person_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            images_processed_for_person = 0
            for image_file in image_files:
                image_path = os.path.join(person_folder_path, image_file)
                image_np_array = None
                try:
                    self.update_gui_from_thread(self.train_status_label.config, text=f"Processing: {person_name}/{image_file}")
                    
                    print(f"DEBUG: Attempting to load {image_path} with OpenCV.")
                    cv2_image = cv2.imread(image_path)

                    if cv2_image is None:
                        print(f"Error: OpenCV (cv2.imread) could not load image {image_path}. Skipping.")
                        
                        
                        print(f"DEBUG: Falling back to Pillow for {image_path}.")
                        pil_img = PIL.Image.open(image_path)
                        print(f"DEBUG: Opened {image_path} with Pillow. Mode: {pil_img.mode}")
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        image_np_array = np.array(pil_img)
                        
                        if image_np_array.dtype != np.uint8:
                            image_np_array = image_np_array.astype(np.uint8)

                    else:
                        print(f"DEBUG: Successfully loaded {image_path} with OpenCV. Shape: {cv2_image.shape}, dtype: {cv2_image.dtype}")
                        
                        image_np_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                        print(f"DEBUG: Converted OpenCV image to RGB. Shape: {image_np_array.shape}, dtype: {image_np_array.dtype}")

                    
                    if image_np_array is None:
                        print(f"CRITICAL: image_np_array is None for {image_path} after all loading attempts. Skipping.")
                        continue

                    print(f"DEBUG: Before face_encodings call: path={image_path}, dtype={image_np_array.dtype}, shape={image_np_array.shape}, min={np.min(image_np_array)}, max={np.max(image_np_array)}")
                    
                    
                    
                    if not ((image_np_array.ndim == 3 and image_np_array.shape[2] == 3 and image_np_array.dtype == np.uint8) or \
                            (image_np_array.ndim == 2 and image_np_array.dtype == np.uint8)):
                        
                        
                        if image_np_array.dtype != np.uint8:
                            print(f"Warning: dtype is {image_np_array.dtype} before final check. Forcing to np.uint8 for {image_path}.")
                            image_np_array = image_np_array.astype(np.uint8)
                            print(f"DEBUG: After forced astype(np.uint8): dtype={image_np_array.dtype}, min={np.min(image_np_array)}, max={np.max(image_np_array)}")

                        if not ((image_np_array.ndim == 3 and image_np_array.shape[2] == 3 and image_np_array.dtype == np.uint8) or \
                                (image_np_array.ndim == 2 and image_np_array.dtype == np.uint8)):
                            print(f"CRITICAL ERROR: Image {image_path} is NOT in expected format for dlib after all conversions. dtype={image_np_array.dtype}, shape={image_np_array.shape}. Skipping.")
                            continue
                    
                    
                    if not image_np_array.flags['C_CONTIGUOUS']:
                        print(f"DEBUG: Image {image_path} is not C-contiguous. Making it contiguous.")
                        image_np_array = np.ascontiguousarray(image_np_array, dtype=np.uint8)
                        print(f"DEBUG: After np.ascontiguousarray: flags={image_np_array.flags}")

                    face_encodings_in_image = face_recognition.face_encodings(image_np_array)
                    
                    if face_encodings_in_image:
                        known_encodings.append(face_encodings_in_image[0])
                        known_names.append(person_name)
                        images_processed_for_person += 1
                    else:
                        print(f"Warning: No faces found in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            
            if images_processed_for_person == 0:
                 print(f"Warning: No usable images found for {person_name}")

            folders_processed += 1
            progress = (folders_processed / total_folders) * 100
            self.update_gui_from_thread(self.progress_bar.config, value=progress)

        if not known_encodings:
            self.update_gui_from_thread(messagebox.showerror, "Training Failed", "No faces could be encoded from the dataset. Check images and subdirectories.")
            self.update_gui_from_thread(self.train_status_label.config, text="Status: Training failed.")
        else:
            self.known_face_encodings = known_encodings
            self.known_face_names = known_names
            self.model_loaded = True
            
            save_path = filedialog.asksaveasfilename(
                initialfile=self.default_model_filename,
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl")]
            )
            if save_path:
                self.current_model_path = save_path
                with open(self.current_model_path, "wb") as f:
                    pickle.dump((self.known_face_encodings, self.known_face_names), f)
                self.update_gui_from_thread(self.train_status_label.config, text=f"Status: Training complete! Model saved to {os.path.basename(self.current_model_path)}")
                self.update_gui_from_thread(self.model_status_label.config, text=f"Model: {os.path.basename(self.current_model_path)}")
                self.update_gui_from_thread(messagebox.showinfo, "Training Complete", f"Model trained and saved to {self.current_model_path}")
            else:
                 self.update_gui_from_thread(self.train_status_label.config, text="Status: Training complete, but model not saved (user cancelled).")
                 self.update_gui_from_thread(messagebox.showwarning, "Model Not Saved", "Training data processed, but you chose not to save the model.")

        self.update_gui_from_thread(self.progress_bar.config, value=100 if known_encodings else 0)
        self.update_gui_from_thread(self.update_button_states)

    def try_load_default_model(self):
        if os.path.exists(self.default_model_filename):
            self.load_model_from_file(self.default_model_filename, silent=True)
        else:
            self.model_status_label.config(text="Model: None loaded (default not found)")
        self.update_button_states()

    def load_model_from_file_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Select Trained Model File",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")]
        )
        if filepath:
            self.load_model_from_file(filepath)

    def load_model_from_file(self, filepath, silent=False):
        try:
            with open(filepath, "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            self.model_loaded = True
            self.current_model_path = filepath
            self.model_status_label.config(text=f"Model: {os.path.basename(self.current_model_path)}")
            if not silent:
                messagebox.showinfo("Model Loaded", f"Successfully loaded model from {os.path.basename(filepath)}")
            self.attendance_status_label.config(text="Status: Model loaded. Ready for attendance.")
        except FileNotFoundError:
            if not silent: messagebox.showerror("Error", f"Model file not found: {filepath}")
            self.model_status_label.config(text="Model: Load failed (not found).")
            self.model_loaded = False
        except Exception as e:
            if not silent: messagebox.showerror("Error", f"Failed to load model: {e}")
            self.model_status_label.config(text="Model: Load failed (error).")
            self.model_loaded = False
        self.update_button_states()

    def start_attendance_feed(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "No model loaded. Please load or train a model first.")
            return

        source_to_use = self.default_video_source
        self.processing_video_file_flag = False
        status_message = "Status: Webcam feed running..."

        if self.selected_video_file_path:
            source_to_use = self.selected_video_file_path
            self.processing_video_file_flag = True
            status_message = f"Status: Processing video '{os.path.basename(self.selected_video_file_path)}'..."
            print(f"Attempting to open video file: {source_to_use}")
        else:
            print(f"Attempting to open webcam: {source_to_use}")

        self.vid = cv2.VideoCapture(source_to_use)
        if not self.vid.isOpened():
            error_msg = f"Cannot open video source: {source_to_use}. "
            if self.processing_video_file_flag:
                error_msg += "Check video file path and format."
            else:
                error_msg += "Check if webcam is connected and not in use."
            messagebox.showerror("Video Error", error_msg)
            self.vid = None
            self.processing_video_file_flag = False
            self.update_button_states()
            return

        self.video_running = True
        self.attendance_status_label.config(text=status_message)
        
        self.detected_persons_session.clear()
        self.detected_names_listbox.delete(0, tk.END)
        
        self.update_button_states()
        self.update_video_feed()

    def update_video_feed(self):
        if not self.video_running or self.vid is None or not self.vid.isOpened():
            if self.processing_video_file_flag and self.vid is None and self.video_running:
                 self.update_gui_from_thread(self.attendance_status_label.config, text=f"Status: Video processing finished. {len(self.detected_persons_session)} person(s) detected.")
                 self.update_gui_from_thread(self.stop_attendance_feed)
            return

        ret, frame = self.vid.read()
        
        if not ret:
            if self.processing_video_file_flag:
                self.update_gui_from_thread(self.attendance_status_label.config, text=f"Status: Video processing finished. {len(self.detected_persons_session)} person(s) detected.")
                self.update_gui_from_thread(self.stop_attendance_feed)
            else:
                if self.video_running:
                    self.update_gui_from_thread(messagebox.showwarning, "Webcam Error", "Webcam feed lost or ended.")
                    self.update_gui_from_thread(self.stop_attendance_feed)
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.face_match_tolerance)
            name = "Unknown"
            color = (0, 0, 255)

            if not self.known_face_encodings:
                print("Warning: known_face_encodings is empty. Cannot compare faces.")
            elif True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        color = (0, 255, 0)
                        if name not in self.detected_persons_session:
                            self.detected_persons_session.add(name)
                            self.detected_names_listbox.insert(tk.END, name)
                           
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(cv2image)
        
        panel_width = self.video_panel.winfo_width()
        panel_height = self.video_panel.winfo_height()
        
        if panel_width > 1 and panel_height > 1:
            img_width, img_height = img.size
            if img_width > 0 and img_height > 0:
                scale = min(panel_width / img_width, panel_height / img_height)
                if scale > 0 :
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    if new_width > 0 and new_height > 0:
                        img = img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)

        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.video_panel.imgtk = imgtk
        self.video_panel.config(image=imgtk)

        if self.video_running:
            delay = 15
            if self.processing_video_file_flag:
                pass
            self.window.after(delay, self.update_video_feed)

    def stop_attendance_feed(self):
        self.video_running = False
        if self.vid:
            self.vid.release()
            self.vid = None
        
        self.video_panel.config(image='')
        self.video_panel.config(text="Video feed stopped. Select an option.")

        if self.processing_video_file_flag:
             self.attendance_status_label.config(text=f"Status: Video processing stopped. {len(self.detected_persons_session)} person(s) detected.")
        else:
             self.attendance_status_label.config(text=f"Status: Webcam feed stopped. {len(self.detected_persons_session)} person(s) detected.")
        
        self.processing_video_file_flag = False

        self.update_button_states()

    def export_to_excel(self):
        if not self.detected_persons_session:
            messagebox.showinfo("Export", "No persons detected in this session to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile="attendance_report.xlsx",
            title="Save Attendance Report As"
        )
        if filepath:
            try:
                df = pd.DataFrame(list(self.detected_persons_session), columns=["Detected Persons"])
                df.to_excel(filepath, index=False)
                messagebox.showinfo("Export Successful", f"Attendance report saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save Excel file: {e}")
        else:
            messagebox.showinfo("Export Cancelled", "Excel export was cancelled.")

    def update_button_states(self):
        if self.video_running:
            self.btn_select_dataset.config(state=tk.DISABLED)
            self.btn_train_model.config(state=tk.DISABLED)
            self.btn_load_model.config(state=tk.DISABLED)
            self.btn_select_video.config(state=tk.DISABLED)
            self.btn_start_cam.config(state=tk.DISABLED)
            self.btn_stop_cam.config(state=tk.NORMAL)
            self.btn_export_excel.config(state=tk.NORMAL if self.detected_persons_session else tk.DISABLED)
        else:
            self.btn_select_dataset.config(state=tk.NORMAL)
            self.btn_train_model.config(state=tk.NORMAL if self.dataset_path else tk.DISABLED)
            self.btn_load_model.config(state=tk.NORMAL)
            self.btn_select_video.config(state=tk.NORMAL)
            self.btn_start_cam.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)
            self.btn_stop_cam.config(state=tk.DISABLED)
            self.btn_export_excel.config(state=tk.NORMAL if self.detected_persons_session else tk.DISABLED)

    def on_closing(self):
        if self.video_running:
            self.stop_attendance_feed()
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.window.destroy()

if __name__ == "__main__":
    FaceAttendanceApp("Face Recognition Attendance System")