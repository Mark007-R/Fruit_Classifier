import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class ModernFruitClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fruit Classifier")
        self.root.geometry("1600x900")
        self.root.configure(bg='#0f0f1e')

        # Model & data variables
        self.svm_model = None
        self.dt_model = None
        self.knn_model = None
        self.scaler = StandardScaler()
        self.class_names = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.current_image_path = None

        # Color scheme
        self.bg_primary = '#0f0f1e'
        self.bg_secondary = '#1a1a2e'
        self.bg_card = '#16213e'
        self.accent_purple = '#8b5cf6'
        self.accent_blue = '#3b82f6'
        self.accent_green = '#10b981'
        self.accent_orange = '#f59e0b'
        self.text_primary = '#f8fafc'
        self.text_secondary = '#94a3b8'

        self.setup_styles()
        self.create_modern_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("Modern.Horizontal.TProgressbar",
                       background=self.accent_purple,
                       troughcolor=self.bg_secondary,
                       borderwidth=0,
                       thickness=4)

    def create_modern_ui(self):
        # Header - compact
        header = tk.Frame(self.root, bg=self.bg_primary)
        header.pack(fill='x', padx=25, pady=(15, 10))
        
        tk.Label(header, text="🍎 AI Fruit Classifier", 
                font=('Arial', 24, 'bold'), bg=self.bg_primary, 
                fg=self.text_primary).pack(side='left')
        
        tk.Label(header, text="Multi-Model ML Classification", 
                font=('Arial', 10), bg=self.bg_primary, 
                fg=self.text_secondary).pack(side='left', padx=(15, 0))

        # Main content in 3 columns
        main_frame = tk.Frame(self.root, bg=self.bg_primary)
        main_frame.pack(fill='both', expand=True, padx=25, pady=(0, 15))
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Left Column - Training
        left_col = tk.Frame(main_frame, bg=self.bg_primary)
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.create_training_section(left_col)

        # Middle Column - Image Upload
        middle_col = tk.Frame(main_frame, bg=self.bg_primary)
        middle_col.grid(row=0, column=1, sticky='nsew', padx=(0, 10))
        self.create_upload_section(middle_col)

        # Right Column - Results
        right_col = tk.Frame(main_frame, bg=self.bg_primary)
        right_col.grid(row=0, column=2, sticky='nsew')
        self.create_results_section(right_col)

    def create_compact_card(self, parent, title, icon=""):
        card = tk.Frame(parent, bg=self.bg_card, highlightbackground='#2d3748', 
                       highlightthickness=1)
        card.pack(fill='both', expand=True, pady=(0, 10))
        
        header = tk.Frame(card, bg=self.bg_card)
        header.pack(fill='x', padx=15, pady=(10, 8))
        
        tk.Label(header, text=f"{icon} {title}", font=('Arial', 12, 'bold'),
                bg=self.bg_card, fg=self.text_primary).pack(anchor='w')
        
        content_frame = tk.Frame(card, bg=self.bg_card)
        content_frame.pack(fill='both', expand=True, padx=15, pady=(0, 10))
        
        return content_frame

    def create_button(self, parent, text, command, bg_color):
        btn = tk.Button(parent, text=text, command=command,
                       font=('Arial', 9, 'bold'), bg=bg_color, fg='white',
                       relief='flat', padx=15, pady=8, cursor='hand2',
                       activebackground=bg_color, activeforeground='white',
                       borderwidth=0)
        return btn

    def create_training_section(self, parent):
        # Dataset Selection
        card1 = self.create_compact_card(parent, "Dataset", "📂")
        
        self.create_button(card1, "📁 Browse Folder", 
                          self.select_dataset, self.accent_blue).pack(fill='x', pady=(0, 8))
        
        self.dataset_label = tk.Label(card1, text="No dataset", 
                                      font=('Arial', 9), bg=self.bg_card,
                                      fg=self.text_secondary, wraplength=280)
        self.dataset_label.pack(fill='x')

        # Training
        card2 = self.create_compact_card(parent, "Training", "🚀")
        
        self.btn_train = self.create_button(card2, "⚡ Train Models",
                                            self.train_models, self.accent_purple)
        self.btn_train.pack(fill='x', pady=(0, 8))
        self.btn_train.config(state='disabled')
        
        self.progress = ttk.Progressbar(card2, mode='indeterminate',
                                       style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill='x', pady=(0, 8))
        
        # Compact stats
        stats = tk.Frame(card2, bg=self.bg_secondary)
        stats.pack(fill='x')
        
        self.stats_labels = []
        for model, color in [("SVM", self.accent_purple), 
                            ("D-Tree", self.accent_blue),
                            ("KNN", self.accent_green)]:
            row = tk.Frame(stats, bg=self.bg_secondary)
            row.pack(fill='x', pady=2, padx=8)
            
            tk.Label(row, text=model, font=('Arial', 8, 'bold'),
                    bg=self.bg_secondary, fg=color, width=8).pack(side='left')
            
            label = tk.Label(row, text="—", font=('Arial', 8),
                           bg=self.bg_secondary, fg=self.text_secondary)
            label.pack(side='right')
            self.stats_labels.append(label)

        # Model Management
        card3 = self.create_compact_card(parent, "Models", "💾")
        
        btn_frame = tk.Frame(card3, bg=self.bg_card)
        btn_frame.pack(fill='x')
        
        self.create_button(btn_frame, "💾 Save", 
                          self.save_models, self.accent_green).pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.create_button(btn_frame, "📂 Load", 
                          self.load_models, self.accent_orange).pack(side='left', expand=True, fill='x')

    def create_upload_section(self, parent):
        card = self.create_compact_card(parent, "Image Upload", "🖼️")
        
        # Image display
        self.image_container = tk.Frame(card, bg=self.bg_secondary)
        self.image_container.pack(fill='both', expand=True, pady=(0, 10))
        
        self.image_label = tk.Label(self.image_container, 
                                    text="No Image\n\nUpload to classify",
                                    font=('Arial', 11), bg=self.bg_secondary,
                                    fg=self.text_secondary)
        self.image_label.pack(expand=True)
        
        # Buttons
        btn_frame = tk.Frame(card, bg=self.bg_card)
        btn_frame.pack(fill='x')
        
        self.create_button(btn_frame, "📤 Upload", 
                          self.upload_image, self.accent_blue).pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.btn_classify = self.create_button(btn_frame, "🔍 Classify",
                                              self.classify_image, self.accent_purple)
        self.btn_classify.pack(side='left', expand=True, fill='x')
        self.btn_classify.config(state='disabled')

    def create_results_section(self, parent):
        card = self.create_compact_card(parent, "Results", "🎯")
        
        self.results_text = tk.Text(card, font=('Consolas', 10),
                                   bg=self.bg_secondary, fg=self.text_primary,
                                   padx=12, pady=12, wrap='word', relief='flat',
                                   borderwidth=0, insertbackground=self.text_primary)
        self.results_text.pack(fill='both', expand=True)
        
        self.results_text.insert('1.0', 
            "Awaiting classification...\n\n"
            "1. Select dataset\n"
            "2. Train models\n"
            "3. Upload image\n"
            "4. Click classify")
        self.results_text.config(state='disabled')

    # --------- Functional Methods ----------

    def extract_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to load image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (128, 128))

            mean_r, mean_g, mean_b = np.mean(img_resized[:, :, 0]), np.mean(img_resized[:, :, 1]), np.mean(img_resized[:, :, 2])
            std_r, std_g, std_b = np.std(img_resized[:, :, 0]), np.std(img_resized[:, :, 1]), np.std(img_resized[:, :, 2])

            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist /= hist.sum()

            texture_mean = np.mean(hist)
            texture_std = np.std(hist)
            texture_energy = np.sum(hist ** 2)
            texture_entropy = -np.sum(hist * np.log2(hist + 1e-10))

            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (128 * 128)

            return np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b,
                           texture_mean, texture_std, texture_energy, texture_entropy, edge_density])
        except Exception as e:
            print(f"[Error] Feature extraction: {e}")
            return None

    def load_dataset(self, dataset_path):
        X, y, class_names = [], [], []
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in class_names:
                class_names.append(class_name)
            class_idx = class_names.index(class_name)
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                features = self.extract_features(os.path.join(class_path, img_name))
                if features is not None:
                    X.append(features)
                    y.append(class_idx)
        return np.array(X), np.array(y), class_names

    def select_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path = folder
            try:
                classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
                self.dataset_label.config(text=f"✓ {len(classes)} classes", fg=self.accent_green)
                self.btn_train.config(state='normal')
            except:
                self.dataset_label.config(text="✓ Selected", fg=self.accent_green)
                self.btn_train.config(state='normal')

    def train_models(self):
        if not hasattr(self, 'dataset_path'):
            messagebox.showerror("Error", "No dataset selected.")
            return
        
        self.progress.start()
        self.btn_train.config(state='disabled')
        self.root.update()

        try:
            X, y, self.class_names = self.load_dataset(self.dataset_path)
            if len(X) == 0:
                raise Exception("No valid data found.")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

            self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.svm_model.fit(self.X_train, self.y_train)
            svm_acc = accuracy_score(self.y_test, self.svm_model.predict(self.X_test))

            self.dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
            self.dt_model.fit(self.X_train, self.y_train)
            dt_acc = accuracy_score(self.y_test, self.dt_model.predict(self.X_test))

            self.knn_model = KNeighborsClassifier(n_neighbors=5)
            self.knn_model.fit(self.X_train, self.y_train)
            knn_acc = accuracy_score(self.y_test, self.knn_model.predict(self.X_test))

            self.stats_labels[0].config(text=f"{svm_acc*100:.1f}%", fg=self.accent_green)
            self.stats_labels[1].config(text=f"{dt_acc*100:.1f}%", fg=self.accent_green)
            self.stats_labels[2].config(text=f"{knn_acc*100:.1f}%", fg=self.accent_green)

            self.progress.stop()
            self.btn_train.config(state='normal')
            self.btn_classify.config(state='normal')
            messagebox.showinfo("Success", "Models trained!")
            
        except Exception as e:
            self.progress.stop()
            self.btn_train.config(state='normal')
            messagebox.showerror("Error", str(e))

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            try:
                self.current_image_path = file_path
                img = Image.open(file_path)
                
                # Calculate dimensions to fit container
                container_width = 400
                container_height = 580
                img.thumbnail((container_width, container_height))
                
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                if self.svm_model is not None:
                    self.btn_classify.config(state='normal')
                
                self.results_text.config(state='normal')
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert('1.0', "Image loaded!\nReady to classify.")
                self.results_text.config(state='disabled')
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def classify_image(self):
        if not self.current_image_path:
            messagebox.showerror("Error", "Upload an image first.")
            return
            
        if not self.svm_model:
            messagebox.showerror("Error", "Train or load models first.")
            return
        
        try:
            features = self.extract_features(self.current_image_path)
            if features is None:
                messagebox.showerror("Error", "Feature extraction failed.")
                return
            
            scaled = self.scaler.transform([features])
            
            svm_pred = self.svm_model.predict(scaled)[0]
            svm_proba = self.svm_model.predict_proba(scaled)[0]
            
            dt_pred = self.dt_model.predict(scaled)[0]
            knn_pred = self.knn_model.predict(scaled)[0]

            votes = [svm_pred, dt_pred, knn_pred]
            final_pred = max(set(votes), key=votes.count)

            result = f"""╔═══════════════════════════════╗
║    CLASSIFICATION RESULTS     ║
╚═══════════════════════════════╝

🤖 SVM
  → {self.class_names[svm_pred]}
  → {max(svm_proba)*100:.1f}% confidence

🌳 Decision Tree
  → {self.class_names[dt_pred]}

📊 K-NN
  → {self.class_names[knn_pred]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 FINAL PREDICTION

  ► {self.class_names[final_pred].upper()} ◄

  ({votes.count(final_pred)}/3 models agree)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result)
            self.results_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_models(self):
        if not self.svm_model:
            messagebox.showerror("Error", "Train models first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")])
        if path:
            try:
                with open(path, 'wb') as f:
                    pickle.dump({
                        'svm': self.svm_model,
                        'dt': self.dt_model,
                        'knn': self.knn_model,
                        'scaler': self.scaler,
                        'class_names': self.class_names
                    }, f)
                messagebox.showinfo("Success", "Models saved!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_models(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if path:
            try:
                with open(path, 'rb') as f:
                    models = pickle.load(f)
                    self.svm_model = models['svm']
                    self.dt_model = models['dt']
                    self.knn_model = models['knn']
                    self.scaler = models['scaler']
                    self.class_names = models['class_names']
                
                for label in self.stats_labels:
                    label.config(text="Loaded ✓", fg=self.accent_green)
                
                self.btn_classify.config(state='normal')
                messagebox.showinfo("Success", "Models loaded!")
            except Exception as e:
                messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = ModernFruitClassifier(root)
    root.mainloop()


if __name__ == "__main__":
    main()