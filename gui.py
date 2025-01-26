import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2 as cv
from src import pre_analyze, wordcloud_gen

class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WorkCloudProjectAO")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2C3E50")  

        style = ttk.Style()
        style.configure("TButton",
                        font=("Helvetica", 14),
                        padding=10,
                        relief="flat",
                        anchor="center",
                        width=20)
        style.configure("TLabel", font=("Helvetica", 12), background="#2C3E50", fg="white")
        style.configure("TCombobox", font=("Helvetica", 12), padding=5)
        style.configure("TText", font=("Helvetica", 10), padding=5)
        style.map("TButton",
                  background=[("active", "#3498DB"),
                              ("!active", "#2C3E50")],  
                  relief=[("active", "flat")])

        self.left_frame = tk.Frame(self.root, bg="#2C3E50", padx=20, pady=20)
        self.left_frame.pack(side="left", fill="y")

        self.text_input_label = tk.Label(self.left_frame, text="Text Input:", bg="#2C3E50", fg="white", font=("Helvetica", 12))
        self.text_input_label.pack(pady=5)

        self.text_input = tk.Entry(self.left_frame, font=("Helvetica", 12), width=20, relief="flat")
        self.text_input.pack(pady=10)

        self.choice_var = tk.StringVar(value="Watershed")
        self.Watershed_radiobutton = tk.Radiobutton(self.left_frame, text="Watershed", variable=self.choice_var, value="Watershed", bg="#2C3E50", fg="white", font=("Helvetica", 12), command=self.update_chosen_option)
        self.EdgeDetection_radiobutton = tk.Radiobutton(self.left_frame, text="EdgeDetection", variable=self.choice_var, value="EdgeDetection", bg="#2C3E50", fg="white", font=("Helvetica", 12), command=self.update_chosen_option)
        
        self.chosen_option = tk.Label(self.left_frame, text="", bg="#2C3E50", font=("Helvetica", 12), fg="white")
        self.Watershed_radiobutton.pack(pady=10)
        self.EdgeDetection_radiobutton.pack(pady=10)
        self.chosen_option.pack(pady=10)

        self.file_label = tk.Label(self.left_frame, text="NO FILE", bg="#2C3E50", font=("Helvetica", 14), fg="white")
        self.file_label.pack(pady=10)

        self.browse_button = ttk.Button(self.left_frame, text="Select File", command=self.browse_file)
        self.browse_button.pack(pady=20)

        self.invert_before_var = tk.BooleanVar(value=False)
        self.invert_after_var = tk.BooleanVar(value=False)

        self.invert_before_checkbox = tk.Checkbutton(self.left_frame, text="Invert Before (Watershed Only)", variable=self.invert_before_var, bg="#2C3E50", fg="white", font=("Helvetica", 12), command=self.update_invert_options)
        self.invert_after_checkbox = tk.Checkbutton(self.left_frame, text="Invert After (Both)", variable=self.invert_after_var, bg="#2C3E50", fg="white", font=("Helvetica", 12), command=self.update_invert_options)

        self.invert_before_checkbox.pack(pady=10)
        self.invert_after_checkbox.pack(pady=10)

        self.invert_options_label = tk.Label(self.left_frame, text="", bg="#2C3E50", font=("Helvetica", 12), fg="white")
        self.invert_options_label.pack(pady=10)

        self.preview_label = tk.Label(self.left_frame, text="Image Preview:", bg="#2C3E50", fg="white", font=("Helvetica", 12))
        self.preview_label.pack(pady=5)

        self.image_label = tk.Label(self.left_frame, bg="#2C3E50", width=200, height=300, relief="solid")  # Increased height
        self.image_label.pack(pady=10)

        self.right_frame = tk.Frame(self.root, bg="#2C3E50")
        self.right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

        self.photo_display_label = tk.Label(self.right_frame, bg="#2C3E50")
        self.photo_display_label.pack(fill="both", expand=True)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("JPG File", "*.jpg"), ("PNG File", "*.png")])
        if filepath:
            self.file_label.config(text=filepath)
            self.load_image(filepath)

    def load_image(self, filepath):
        try:
            image = cv.imread(filepath)
            invert_before = self.invert_before_var.get()
            invert_after = self.invert_after_var.get()

            if self.choice_var.get() == "EdgeDetection":
                image = pre_analyze.edge_detection_mask(image)
            else:
                image = pre_analyze.watershed_mask(image, invert_before, invert_after)

            cv.imwrite("mask.png", image)
            words = self.text_input.get().split()

            packer = wordcloud_gen.WordShapePacker("mask.png", words)
            wordcloud_image = packer.visualize("output.png")

            preview_width = 300
            preview_height = 300

            original_height, original_width = image.shape[:2]
            aspect_ratio = original_width / original_height

            if original_width > original_height:
                new_width = preview_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = preview_height
                new_width = int(new_height * aspect_ratio)

            image = cv.resize(image, (new_width, new_height))
            image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

            self.image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image)

            right_display_size = (original_height, original_width)
            image = image.resize(right_display_size)
            self.display_image = ImageTk.PhotoImage(wordcloud_image)

            self.photo_display_label.config(image=self.display_image)

        except Exception as e:
            print(f"Error loading image: {e}")

    def update_chosen_option(self):
        self.chosen_option.config(text=f"Wybrana opcja: {self.choice_var.get()}")

    def update_invert_options(self):
        invert_before = "True" if self.invert_before_var.get() else "False"
        invert_after = "True" if self.invert_after_var.get() else "False"
        self.invert_options_label.config(text=f"Invert Before: {invert_before}, Invert After: {invert_after}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
