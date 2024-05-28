import os.path
import tkinter as tk
from tkinter import Label
from tkinter.filedialog import askopenfilename
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os

from prediction import predictor

window = tk.Tk()
window.title("Number prediction")


def run_window():
    # Create the main window
    window.geometry('640x600')  # Set the window size
    window.resizable(False, False)  # Make the window non-resizable
    window.configure(bg="white")

    frame = tk.Frame(window, bg="white")
    frame.pack(fill='x', side='top')

    label = Label(frame, text='', font=("Helvetica", 11), anchor="w", bg="#EEEEEE", width=57, wraplength=490)
    label.pack(side='left', padx=5, pady=5)  # Added some padding for better spacing

    open_button = tk.Button(
        frame,
        text='Select image',
        font=("Helvetica", 11),
        command=lambda: open_image_file(label)
    )

    open_button.pack(side="right", padx=5, pady=5)

    window.mainloop()


def display_image(file_path, root):
    if hasattr(display_image, 'canvas'):
        display_image.canvas.get_tk_widget().destroy()
    if hasattr(display_image, 'predict_button'):
        display_image.predict_button.destroy()

    image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image_array, (28, 28))
    image = 255 - image  # inverse
    # window.image = image

    # Create a Matplotlib figure and axis
    fig = Figure(figsize=(5, 5))  # in inches
    ax = fig.add_subplot(111)

    # Display the image
    ax.imshow(image)
    # ax.axis('off')  # Hide the axes

    # Integrate the Matplotlib figure with Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    predict_button = tk.Button(
        window,
        text='Predict',
        font=('Helvetica', 11),
        command=lambda: predict_image(image)
    )
    predict_button.pack(side=tk.BOTTOM, anchor=tk.CENTER, padx=20, pady=10)

    # Store the canvas widget to be able to destroy it later
    display_image.canvas = canvas
    display_image.predict_button = predict_button


def predict_image(image: np.array):
    result = predictor.predict(image)
    result_text = result[0] if len(result) == 1 else f'{result[0]} or {result[1]}'
    modal_dialog = tk.Toplevel(window)
    modal_dialog.title('Result')
    modal_dialog.geometry('200x100')
    modal_dialog.grab_set()
    modal_dialog.resizable(False, False)  # Make the window non-resizable
    modal_dialog.configure(bg='white')

    label = tk.Label(modal_dialog, font=('Helvetica', 32), text=result_text, bg="white")
    label.pack(padx=10, pady=10)


def open_image_file(result_shower: Label):
    image_files = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']
    ftypes = [
        ('image files', image_files)
    ]
    file_path = askopenfilename(initialdir="D:\Docs\Python\ML\Mnist-number-detection\\numbers", title="Select file", filetypes=ftypes)
    result_shower.config(text=file_path)
    if file_path and os.path.isfile(file_path):
        display_image(file_path, root=window)


if __name__ == '__main__':
    run_window()
