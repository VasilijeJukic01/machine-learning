import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

max_values = []
min_values = []

with open('SaYoPillow.csv', 'r') as file:
    category = file.readline().strip().split(',')

    temp_max = [float('-inf')] * len(category)
    temp_min = [float('inf')] * len(category)

    for red in file:
        vals = list(map(float, red.strip().split(',')))
        for i in range(len(vals)):
            if vals[i] > temp_max[i]:
                temp_max[i] = vals[i]
            if vals[i] < temp_min[i]:
                temp_min[i] = vals[i]
    max_values.extend(temp_max)
    min_values.extend(temp_min)

with open('mlp_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict():
    values = [[]]
    for entry in entries:
        value = entry.get()
        if not value:
            messagebox.showerror('Error', 'You must enter all values.')
            return
        values[0].append(float(value))

    prediction = model.predict(values)
    predicted_class = np.argmax(prediction)

    result_label.config(text=f"Your stress level is: {predicted_class}")

window = tk.Tk()
window.title("Stress Level")
window.geometry("350x330")
window.resizable(False, False)

labels = ["Snoring rate", "Respiration rate", "Body temperature", "Limb movement", "Blood oxygen",
          "Eye movement", "Sleeping hours", "Heart rate"]
entries = []
for i, label_text in enumerate(labels):
    label = tk.Label(window, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(window)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)
    value_info = tk.Label(window, text=f"[{min_values[i]}, {max_values[i]}]")
    value_info.grid(row=i, column=2, padx=10, pady=5)

predict_button = tk.Button(window, text="Check Stress", command=predict)
predict_button.grid(row=len(labels), column=1, columnspan=1, padx=10, pady=5)

result_label = tk.Label(window, text="")
result_label.grid(row=len(labels)+1, column=1, columnspan=1, padx=10, pady=5)

window.mainloop()
