import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
import matplotlib.pyplot as plt

import os

# automatycznie znajdź ścieżkę do pliku hist_eq względem skryptu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
CPP_EXECUTABLE = os.path.join(SCRIPT_DIR, "hist_eq")


# --- Funkcja do uruchamiania programu C++ ---
def run_cpp_program(image_path):
    try:
        result = subprocess.run([CPP_EXECUTABLE, image_path], capture_output=True, text=True)
        if result.returncode != 0:
            messagebox.showerror("Błąd", f"Program zwrócił błąd:\n{result.stderr}")
        else:
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, result.stdout)
            parse_scalability_results()
    except Exception as e:
        messagebox.showerror("Błąd", str(e))

# --- Funkcja wyboru obrazu ---
def choose_file():
    file_path = filedialog.askopenfilename(
    title="Wybierz obraz",
    filetypes=[
        ("Pliki graficzne", "*.png *.jpg *.jpeg *.bmp"),
        ("PNG", "*.png"),
        ("JPEG", "*.jpg *.jpeg"),
        ("Bitmapy", "*.bmp"),
        ("Wszystkie pliki", "*.*")
    ]
)

    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

# --- Funkcja do parsowania i wyświetlania wyników skalowalności ---
def parse_scalability_results():
    # Lista plików CSV i ich czytelne opisy
    files = {
        "scalability_results.csv": "Skalowalność OpenMP (Grayscale)",
        "scalability_results_color.csv": "Skalowalność OpenMP (Color)",
        "scalability_results_cuda_gray.csv": "Skalowalność CUDA (Grayscale)",
        "scalability_results_cuda_color.csv": "Skalowalność CUDA (Color)"
    }

    plots = []

    for file_name, title in files.items():
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(file_name)
                if df.empty:
                    continue

                fig, ax = plt.subplots()
                ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', linestyle='-', linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel('Czas [ms]')
                ax.grid(True)
                plots.append(fig)
            except Exception as e:
                print(f"Błąd podczas przetwarzania {file_name}: {e}")

    # Wyświetl wszystkie wykresy równocześnie
    if plots:
        plt.show(block=False)
    else:
        messagebox.showinfo("Informacja", "Nie znaleziono żadnych wyników skalowalności.")


# --- GUI ---
root = tk.Tk()
root.title("Interfejs do Equalizacji Histogramu")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

tk.Label(frame, text="Wybierz obraz:").grid(row=0, column=0, sticky='w')
file_entry = tk.Entry(frame, width=50)
file_entry.grid(row=0, column=1, padx=5)
tk.Button(frame, text="Wybierz plik", command=choose_file).grid(row=0, column=2)

run_button = tk.Button(frame, text="Uruchom pomiary", command=lambda: run_cpp_program(file_entry.get()))
run_button.grid(row=1, column=0, columnspan=3, pady=10)

tk.Label(frame, text="Output programu:").grid(row=2, column=0, sticky='w', pady=(10,0))
output_text = tk.Text(frame, width=80, height=20)
output_text.grid(row=3, column=0, columnspan=3, pady=5)

root.mainloop()
