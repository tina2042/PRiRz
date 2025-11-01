import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Automatyczne określenie ścieżki do pliku wykonywalnego ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_EXECUTABLE = os.path.join(SCRIPT_DIR, "hist_eq")  # zmień, jeśli plik ma inną nazwę

# --- Funkcja do uruchamiania programu C++ ---
def run_cpp_program():
    image_path = file_entry.get().strip()
    if not os.path.exists(image_path):
        messagebox.showerror("Błąd", "Nie znaleziono pliku obrazu.")
        return

    if not os.path.exists(CPP_EXECUTABLE):
        messagebox.showerror("Błąd", f"Nie znaleziono pliku wykonywalnego:\n{CPP_EXECUTABLE}")
        return

    try:
        run_button.config(state=tk.DISABLED)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Uruchamianie pomiarów dla:\n{image_path}\n\n")

        result = subprocess.run([CPP_EXECUTABLE, image_path], capture_output=True, text=True)
        output_text.insert(tk.END, result.stdout)

        if result.returncode != 0:
            messagebox.showerror("Błąd wykonania", result.stderr)
        else:
            output_text.insert(tk.END, "\n--- Pomiar zakończony ---\n")
            load_scalability_buttons()
            show_summary_plot(result.stdout)
            update_summary_table(result.stdout)  # ⬅️ NOWE: aktualizacja podsumowania
            update_verification_table(result.stdout)


    except Exception as e:
        messagebox.showerror("Błąd", str(e))
    finally:
        run_button.config(state=tk.NORMAL)


def update_summary_table(output_text_content):
    """
    Wyciąga z wyjścia C++ wszystkie czasy i wyświetla je w tabeli w interfejsie.
    """
    for item in summary_table.get_children():
        summary_table.delete(item)

    # szukamy sekcji i czasów
    pattern = re.compile(r"-+\s*(.*?)\s*-+|[ŚS]redni czas.*?:\s*([\d.]+)\s*ms", re.IGNORECASE)

    current_section = None
    for line in output_text_content.splitlines():
        match = pattern.search(line)
        if match:
            if match.group(1):
                current_section = match.group(1).strip()
            elif match.group(2) and current_section:
                time_ms = float(match.group(2))
                method = "Sekwencyjny" if "Sekwencyjny" in current_section else \
                         "OpenMP" if "OpenMP" in current_section else \
                         "CUDA" if "CUDA" in current_section else "?"
                image_type = "Color" if "Color" in current_section else "Grayscale"
                summary_table.insert("", "end", values=(method, image_type, f"{time_ms:.2f}"))

    image_path = file_entry.get().strip()
    if not os.path.exists(image_path):
        messagebox.showerror("Błąd", "Nie znaleziono pliku obrazu.")
        return

    if not os.path.exists(CPP_EXECUTABLE):
        messagebox.showerror("Błąd", f"Nie znaleziono pliku wykonywalnego:\n{CPP_EXECUTABLE}")
        return

    try:
        run_button.config(state=tk.DISABLED)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Uruchamianie pomiarów dla:\n{image_path}\n\n")

        result = subprocess.run([CPP_EXECUTABLE, image_path], capture_output=True, text=True)
        output_text.insert(tk.END, result.stdout)

        if result.returncode != 0:
            messagebox.showerror("Błąd wykonania", result.stderr)
        else:
            output_text.insert(tk.END, "\n--- Pomiar zakończony ---\n")
            load_scalability_buttons()
            # show_summary_plot(result.stdout)

    except Exception as e:
        messagebox.showerror("Błąd", str(e))
    finally:
        run_button.config(state=tk.NORMAL)

def update_verification_table(output_text_content):
    """
    Wyciąga z wyjścia C++ informacje o różnicach histogramów i pokazuje je w tabeli.
    """
    for item in verification_table.get_children():
        verification_table.delete(item)

    pattern = re.compile(r"Różnica histogramów(?:.*?)?:\s*(\d+)", re.IGNORECASE)
    labeled_pattern = re.compile(
        r"Różnica histogramów(?:\s*(SEQ|OMP|CUDA)?.*?(Grayscale|Color|B|G|R|B\+G\+R)?)?:\s*(\d+)",
        re.IGNORECASE,
    )

    # Słownik etykiet i wartości
    comparisons = []

    for line in output_text_content.splitlines():
        # 1️⃣ Ogólne dopasowania (bez nazw kanałów)
        match = re.search(r"Różnica histogramów(?:.*?)?:\s*(\d+)", line)
        if match:
            diff = int(match.group(1))

            # dopasowanie typu porównania
            if "SEQ vs CUDA" in line:
                comparisons.append(("SEQ vs CUDA (Gray)", diff))
            elif "SEQ vs OMP" in line and "Color" not in line:
                comparisons.append(("SEQ vs OMP (Gray)", diff))
            elif "SEQ vs OMP" in line and "Color" in line and "B+G+R" in line:
                comparisons.append(("SEQ vs OMP (Color)", diff))
            elif "SEQ vs CUDA" in line and "Color" in line and "B+G+R" in line:
                comparisons.append(("SEQ vs CUDA (Color)", diff))
            elif "OMP Color vs CUDA Color" in line:
                comparisons.append(("OMP vs CUDA (Color)", diff))
            elif "Łączna różnica histogramów" in line:
                # Łączna różnica (dla B+G+R)
                if "SEQ vs OMP" in line:
                    comparisons.append(("SEQ vs OMP (Color)", diff))
                elif "SEQ vs CUDA" in line:
                    comparisons.append(("SEQ vs CUDA (Color)", diff))
            elif "Różnica histogramów kanału" not in line:
                # np. Różnica histogramów: 0
                comparisons.append(("SEQ vs OMP (Gray)", diff))

    # Unikalne wyniki (na wypadek powtórek)
    seen = set()
    for label, diff in comparisons:
        if label not in seen:
            color_tag = "ok" if diff == 0 else "warn"
            verification_table.insert("", "end", values=(label, diff), tags=(color_tag,))
            seen.add(label)

    # Kolory
    verification_table.tag_configure("ok", background="#d4edda")  # zielony
    verification_table.tag_configure("warn", background="#fff3cd")  # żółty/pomarańczowy


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


# --- Funkcja wyświetlająca wykresy ---
def show_plot(file_name, title):
    if not os.path.exists(file_name):
        messagebox.showwarning("Brak danych", f"Nie znaleziono pliku: {file_name}")
        return

    df = pd.read_csv(file_name)
    if df.empty:
        messagebox.showinfo("Pusty plik", f"Plik {file_name} jest pusty.")
        return

    fig, ax = plt.subplots()
    ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel('Czas [ms]')
    ax.grid(True)
    plt.show(block=False)


# --- Tworzenie przycisków wykresów po zakończeniu pomiarów ---
def load_scalability_buttons():
    for widget in right_frame.winfo_children():
        widget.destroy()

    tk.Label(right_frame, text="Wykresy skalowalności", font=("Arial", 12, "bold")).pack(pady=5)

    buttons = [
        ("OpenMP (Grayscale)", "scalability_results.csv"),
        ("OpenMP (Color)", "scalability_results_color.csv"),
        ("CUDA (Grayscale)", "scalability_results_cuda_gray.csv"),
        ("CUDA (Color)", "scalability_results_cuda_color.csv"),
    ]

    for label, file in buttons:
        ttk.Button(
            right_frame,
            text=label,
            command=lambda f=file, l=label: show_plot(f, f"Skalowalność {l}")
        ).pack(fill='x', pady=3, padx=5)


# --- Nowa funkcja: generowanie zbiorczego wykresu porównawczego ---
def show_summary_plot(output_text_content):
    """
    Analizuje tekst wyjściowy programu C++ i tworzy porównawczy wykres
    średnich czasów dla SEQ / OMP / CUDA w grayscale i color.
    """
    # Szukamy linii typu:
    # "Sredni czas wykonania (10 runow): 123 ms"
    time_pattern = re.compile(r"-+\s*(.*?)\s*-+|[ŚS]redni czas.*?:\s*([\d.]+)\s*ms", re.IGNORECASE)


    sections = []
    times = []
    current_section = None

    for line in output_text_content.splitlines():
        match = time_pattern.search(line)
        if match:
            if match.group(1):
                # nagłówek sekcji np. "--- 1. Sekwencyjny Proces (Grayscale) ---"
                current_section = match.group(1).strip()
            elif match.group(2) and current_section:
                times.append(float(match.group(2)))
                sections.append(current_section)

    if not times:
        return

    # Wykres słupkowy
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#6A5ACD' if "Sekwencyjny" in s else '#32CD32' if "OpenMP" in s else '#FF6347' for s in sections]
    ax.barh(sections, times, color=colors)
    ax.set_xlabel("Średni czas wykonania [ms]")
    ax.set_title("Porównanie metod equalizacji histogramu")
    plt.tight_layout()
    plt.show(block=False)


# --- Główne okno ---
root = tk.Tk()
root.title("Interfejs do analizy equalizacji histogramu")
root.geometry("1000x600")
root.option_add('*Font', 'Arial 10')

main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Podział na lewą (tekst) i prawą (wykresy) stronę
left_frame = tk.Frame(main_frame)
left_frame.pack(side='left', fill='both', expand=True)

right_frame = tk.Frame(main_frame, relief='groove', bd=2)
right_frame.pack(side='right', fill='y', padx=5, pady=5)

# --- Sekcja wyboru pliku ---
file_frame = tk.Frame(left_frame)
file_frame.pack(fill='x', pady=5)

tk.Label(file_frame, text="Wybierz obraz:", font=("Arial", 11, "bold")).pack(anchor='w')
file_entry = tk.Entry(file_frame, width=60)
file_entry.pack(side='left', padx=5)
ttk.Button(file_frame, text="Przeglądaj...", command=choose_file).pack(side='left', padx=5)

# --- Przycisk uruchomienia ---
run_button = ttk.Button(left_frame, text="Uruchom pomiary", command=run_cpp_program)
run_button.pack(pady=10)

# --- Sekcja podsumowania wyników ---
tk.Label(left_frame, text="Podsumowanie wyników:", font=("Arial", 11, "bold")).pack(anchor='w', pady=(10, 0))

summary_table = ttk.Treeview(left_frame, columns=("method", "type", "time"), show="headings", height=5)
summary_table.pack(fill="x", padx=5, pady=5)

summary_table.heading("method", text="Metoda")
summary_table.heading("type", text="Typ obrazu")
summary_table.heading("time", text="Średni czas [ms]")

summary_table.column("method", width=150, anchor="center")
summary_table.column("type", width=120, anchor="center")
summary_table.column("time", width=120, anchor="center")

# --- Sekcja weryfikacji poprawności ---
tk.Label(left_frame, text="Weryfikacja poprawności wyników:", font=("Arial", 11, "bold")).pack(anchor='w', pady=(10, 0))

verification_table = ttk.Treeview(left_frame, columns=("compare", "diff"), show="headings", height=5)
verification_table.pack(fill="x", padx=5, pady=5)

verification_table.heading("compare", text="Porównanie")
verification_table.heading("diff", text="Różnica histogramów")

verification_table.column("compare", width=200, anchor="center")
verification_table.column("diff", width=120, anchor="center")

# --- Sekcja wyników tekstowych ---
tk.Label(left_frame, text="Wyniki pomiarów:", font=("Arial", 11, "bold")).pack(anchor='w')

text_frame = tk.Frame(left_frame)
text_frame.pack(fill='both', expand=True)

scrollbar = tk.Scrollbar(text_frame)
scrollbar.pack(side='right', fill='y')

output_text = tk.Text(text_frame, wrap='word', yscrollcommand=scrollbar.set)
output_text.pack(fill='both', expand=True)
scrollbar.config(command=output_text.yview)

root.mainloop()
