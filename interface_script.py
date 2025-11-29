import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_EXECUTABLE = os.path.join(SCRIPT_DIR, "hist_eq")  
MPI_EXECUTABLE = "mpirun"  
MPI_PROCS = 4 
MPI_RUNNER_EXECUTABLE = os.path.join(SCRIPT_DIR, "mpi_runner") 

def run_cpp_program():
    reset_interface()
    image_path = file_entry.get().strip()
    if not os.path.exists(image_path):
        messagebox.showerror("Błąd", "Nie znaleziono pliku obrazu.")
        return

    if not os.path.exists(CPP_EXECUTABLE):
        messagebox.showerror("Błąd", f"Nie znaleziono pliku wykonywalnego:\n{CPP_EXECUTABLE}")
        return
    
    selected_mode = measurement_mode.get()
    mode_code = mode_map.get(selected_mode, "ALL")
    try:
        num_bins = int(bin_count.get())
        if num_bins < 1 or num_bins > 256:
            raise ValueError
    except:
        messagebox.showerror("Błąd", "Nieprawidłowa liczba przedziałów (wymagane 1-256).")
        return
    
    commands_to_run = []

    if mode_code == "ALL":
        commands_to_run.append((
            [CPP_EXECUTABLE, image_path, mode_code, str(num_bins)], 
            "Standardowe (SEQ/OMP/CUDA)", 
            CPP_EXECUTABLE
        ))
        commands_to_run.append((
            [MPI_EXECUTABLE, "-np", str(MPI_PROCS), MPI_RUNNER_EXECUTABLE, image_path, "MPI_GRAY", str(num_bins)], 
            f"MPI ({MPI_PROCS} procesów)", 
            MPI_RUNNER_EXECUTABLE
        ))
        commands_to_run.append((
            [MPI_EXECUTABLE, "-np", str(MPI_PROCS), MPI_RUNNER_EXECUTABLE, image_path, "MPI_COLOR", str(num_bins)], 
            f"MPI (Color, {MPI_PROCS} procesów)", 
            MPI_RUNNER_EXECUTABLE
        ))
    elif mode_code == "MPI":
        commands_to_run.append((
            [MPI_EXECUTABLE, "-np", str(MPI_PROCS), MPI_RUNNER_EXECUTABLE, image_path, "MPI_GRAY", str(num_bins)], 
            f"MPI ({MPI_PROCS} procesów)", 
            MPI_RUNNER_EXECUTABLE
        ))
        commands_to_run.append((
            [MPI_EXECUTABLE, "-np", str(MPI_PROCS), MPI_RUNNER_EXECUTABLE, image_path, "MPI_COLOR", str(num_bins)], 
            f"MPI ({MPI_PROCS} procesów)", 
            MPI_RUNNER_EXECUTABLE
        ))
    else:
        commands_to_run.append((
            [CPP_EXECUTABLE, image_path, mode_code, str(num_bins)], 
            selected_mode, 
            CPP_EXECUTABLE
        ))

    try:
        run_button.config(state=tk.DISABLED)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Uruchamianie pomiarów dla:\n{image_path}\n\n")

        full_output = ""
        any_error = False

        for command, description, check_path in commands_to_run:
            
            if not os.path.exists(check_path) and check_path not in ["mpirun"]:
                msg = f"Nie znaleziono pliku wykonywalnego dla {description}:\n{check_path}. Pomijam ten test."
                messagebox.showwarning("Ostrzeżenie/Błąd", msg)
                full_output += f"\n--- {description} - POMINIĘTO (Brak pliku wykonywalnego: {check_path}) ---\n"
                any_error = True
                continue
                
            output_text.insert(tk.END, f"--- Uruchamiam: {description} ---\n")
            output_text.insert(tk.END, f"Komenda: {' '.join(command)}\n")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            full_output += result.stdout
            
            if result.returncode != 0:
                error_message = result.stderr if result.stderr else f"Nieznany błąd wykonania dla {description}."
                full_output += f"\n[BŁĄD WYKONANIA DLA {description.upper()}]:\n{error_message}\n"
                messagebox.showwarning(f"Błąd wykonania ({description})", error_message)
                any_error = True
            
            output_text.insert(tk.END, result.stdout)
            output_text.see(tk.END) 

        if mode_code == "SCALING" or mode_code == "ALL":
            
            csv_gray = "scalability_results_mpi_gray.csv"
            command_mpi_gray = [
                "python3", os.path.join(SCRIPT_DIR, "src/mpi_scaling_runner.py"), 
                image_path, "GRAY", csv_gray, str(MPI_PROCS)
            ]
            
            if os.path.exists(csv_gray): os.remove(csv_gray)
            
            output_text.insert(tk.END, f"\n--- Uruchamianie skryptu SCALING MPI (Gray) ---\n")
            output_text.insert(tk.END, f"Komenda: {' '.join(command_mpi_gray)}\n")
            
            result_mpi_gray = subprocess.run(command_mpi_gray, capture_output=True, text=True)
            full_output += result_mpi_gray.stdout 
            output_text.insert(tk.END, result_mpi_gray.stdout)
            
            if result_mpi_gray.returncode != 0:
                any_error = True
                full_output += f"\n[BŁĄD SKALOWALNOŚCI MPI GRAY]: {result_mpi_gray.stderr}\n"
                messagebox.showwarning("Błąd skalowalności", result_mpi_gray.stderr)

            
            csv_color = "scalability_results_mpi_color.csv"
            command_mpi_color = [
                "python3", os.path.join(SCRIPT_DIR, "src/mpi_scaling_runner.py"), 
                image_path, "COLOR", csv_color, str(MPI_PROCS)
            ]
            
            if os.path.exists(csv_color): os.remove(csv_color)

            output_text.insert(tk.END, f"\n--- Uruchamianie skryptu SCALING MPI (Color) ---\n")
            output_text.insert(tk.END, f"Komenda: {' '.join(command_mpi_color)}\n")
            
            result_mpi_color = subprocess.run(command_mpi_color, capture_output=True, text=True)
            full_output += result_mpi_color.stdout
            output_text.insert(tk.END, result_mpi_color.stdout)
            
            if result_mpi_color.returncode != 0:
                any_error = True
                full_output += f"\n[BŁĄD SKALOWALNOŚCI MPI COLOR]: {result_mpi_color.stderr}\n"
                messagebox.showwarning("Błąd skalowalności", result_mpi_color.stderr)
       
        if not any_error:
            output_text.insert(tk.END, "\n--- Pomiar zakończony pomyślnie ---\n")
        else:
            output_text.insert(tk.END, "\n--- Pomiar zakończony z błędami/ostrzeżeniami ---\n")

        load_scalability_buttons()
        if mode_code not in ["SCALING", "MPI_SCALING"]:
            show_summary_plot(full_output) 

        update_summary_table(full_output)
        update_verification_table(full_output)

    except FileNotFoundError as e:
        messagebox.showerror("Błąd uruchomienia", f"Nie znaleziono narzędzia: {e.filename} (czy {MPI_EXECUTABLE} jest w PATH?)")
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

    pattern = re.compile(r"-+\s*(.*?)\s*-+|[ŚS]redni czas.*?:\s*([\d.]+)\s*ms", re.IGNORECASE)
    mpi_proc_pattern = re.compile(r"MPI.*?(\d+)\s* proces(ow|y)", re.IGNORECASE)

    current_section = None
    for line in output_text_content.splitlines():
        match = pattern.search(line)
        if match:
            if match.group(1):
                current_section = match.group(1).strip()
                if "SKALOWALNOŚĆ" in current_section:
                    continue
            elif match.group(2) and current_section:
                time_ms = float(match.group(2))
                if "Sekwencyjny" in current_section:
                    method = "Sekwencyjny"
                elif "OpenMP" in current_section:
                    method = "OpenMP"
                elif "CUDA" in current_section:
                    method = "CUDA"
                elif "MPI" in current_section:
                    method = "MPI"
                    proc_match = mpi_proc_pattern.search(current_section)
                    
                    if proc_match:
                        num_procs = int(proc_match.group(1))
                        
                        if num_procs != 4:
                            continue  
                        
                        image_type = "Color" if "Color" in current_section else "Grayscale"                       
                        image_type = f"{image_type} (4 proc.)"
                    elif "SKALOWALNOŚĆ" in current_section:
                        continue
                else:
                    method = "?"

                if method != "MPI":
                    image_type = "Color" if "Color" in current_section else "Grayscale"
                summary_table.insert("", "end", values=(method, image_type, f"{time_ms:.2f}"))

def reset_interface():
    output_text.delete("1.0", tk.END)

    for table in (summary_table, verification_table):
        for item in table.get_children():
            table.delete(item)

    for widget in right_frame.winfo_children():
        widget.destroy()

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

    comparisons = []

    for line in output_text_content.splitlines():
        match = re.search(r"Różnica histogramów(?:.*?)?:\s*(\d+)", line)
        if match:
            diff = int(match.group(1))

            if "SEQ vs MPI" in line and "Color" not in line and "kanału" not in line:
                comparisons.append(("SEQ vs MPI (Gray)", diff))
            elif "SEQ Color vs MPI Color" in line:
                comparisons.append(("SEQ vs MPI (Color)", diff))
            elif "SEQ vs CUDA" in line:
                comparisons.append(("SEQ vs CUDA (Gray)", diff))
            elif "SEQ vs OMP" in line and "Color" not in line:
                comparisons.append(("SEQ vs OMP (Gray)", diff))
            elif "SEQ vs OMP" in line and "Color" in line and "B+G+R" in line:
                comparisons.append(("SEQ vs OMP (Color)", diff))
            elif "SEQ vs CUDA" in line and "Color" in line and "B+G+R" in line:
                comparisons.append(("SEQ vs CUDA (Color)", diff))
            elif "OMP Color vs CUDA Color" in line:
                comparisons.append(("OMP vs CUDA (Color)", diff))
            elif "SEQ Color vs CUDA Color" in line:
                comparisons.append(("SEQ vs CUDA (Color)", diff))
            elif "SEQ Color vs OMP Color" in line:
                comparisons.append(("SEQ vs OMP (Color)", diff))
            elif "Łączna różnica histogramów" in line:
                if "SEQ vs OMP" in line:
                    comparisons.append(("SEQ vs OMP (Color)", diff))
                elif "SEQ vs CUDA" in line:
                    comparisons.append(("SEQ vs CUDA (Color)", diff))
            elif "Różnica histogramów kanału" not in line:
                comparisons.append(("SEQ vs OMP (Gray)", diff))

    seen = set()
    for label, diff in comparisons:
        if label not in seen:
            color_tag = "ok" if diff == 0 else "warn"
            verification_table.insert("", "end", values=(label, diff), tags=(color_tag,))
            seen.add(label)

    verification_table.tag_configure("ok", background="#d4edda")  
    verification_table.tag_configure("warn", background="#fff3cd")  

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


def load_scalability_buttons():
    for widget in right_frame.winfo_children():
        widget.destroy()

    tk.Label(right_frame, text="Wykresy skalowalności", font=("Arial", 12, "bold")).pack(pady=5)

    buttons = [
        ("OpenMP (Grayscale)", "scalability_results.csv"),
        ("OpenMP (Color)", "scalability_results_color.csv"),
        ("CUDA (Grayscale)", "scalability_results_cuda_gray.csv"),
        ("CUDA (Color)", "scalability_results_cuda_color.csv"),
        ("MPI (Grayscale)", "scalability_results_mpi_gray.csv"),
        ("MPI (Color)", "scalability_results_mpi_color.csv"), 
    ]

    for label, file in buttons:
        ttk.Button(
            right_frame,
            text=label,
            command=lambda f=file, l=label: show_plot(f, f"Skalowalność {l}")
        ).pack(fill='x', pady=3, padx=5)


def show_summary_plot(output_text_content):
    """
    Analizuje tekst wyjściowy programu C++ i tworzy porównawczy wykres
    średnich czasów dla SEQ / OMP / CUDA w grayscale i color.
    """
    time_pattern = re.compile(r"-+\s*(.*?)\s*-+|[ŚS]redni czas.*?:\s*([\d.]+)\s*ms", re.IGNORECASE)

    current_section = None
    filtered_results = []
    mpi_results = []

    for line in output_text_content.splitlines():
        match = time_pattern.search(line)
        
        if match:
            if match.group(1):
                current_section = match.group(1).strip()
            
            elif match.group(2) and current_section:
                time_ms = float(match.group(2))
                section_name = current_section
                
                if "MPI" in section_name:
                    is_scaling = "SKALOWALNOŚĆ MPI" in section_name
                    proc_match = re.search(r"(\d+)\s* proces(ow|y)", section_name)
                    
                    if proc_match:
                        num_procs = int(proc_match.group(1))
                        
                        if not is_scaling:
                            if "Color" in section_name:
                                label = f"MPI (Color, {num_procs} proc.)"
                            else:
                                label = f"MPI (Grayscale, {num_procs} proc.)"
                            
                            mpi_results.append({'label': label, 'time': time_ms, 'procs': num_procs})
                else:
                    filtered_results.append((section_name, time_ms))

    types_to_add = set()
    
    for result in mpi_results:
        if result['procs'] == 4:
            type_key = "Color" if "Color" in result['label'] else "Grayscale"
            
            if type_key not in types_to_add:
                filtered_results.append((result['label'], result['time']))
                types_to_add.add(type_key)
                   
    final_sections = [s[0] for s in filtered_results]
    final_times = [s[1] for s in filtered_results]

    if not final_times:
        return
    
    sections_to_plot = [s for s in final_sections if "SKALOWALNOŚĆ" not in s]
    times_to_plot = [final_times[i] for i, s in enumerate(final_sections) if "SKALOWALNOŚĆ" not in s]

    def get_color(section):
        if "Sekwencyjny" in section:
            return '#6A5ACD'  
        elif "OpenMP" in section:
            return '#32CD32'  
        elif "CUDA" in section:
            return '#FF6347'  
        elif "MPI" in section:
            return '#00BFFF'  
        return '#808080'  

    colors = [get_color(s) for s in sections_to_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sections_to_plot, times_to_plot, color=colors)
    ax.set_xlabel("Średni czas wykonania [ms]")
    ax.set_title("Porównanie metod equalizacji histogramu (MPI: 4 procesy)")
    plt.tight_layout()
    plt.show(block=False)

root = tk.Tk()
root.title("Interfejs do analizy equalizacji histogramu")
root.geometry("1000x600")
root.option_add('*Font', 'Arial 10')

main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

left_frame = tk.Frame(main_frame)
left_frame.pack(side='left', fill='both', expand=True)

right_frame = tk.Frame(main_frame, relief='groove', bd=2)
right_frame.pack(side='right', fill='y', padx=5, pady=5)

file_frame = tk.Frame(left_frame)
file_frame.pack(fill='x', pady=5)

tk.Label(file_frame, text="Wybierz obraz:", font=("Arial", 11, "bold")).pack(anchor='w')
file_entry = tk.Entry(file_frame, width=60)
file_entry.pack(side='left', padx=5)
ttk.Button(file_frame, text="Przeglądaj...", command=choose_file).pack(side='left', padx=5)

run_button = ttk.Button(left_frame, text="Uruchom pomiary", command=run_cpp_program)
run_button.pack(pady=10)

tk.Label(left_frame, text="Tryb pomiaru:", font=("Arial", 11, "bold")).pack(anchor='w', pady=(10, 0))

measurement_mode = ttk.Combobox(left_frame, state="readonly", width=40)
measurement_mode['values'] = [
    "Wszystkie",
    "Sekwencyjny",
    "OpenMP",
    "CUDA",
    "Sekwencyjny + OpenMP",
    "Sekwencyjny + CUDA",
    "OpenMP + CUDA",
    "MPI",
    "Tylko skalowalność"
]
measurement_mode.current(0)  
measurement_mode.pack(anchor='w', padx=5, pady=5)
mode = measurement_mode.get()

mode_map = {
    "Wszystkie": "ALL",
    "Sekwencyjny": "SEQ",
    "OpenMP": "OMP",
    "CUDA": "CUDA",
    "Sekwencyjny + OpenMP": "SEQ_OMP",
    "Sekwencyjny + CUDA": "SEQ_CUDA",
    "OpenMP + CUDA": "OMP_CUDA",
    "MPI": "MPI",      
    "Tylko skalowalność": "SCALING", 
}

tk.Label(left_frame, text="Liczba przedziałów (L_bins):", font=("Arial", 11, "bold")).pack(anchor='w', pady=(10, 0))

bin_count = tk.StringVar(root) 
bin_options = [256, 128, 64, 32, 16] 
bin_count.set(bin_options[0]) 

bin_combobox = ttk.Combobox(left_frame, textvariable=bin_count, state="readonly", width=40)
bin_combobox['values'] = bin_options
bin_combobox.pack(anchor='w', padx=5, pady=5)

tk.Label(left_frame, text="Podsumowanie wyników:", font=("Arial", 11, "bold")).pack(anchor='w', pady=(10, 0))

summary_table = ttk.Treeview(left_frame, columns=("method", "type", "time"), show="headings", height=5)
summary_table.pack(fill="x", padx=5, pady=5)

summary_table.heading("method", text="Metoda")
summary_table.heading("type", text="Typ obrazu")
summary_table.heading("time", text="Średni czas [ms]")

summary_table.column("method", width=150, anchor="center")
summary_table.column("type", width=120, anchor="center")
summary_table.column("time", width=120, anchor="center")

tk.Label(left_frame, text="Weryfikacja poprawności wyników:", font=("Arial", 11, "bold")).pack(anchor='w', pady=(10, 0))

verification_table = ttk.Treeview(left_frame, columns=("compare", "diff"), show="headings", height=5)
verification_table.pack(fill="x", padx=5, pady=5)

verification_table.heading("compare", text="Porównanie")
verification_table.heading("diff", text="Różnica histogramów")

verification_table.column("compare", width=200, anchor="center")
verification_table.column("diff", width=120, anchor="center")

tk.Label(left_frame, text="Wyniki pomiarów:", font=("Arial", 11, "bold")).pack(anchor='w')

text_frame = tk.Frame(left_frame)
text_frame.pack(fill='both', expand=True)

scrollbar = tk.Scrollbar(text_frame)
scrollbar.pack(side='right', fill='y')

output_text = tk.Text(text_frame, wrap='word', yscrollcommand=scrollbar.set)
output_text.pack(fill='both', expand=True)
scrollbar.config(command=output_text.yview)

root.mainloop()
