import subprocess
import os
import re
import sys
import argparse

# --- Konfiguracja ścieżek i stałych ---
# Zakładamy, że ten skrypt jest uruchamiany z tego samego katalogu co inne pliki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPI_EXECUTABLE = "mpirun" 
MPI_RUNNER_EXECUTABLE = os.path.join(SCRIPT_DIR, "mpi_runner")


# Funkcja do parsowania pojedynczego czasu z outputu C++
def parse_time(output):
    """Wyciąga czas wykonania z outputu C++."""
    # Szuka linii typu: Sredni czas wykonania (MPI): 123.45 ms
    last_line = output.strip().splitlines()[-1] if output else ""
    
    # regex bez IGNORECASE, aby być bardziej precyzyjnym
    match = re.search(r"Sredni czas wykonania \(MPI\):\s*([\d.]+)\s*ms", last_line)
    
    if match:
        return float(match.group(1))
    
    # Dodaj to do debugowania, jeśli parsowanie zawodzi:
    print(f"[DEBUG PARSE]: Nie znaleziono czasu w: '{last_line.strip()}'", file=sys.stderr) 
    return None

def run_scaling(image_path, mode_prefix, output_csv_path, initial_procs):
    """
    Wykonuje pętlę skalowalności dla MPI.
    """
    # Wektory procesów do testowania (można dostosować)
    # Zaczynamy od N=1
    process_counts = [1]
    
    # Dodajemy procesy w potęgach 2, aż do 8 (lub więcej, w zależności od maszyny)
    p = initial_procs
    while p <= 8 and p not in process_counts:
         process_counts.append(p)
         p *= 2

    # Nagłówek dla stdout (dla interfejsu Pythona)
    print(f"--- Sprawdzenie skalowalności MPI ({mode_prefix}) ---")
    
    # 1. Otwarcie pliku CSV
    try:
        with open(output_csv_path, 'w') as f:
            f.write("processes,time_ms\n")
            
            # 2. Pętla pomiarowa dla Grayscale lub Color
            for N in process_counts:
                # W MPI, argumentem trybu zawsze będzie MPI_GRAY lub MPI_COLOR
                mode_arg = f"MPI_{mode_prefix.upper()}"
                
                # Budowanie komendy: mpirun -np N ./mpi_runner <image> <mode>
                command = [MPI_EXECUTABLE, "-np", str(N), MPI_RUNNER_EXECUTABLE, image_path, mode_arg]
                
                # Uruchomienie
                result = subprocess.run(command, capture_output=True, text=True)
                
                time_ms = parse_time(result.stdout)
                
                if time_ms is not None and result.returncode == 0:
                    # Zapis do pliku CSV
                    f.write(f"{N},{time_ms}\n")
                    # Wypisanie wyniku w formacie oczekiwanym przez użytkownika
                    print(f"Procesy: {N} -> {time_ms:.2f} ms")
                else:
                    print(f"Procesy: {N} -> BŁĄD. Szczegóły w stderr.")
                    # Opcjonalnie: print(result.stderr, file=sys.stderr)
                    
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono narzędzia lub pliku: {e.filename}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}", file=sys.stderr)
        return True
        
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zewnętrzny skrypt do pomiaru skalowalności MPI.")
    parser.add_argument("image_path", help="Ścieżka do obrazu wejściowego.")
    parser.add_argument("mode", choices=['GRAY', 'COLOR'], help="Tryb: GRAY lub COLOR.")
    parser.add_argument("csv_path", help="Ścieżka do zapisu pliku CSV.")
    parser.add_argument("procs", type=int, help="Domyślna liczba procesów do rozpoczęcia pomiaru (np. 4).")
    
    args = parser.parse_args()
    
    # Uruchomienie funkcji skalowalności
    error = run_scaling(args.image_path, args.mode, args.csv_path, args.procs)
    
    if error:
        sys.exit(1)
    else:
        sys.exit(0)