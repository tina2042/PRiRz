import subprocess
import os
import re
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPI_EXECUTABLE = "mpirun" 
MPI_RUNNER_EXECUTABLE = os.path.join(SCRIPT_DIR, "mpi_runner")

def parse_time(output):
    """Wyciąga czas wykonania z outputu C++."""
    last_line = output.strip().splitlines()[-1] if output else ""
    
    match = re.search(r"Sredni czas wykonania \(MPI\):\s*([\d.]+)\s*ms", last_line)
    
    if match:
        return float(match.group(1))
    
    print(f"[DEBUG PARSE]: Nie znaleziono czasu w: '{last_line.strip()}'", file=sys.stderr) 
    return None

def run_scaling(image_path, mode_prefix, output_csv_path, initial_procs):
    """
    Wykonuje pętlę skalowalności dla MPI.
    """
    process_counts = [1]
    
    p = initial_procs
    while p <= 8 and p not in process_counts:
         process_counts.append(p)
         p *= 2

    print(f"--- Sprawdzenie skalowalności MPI ({mode_prefix}) ---")
    
    try:
        with open(output_csv_path, 'w') as f:
            f.write("processes,time_ms\n")
            
            for N in process_counts:
                mode_arg = f"MPI_{mode_prefix.upper()}"
                
                command = [MPI_EXECUTABLE, "-np", str(N), MPI_RUNNER_EXECUTABLE, image_path, mode_arg]
                
                result = subprocess.run(command, capture_output=True, text=True)
                
                time_ms = parse_time(result.stdout)
                
                if time_ms is not None and result.returncode == 0:
                    f.write(f"{N},{time_ms}\n")
                    print(f"Procesy: {N} -> {time_ms:.2f} ms")
                else:
                    print(f"Procesy: {N} -> BŁĄD. Szczegóły w stderr.")
                    
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
    
    error = run_scaling(args.image_path, args.mode, args.csv_path, args.procs)
    
    if error:
        sys.exit(1)
    else:
        sys.exit(0)