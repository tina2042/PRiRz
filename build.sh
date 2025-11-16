#!/bin/bash
# ==============================================
#  build.sh — kompilacja projektu PRiR (CUDA + OpenMP + OpenCV)
# ==============================================

# Zatrzymaj skrypt, jeśli wystąpi błąd
set -e

# Ścieżka do katalogu źródeł
SRC_DIR="./src"

# Nazwa pliku wynikowego
OUTPUT="hist_eq"

# Nazwa pliku wynikowego dla SEQ/OMP/CUDA
OUTPUT_SEQ_OMP_CUDA="hist_eq"

# Nazwa pliku wynikowego dla MPI
OUTPUT_MPI="mpi_runner"

echo "=============================================="
echo " 🔧 Kompilacja projektu PRiR"
echo "=============================================="

# Kompilacja z nvcc (CUDA + OpenMP + OpenCV)
nvcc -Xcompiler -fopenmp -std=c++17 \
    "$SRC_DIR/main.cpp" \
    "$SRC_DIR/sequential_proc.cpp" \
    "$SRC_DIR/parallel_omp.cpp" \
    "$SRC_DIR/parallel_cuda.cu" \
    -o "$OUTPUT" \
    `pkg-config --cflags --libs opencv4`


# --- 2. Kompilacja pliku MPI (mpi_runner) za pomocą mpicxx ---
echo ""
echo "--- 2. Kompilacja: $OUTPUT_MPI (MPI) ---"
echo "Używam mpicxx..."

# Używamy mpicxx, aby automatycznie linkować biblioteki MPI.
# Upewniamy się, że nie dodajemy flagi -fopenmp, chyba że planujemy użyć OMP wewnątrz MPI.
mpicxx -std=c++17 -o mpi_runner \
    "$SRC_DIR/main_mpi.cpp" \
    "$SRC_DIR/parallel_mpi.cpp" \
    "$SRC_DIR/sequential_proc.cpp" \
    `pkg-config --cflags --libs opencv4`

if [ $? -ne 0 ]; then
    echo "BŁĄD: Kompilacja $OUTPUT_MPI nie powiodła się."
    exit 1
fi
echo "$OUTPUT_MPI skompilowany pomyślnie."

echo ""
echo "Kompilacja zakończona pomyślnie!"
echo "Komenda Uruchomienia (SEQ/OMP/CUDA):"
echo " ./$OUTPUT_SEQ_OMP_CUDA <ścieżka_do_obrazu> ALL <liczba_przedziałów>"
echo ""
echo "Komenda Uruchomienia (MPI):"
echo " mpirun -np N ./$OUTPUT_MPI <ścieżka_do_obrazu> MPI_GRAY <liczba_przedziałów>"
echo "=============================================="