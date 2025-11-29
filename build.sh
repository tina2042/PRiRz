#!/bin/bash
# ==============================================
#  build.sh — kompilacja projektu (CUDA + OpenMP + OpenCV)
# ==============================================

set -e

SRC_DIR="./src"

OUTPUT="hist_eq"

OUTPUT_SEQ_OMP_CUDA="hist_eq"

OUTPUT_MPI="mpi_runner"

echo "=============================================="
echo " Kompilacja projektu"
echo "=============================================="

nvcc -Xcompiler -fopenmp -std=c++17 \
    "$SRC_DIR/main.cpp" \
    "$SRC_DIR/sequential_proc.cpp" \
    "$SRC_DIR/parallel_omp.cpp" \
    "$SRC_DIR/parallel_cuda.cu" \
    -o "$OUTPUT" \
    `pkg-config --cflags --libs opencv4`


echo ""
echo "--- 2. Kompilacja: $OUTPUT_MPI (MPI) ---"
echo "Używam mpicxx..."

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