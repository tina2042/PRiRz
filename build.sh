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

echo ""
echo "✅ Kompilacja zakończona pomyślnie!"
echo "➡️  Plik wykonywalny: $OUTPUT"
echo ""
echo "Uruchomienie:"
echo "   ./hist_eq <ścieżka_do_obrazu>"
echo "=============================================="
