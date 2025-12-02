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
echo " Sprawdzanie i instalacja zależności"
echo "=============================================="

# -----------------------------------------------
# Funkcja: jeśli komenda nie istnieje → zainstaluj pakiet
# -----------------------------------------------
install_if_missing() {
    CMD=$1
    PKG=$2

    if ! command -v "$CMD" >/dev/null 2>&1; then
        echo "❗ Brak: $CMD → instaluję pakiet: $PKG"
        sudo apt update
        sudo apt install -y "$PKG"
    else
        echo "✔ $CMD OK"
    fi
}

# -------------------------
# Wymagane narzędzia systemowe
# -------------------------
install_if_missing pkg-config pkgconf
install_if_missing g++ g++
install_if_missing mpicxx openmpi-bin
install_if_missing mpirun openmpi-bin

if [ ! -f /usr/include/mpi.h ] && [ ! -f /usr/include/mpi/mpi.h ]; then
    echo "❗ Brak mpi.h — instaluję libopenmpi-dev"
    sudo apt update
    sudo apt install -y libopenmpi-dev
else
    echo "✔ Plik mpi.h znaleziony"
fi

if ! mpicxx -show >/dev/null 2>&1; then
    echo "❗ mpicxx nie skonfigurowany poprawnie"
    exit 1
fi

# -------------------------
# OpenCV
# -------------------------
if ! pkg-config --exists opencv4; then
    echo "❗ Brak OpenCV → instaluję..."
    sudo apt update
    sudo apt install -y libopencv-dev
else
    echo "✔ OpenCV OK (wersja: $(pkg-config --modversion opencv4))"
fi

# -------------------------
# CUDA toolkit + NVCC
# -------------------------
if ! command -v nvcc >/dev/null 2>&1; then
    echo "❗ Brak NVCC → instaluję nvidia-cuda-toolkit"
    sudo apt update
    sudo apt install -y nvidia-cuda-toolkit
else
    echo "✔ NVCC OK (wersja: $(nvcc --version | grep release))"
fi

# ============================================
# PYTHON
# ============================================
echo ""
echo "== Python: sprawdzanie =="

# tkinter
if ! python3 - <<EOF
import tkinter
EOF
then
    echo "❗ Brak tkinter → instaluję python3-tk"
    sudo apt update
    sudo apt install -y python3-tk
else
    echo "✔ tkinter OK"
fi

# pandas
if ! python3 - <<EOF
import pandas
EOF
then
    echo "❗ Brak pandas → instaluję python3-pandas"
    sudo apt update
    sudo apt install -y python3-pandas
else
    echo "✔ pandas OK"
fi

# matplotlib
if ! python3 - <<EOF
import matplotlib
EOF
then
    echo "❗ Brak matplotlib → instaluję python3-matplotlib"
    sudo apt update
    sudo apt install -y python3-matplotlib
else
    echo "✔ matplotlib OK"
fi

echo "✔ Pythonowe biblioteki OK"


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