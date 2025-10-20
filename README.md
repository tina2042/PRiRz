# Histogram i Equalizacja Kontrastu
## Wprowadzenie
Projekt realizuje operację equalizacji histogramu (wyrównywania kontrastu) obrazu z wykorzystaniem technik programowania równoległego i rozproszonego. Celem jest optymalizacja czasu obliczeń kluczowych etapów (szczególnie zliczania histogramu) oraz porównanie wydajności różnych architektur.
## Kluczowe Technologie
- OpenMP: Równoległość pamięci współdzielonej (lokalne histogramy + redukcja).
- MPI: Równoległość rozproszona (sumowanie częściowych histogramów).
- CUDA/OpenCL: Przetwarzanie na jednostce GPU (histogram per blok, prefix-sum).
- OpenCV: Obsługa i manipulacja obrazami.
## Struktura Projektu
Projekt jest zorganizowany modularnie, aby oddzielić implementacje dla różnych architektur równoległych

Projekt jest zorganizowany modularnie, aby oddzielić implementacje dla różnych architektur równoległych:

```bash
./PRiR
├── src/
│   ├── main.cpp              # Główny plik sterujący i wywołujący testy
│   
│   ├── sequential_proc.hpp   # Deklaracje funkcji bazowych (sekwencyjnych)
│   ├── sequential_proc.cpp   # Implementacja bazowa: Histogram, CDF, Transformacja
│   
│   ├── parallel_omp.hpp      # Deklaracje funkcji OpenMP
│   ├── parallel_omp.cpp      # Implementacja OpenMP (równoległe zliczanie histogramu)
│   
│   ├── parallel_mpi.cpp      # Implementacja MPI (podział danych, komunikacja MPI_Reduce)
│   
│   ├── parallel_cuda.cu      # Implementacja CUDA/OpenCL (jądra: histogram, prefix-sum, transformacja)
│
├── data/
│   ├── input/                # Katalog na obrazy wejściowe
│   └── output/               # Katalog na obrazy wyjściowe po equalizacji
├── reports/
│   ├── final_report.pdf      # Raport końcowy z analizą
│   └── charts/               # Wykresy wydajności i jakości
└── README.md
```
## Zakres Implementacji i Wersje
Projekt zawiera cztery główne implementacje, które będą mierzone i porównywane:
1. Wersja Bazowa (Sekwencyjna)
    - Cel: Punkt odniesienia dla pomiarów czasu.
    - Funkcje: Ładowanie obrazu w skali szarości, obliczenie histogramu, obliczenie skumulowanej dystrybuanty (CDF), zastosowanie transformacji.
2. OpenMP (Pamięć Współdzielona)
    - Technika: Równoległe zliczanie histogramu.
    - Strategia: Każdy wątek przetwarza wydzielony fragment obrazu i oblicza lokalny histogram. Lokalny histogramy są następnie redukowane do globalnego (poprzez redukcję atomową lub dedykowaną klauzulę/funkcję redukcyjną).Wersje: Skala szarości oraz kolorowa (trzy kanały RGB).
3. MPI (Pamięć Rozproszona)
    - Technika: Dystrybucja obrazu i sumowanie częściowych wyników.
    - Strategia: Obraz jest dzielony (MPI_Scatter) między $N$ procesów. Każdy proces oblicza częściowy histogram dla swojego bloku danych. Wyniki są sumowane do procesu głównego za pomocą MPI_Reduce lub MPI_Allreduce.
4. CUDA / OpenCL (GPU)
    - Technika: Wykorzystanie masowego paralelizmu GPU.
    - Strategia:
        - Histogram: Histogram per blok (każdy blok oblicza swój lokalny histogram w pamięci współdzielonej bloku), a następnie łączenie i sumowanie wyników na GPU.
        - Equalizacja: Wykorzystanie algorytmu Prefix-Sum (scan) do efektywnego obliczenia skumulowanej dystrybuanty (CDF) na GPU.
    - Wersje: Skala szarości oraz kolorowa (równoległe przetwarzanie kanałów).
