# IMU & Tremor Dataset Analysis

Scripts for å visualisere og analysere IMU sensor data og tremor dataset.

## Filer

### IMU Test Data (CSV)
- **`plot_imu_data.py`** - Plotting av IMU data fra CSV med interaktiv CLI
- **`plot_example.py`** - Eksempel for rask plotting
- **`calculate_rms.py`** - RMS beregning for IMU test data med interaktiv CLI
- **`rms_example.py`** - Eksempel for rask RMS-beregning

### Tremor Dataset (NPZ)
- **`calculate_rms_tremor.py`** - RMS beregning for tremor dataset med interaktiv CLI
- **`rms_tremor_example.py`** - Eksempel for rask RMS-beregning av tremor data

### Output Directories
- **`plots/`** - IMU plots (PNG files)
- **`rms_results_IMU_test/`** - RMS resultater for IMU test data (TXT files)
- **`rms_results_tremor/`** - RMS resultater for tremor dataset (TXT files)

### Dokumentasjon
- **`README.md`** - Denne filen

## Data Format

Scriptet forventer CSV-filer fra IMU-tester med følgende kolonner:
- `RWrist_TimestampSync_Unix_CAL` - Timestamp (ms)
- `RWrist_Accel_LN_X_CAL/Y/Z` - Accelerometer (m/s²)
- `RWrist_Gyro_X_CAL/Y/Z` - Gyroscope (deg/s)
- `RWrist_Mag_X_CAL/Y/Z` - Magnetometer (local flux)

## Bruk - Plotting

### Metode 1: Interaktiv CLI

Kjør hovedscriptet og følg interaktive prompts:

```bash
python plot_imu_data.py
```

Du vil bli bedt om å velge:
1. **Window duration** (hvor mange sekunder du vil plotte, default: 4s)
2. **Single eller multiple windows**
3. **Start time(s)** (når i datasettet du vil starte)
4. **Lagre plots?** (y/n)

### Metode 2: Direkte kjøring med eksempel-script

Rediger parametere i `plot_example.py`:

```python
# I plot_example.py, modifiser:
START_TIME = 10.0          # Start tid (sekunder)
WINDOW_DURATION = 4.0      # Vindu-lengde (sekunder)
SAVE_PLOT = True           # Lagre til fil

# For flere vinduer:
MULTIPLE_WINDOWS = [0, 10, 20, 30]  # Liste med start-tider
```

Deretter kjør:

```bash
python plot_example.py
```

### Metode 3: Programmatisk bruk

Importer funksjonene i ditt eget script:

```python
from plot_imu_data import load_imu_data, plot_imu_window

# Last data
df = load_imu_data("path/to/csv/file.csv")

# Plot ett vindu
fig = plot_imu_window(
    df, 
    window_start=10.0,     # Start ved 10 sekunder
    window_duration=4.0,   # Vis 4 sekunder
    save_path="my_plot.png"
)
```

For multiple vinduer:

```python
from plot_imu_data import load_imu_data, plot_multiple_windows

df = load_imu_data("path/to/csv/file.csv")

figures = plot_multiple_windows(
    df,
    window_starts=[0, 5, 10, 15, 20],  # 5 vinduer
    window_duration=4.0,
    save_dir="my_plots"
)
```

## Bruk - RMS Beregning

### RMS (Root Mean Square)

RMS kvantifiserer signal-amplituden og beregnes som:
```
RMS = sqrt(mean(signal²))
```

For 3-akse sensorer beregnes både per-akse RMS og total magnitude:
```
RMS_magnitude = sqrt(RMS_x² + RMS_y² + RMS_z²)
```

### Metode 1: Interaktiv RMS-beregning

```bash
python calculate_rms.py
```

Du vil bli bedt om å velge:
1. **Window duration** (vindu-lengde for RMS-beregning)
2. **Single eller multiple windows**
3. **Start time(s)** 
4. **Lagre til TXT?** (y/n)

### Metode 2: Rask RMS-beregning

Rediger parametere i `rms_example.py`:

```python
# I rms_example.py, modifiser:
START_TIME = 10.0          # Start tid (sekunder)
WINDOW_DURATION = 4.0      # Vindu-lengde (sekunder)
SAVE_TXT = True            # Lagre resultater til TXT

# For flere vinduer:
MULTIPLE_WINDOWS = [0, 30, 60, 90]
```

Deretter kjør:

```bash
python rms_example.py
```

### Metode 3: Programmatisk RMS-beregning

```python
from calculate_rms import calculate_all_rms, print_rms_results
from plot_imu_data import load_imu_data

# Last data
df = load_imu_data("path/to/csv/file.csv")

# Beregn RMS for et vindu
results = calculate_all_rms(df, start_time=10.0, duration=4.0)

# Vis resultater
print_rms_results(results)

# Resultat-struktur:
# results = {
#     'window': {'start_time', 'duration', 'end_time', 'num_samples'},
#     'accelerometer': {'x', 'y', 'z', 'magnitude'},
#     'gyroscope': {'x', 'y', 'z', 'magnitude'},
#     'magnetometer': {'x', 'y', 'z', 'magnitude'}
# }
```

### RMS Output

Terminal-output viser:
- Tidsvindu og antall samples
- Per-sensor resultater (Acc, Gyro, Mag)
- Per-akse RMS (X, Y, Z)
- Total magnitude RMS

TXT-filer lagres i `rms_results_IMU_test/` mappen med formatert output:
```
================================================================================
RMS ANALYSIS RESULTS
================================================================================
Time window: [28.00 - 30.00] s
Duration: 2.00 s
Samples: 128
================================================================================

ACCELEROMETER (m/s²)
--------------------------------------------------------------------------------
  X-axis RMS:          7.2689 m/s²
  Y-axis RMS:          4.4696 m/s²
  Z-axis RMS:          5.2843 m/s²
  Magnitude RMS:      10.0368 m/s²

GYROSCOPE (deg/s)
--------------------------------------------------------------------------------
  X-axis RMS:          6.7280 deg/s
  Y-axis RMS:         43.8484 deg/s
  Z-axis RMS:         11.7432 deg/s
  Magnitude RMS:      45.8896 deg/s

MAGNETOMETER (local flux)
--------------------------------------------------------------------------------
  X-axis RMS:          0.2605 local flux
  Y-axis RMS:          0.9893 local flux
  Z-axis RMS:          0.3323 local flux
  Magnitude RMS:       1.0756 local flux

================================================================================
```

Filnavn: `rms_s{start}_d{duration}.txt`
Eksempel: `rms_s28.0_d2.0.txt`

---

## Bruk - Tremor Dataset RMS Analyse

Beregn RMS for alle 8 sensorer fra tremor dataset (NPZ-filer).

### ⚠️ Viktig om Tremor Metadata

**Chest-sensorer har IKKE tremor metadata!**

Tremor datasets er generert med chest (Acc_chest, ECG) som **TREMOR_FREE_SENSORS**. Dette betyr:
- Chest-sensorer har tremor_freq = 0.0 Hz
- Chest-sensorer har tremor_score = 0
- Chest-sensorer har tremor_acc_rms = 0.0

**For tremor metadata brukes derfor Acc_arm** (eller Acc_ankle), som har korrekte verdier:
- **Clean**: freq = 0 Hz, score = 0
- **Mild**: freq = 3.5-7 Hz, score = 1-2, RMS < 0.7 m/s²
- **Severe**: freq = 3.5-7 Hz, score = 3-4, RMS > 0.7 m/s²

### Sensorer i tremor dataset:
- **Accelerometer Chest** (3 akser, m/s²) *- ingen tremor pålagt*
- **ECG Chest** (2 leads, mV) *- ingen tremor pålagt*
- **Accelerometer Ankle** (3 akser, m/s²) ✓ *har tremor metadata*
- **Gyroscope Ankle** (3 akser, deg/s) ✓ *har tremor metadata*
- **Magnetometer Ankle** (3 akser, μT) ✓ *har tremor metadata*
- **Accelerometer Arm** (3 akser, m/s²) ✓ *har tremor metadata*
- **Gyroscope Arm** (3 akser, deg/s) ✓ *har tremor metadata*
- **Magnetometer Arm** (3 akser, μT) ✓ *har tremor metadata*

### Metode 1: Interaktiv tremor RMS-analyse

```bash
python calculate_rms_tremor.py
```

Du vil bli bedt om å velge:
1. **Tremor variant** (clean, mild, eller severe)
2. **Activity** (0-11, eller navn som "Walking")
3. **Window index** (hvilket vindu av denne aktiviteten, default: 0)

### Metode 2: Rask tremor RMS-beregning

Rediger parametere i `rms_tremor_example.py`:

```python
# I rms_tremor_example.py, modifiser:
VARIANT = 'mild'           # 'clean', 'mild', eller 'severe'
ACTIVITY_IDX = 3           # 0-11 (3 = Walking)
WINDOW_IDX = 0             # Hvilket vindu (0 = første)
SAVE_TXT = True            # Lagre til TXT
```

Deretter kjør:

```bash
python rms_tremor_example.py
```

### Metode 3: Programmatisk tremor RMS-beregning

```python
from calculate_rms_tremor import (
    get_activity_window, 
    calculate_all_rms, 
    print_rms_results,
    export_rms_to_txt,
    ACTIVITY_NAMES
)

# Last et vindu for en aktivitet
window_data = get_activity_window(
    variant_key='mild',      # 'clean', 'mild', 'severe'
    activity_idx=3,          # 0-11 (3 = Walking)
    window_idx=0             # Hvilket vindu
)

# Beregn RMS for alle sensorer
results = calculate_all_rms(window_data)

# Vis resultater
print_rms_results(results)

# Lagre til TXT
export_rms_to_txt(results)

# Resultatstruktur:
# results = {
#     'metadata': {
#         'variant': 'mild',
#         'activity_name': 'Walking',
#         'window_idx': 0,
#         'subject_id': 1,
#         'tremor_freq': 5.0,
#         'tremor_score': 0.25
#     },
#     'sensors': {
#         'Acc_chest': {'rms': {...}, 'unit': 'm/s²'},
#         'ECG': {'rms': {...}, 'unit': 'mV'},
#         ...
#     }
# }
```

### Tremor RMS Output

Terminal-output viser:
- Variant og aktivitet
- Window index og global index
- Subject ID
- **Tremor metadata** (frequency, acc RMS, gyro RMS, score) - *fra Acc_arm sensor*
- Per-sensor RMS (alle 8 sensorer)
- Per-akse RMS + magnitude (for 3-akse sensorer)

**Eksempel - MILD variant:**
```
================================================================================
RMS ANALYSIS RESULTS - TREMOR DATASET
================================================================================
Variant: MILD (s2_w2_fs50_tremor_mild_mod)
Activity: Walking (index 3)
Window: 0 (global index: 90)
Subject: 1
Tremor frequency: 3.94 Hz
Tremor score: 1.0000
================================================================================

ACCELEROMETER CHEST (m/s²)
--------------------------------------------------------------------------------
  X-axis RMS:          1.0010 m/s²
  Y-axis RMS:          1.0000 m/s²
  Z-axis RMS:          1.0001 m/s²
  Magnitude RMS:       1.7327 m/s²
```

**Eksempel - SEVERE variant:**
```
================================================================================
RMS ANALYSIS RESULTS - TREMOR DATASET
================================================================================
Variant: SEVERE (s2_w2_fs50_tremor_mod_severe)
Activity: Walking (index 3)
Window: 0 (global index: 90)
Subject: 1
Tremor frequency: 3.94 Hz
Tremor score: 3.0000
================================================================================

ACCELEROMETER ARM (m/s²)
--------------------------------------------------------------------------------
  X-axis RMS:          0.9999 m/s²
  Y-axis RMS:          0.9997 m/s²
  Z-axis RMS:          0.9991 m/s²
  Magnitude RMS:       1.7313 m/s²
```

### Tremor Variant Sammenligning

Typiske verdier for Activity 3 (Walking), Window 0:

| Variant | Tremor Freq | Acc RMS | Gyro RMS | Score | Beskrivelse |
|---------|-------------|---------|----------|-------|-------------|
| **CLEAN** | 0.00 Hz | 0.0000 m/s² | 0.00 deg/s | 0 | Ingen tremor |
| **MILD** | ~4 Hz | ~0.09 m/s² | ~1 deg/s | 1 | Mild tremor |
| **SEVERE** | ~4 Hz | ~1.9 m/s² | ~26 deg/s | 3-4 | Moderat-alvorlig tremor |

**Merk:** RMS-verdiene i tabellen over er *tremor metadata* (original ikke-normalisert data).
De faktiske sensor RMS-verdiene i outputen er ~1.7 m/s² pga. z-score normalisering.

### Output Filnavn

Filnavn: `rms_{variant}_act{activity_idx}_win{window_idx}.txt`
Eksempler: 
- `rms_clean_act3_win0.txt` (Clean - Walking - første vindu)
- `rms_severe_act10_win2.txt` (Severe - Running - tredje vindu)

### Aktiviteter (0-11)

```
0:  Standing still
1:  Sitting and relaxing
2:  Lying down
3:  Walking
4:  Climbing stairs
5:  Waist bends forward
6:  Frontal elevation of arms
7:  Knees bending
8:  Cycling
9:  Jogging
10: Running
11: Jump front & back
```

---

## Output - Plotting

Plottene viser 3 subplots:

1. **Accelerometer** (øverst)
   - X, Y, Z akser i rødt, grønt, blått
   - Enhet: m/s²

2. **Gyroscope** (midten)
   - X, Y, Z akser i rødt, grønt, blått
   - Enhet: deg/s

3. **Magnetometer** (nederst)
   - X, Y, Z akser i rødt, grønt, blått
   - Enhet: local flux

Hver plot inkluderer:
- Tidsvindu i tittel
- Antall samples
- Grid for enkel avlesning
- Legend for hver akse

## Eksempler

### Eksempel 1: Plot 4 sekunder fra t=10s
```bash
python plot_example.py
# Med START_TIME=10.0, WINDOW_DURATION=4.0
```

### Eksempel 2: Plot flere vinduer (0s, 10s, 20s, 30s)
```python
# I plot_example.py:
MULTIPLE_WINDOWS = [0, 10, 20, 30]
WINDOW_DURATION = 4.0
SAVE_PLOT = True
```

### Eksempel 3: Tremor RMS-analyse - Sammenligne varianter
```bash
# Interaktiv modus
python calculate_rms_tremor.py
# Velg: variant=severe, activity=3 (Walking), window=0

# Rask modus - rediger rms_tremor_example.py:
VARIANT = 'severe'
ACTIVITY_IDX = 3
WINDOW_IDX = 0
SAVE_TXT = True

python rms_tremor_example.py
```

**Output:** `rms_results_tremor/rms_severe_act3_win0.txt` med:
- Tremor metadata: freq=3.94 Hz, score=3
- RMS for alle 8 sensorer

### Eksempel 4: Programmatisk tremor analyse
```python
from calculate_rms_tremor import (
    get_activity_window,
    calculate_all_rms,
    export_rms_to_txt
)

# Sammenlign clean vs severe for Walking
for variant in ['clean', 'severe']:
    window_data = get_activity_window(variant, 3, 0)
    results = calculate_all_rms(window_data)
    export_rms_to_txt(results)
    
    meta = window_data['metadata']
    print(f"{variant}: freq={meta['tremor_freq']:.2f} Hz, score={meta['tremor_score']}")
```

### Eksempel 5: Interaktiv IMU plotting
```bash
python plot_imu_data.py
# Følg prompts:
# - Window duration: 4.0
# - Multiple windows? y
# - Start times: 0, 5, 10, 15
# - Save? y
```

### Eksempel 6: Custom tidsvindu
```python
from plot_imu_data import load_imu_data, plot_imu_window
import matplotlib.pyplot as plt

df = load_imu_data("Tremor_Session1_RWrist_Calibrated_PC.csv")

# Plot 2 sekunder fra t=15s
fig = plot_imu_window(df, window_start=15.0, window_duration=2.0)
plt.show()
```

## Tips

- **Utforsk datasettet**: Bruk interaktiv modus første gang for å se total lengde
- **Sammenligne perioder**: Bruk MULTIPLE_WINDOWS for å se flere tidspunkter
- **Korte vinduer**: For detaljert visning, bruk 1-2 sekunder
- **Lange vinduer**: For oversikt, bruk 5-10 sekunder
- **Høy oppløsning**: Plots lagres med 150 DPI

## Data Info

Når du laster data, vises:
- Total antall samples
- Total varighet (sekunder)
- Gjennomsnittlig sampling interval (ms)
- Estimert sampling rate (Hz)
- Tidsområde

## Lagring

Alle plots lagres i `plots/` mappen med navnekonvensjon: `s{start_time}_d{duration}`

**Single plot**:
```
plots/
  imu_plot_s10.0_d4.0.png   (start: 10s, duration: 4s)
  imu_plot_s0.0_d20.0.png   (start: 0s, duration: 20s)
```

**Multiple plots**:
```
plots/
  imu_window_s0.0_d4.0.png
  imu_window_s10.0_d4.0.png
  imu_window_s20.0_d4.0.png
  ...
```

## FAQ - Tremor Dataset

### Hvorfor er alle sensor RMS-verdier ~1.7 m/s²?

**Forklaring:** Tremor datasettene er **z-score normalisert** etter at tremor er pålagt. Dette betyr:
1. Tremor pålegges på rå sensordata (creates variation i amplitude)
2. Data normaliseres: `(x - mean) / std` → RMS blir ~1.0 per akse
3. Magnitude RMS = √(1² + 1² + 1²) ≈ 1.73 for 3-akse sensorer

**Tremor metadata** (tremor_freq, tremor_acc_rms, tremor_score) viser *original* tremor-parametere før normalisering.

### Hvorfor har Acc_chest tremor_freq = 0?

**Forklaring:** Chest-sensorer (Acc_chest, ECG) er definert som **TREMOR_FREE_SENSORS** i datagenereringen.
- Tremor pålegges IKKE på chest-sensorer
- Tremor metadata hentes fra **Acc_arm** som har korrekte verdier
- Arm og ankle har tremor pålagt → korrekte metadata-verdier

### Hvor ser jeg faktisk tremor i signalet?

Selv om RMS er normalisert til ~1.7, er tremor synlig i:
1. **Signal-forskjeller**: Clean vs Severe har MAX diff ~0.76, MEAN diff ~0.10
2. **Tremor metadata**: Viser original tremor amplitude og frekvens
3. **Visuell plotting**: Plot clean vs tremor med `plot_clean_vs_tremor.py`

## Avhengigheter

- numpy
- pandas
- matplotlib

Installasjon:
```bash
pip install numpy pandas matplotlib
```

## Tilpasning

Du kan enkelt modifisere scriptene for å:
- Endre farger (endre `'r-'`, `'g-'`, `'b-'` i plot-kommandoene)
- Endre figurstørrelse (parameter `figsize` i `plt.subplots`)
- Endre linjestil (parameter `linestyle`, `linewidth`)
- Legge til flere visualiseringer (f.eks. spektrogrammer)
