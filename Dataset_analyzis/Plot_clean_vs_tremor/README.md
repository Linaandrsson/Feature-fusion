# Clean vs Tremor Signal Comparison

Dette mappen inneholder scripts for å visualisere sensor-signaler før og etter pålagt Parkinson tremor.

## Filer

- **`plot_clean_vs_tremor.py`** - Hovedscript med interaktiv CLI
- **`plot_example.py`** - Eksempel for rask plotting uten interaksjon
- **`README.md`** - Denne filen

## Funksjoner

Scriptene genererer side-ved-side plots som viser:
- Alle sensorer (Acc_chest, ECG, Acc_ankle, Gyro_ankle, Mag_ankle, Acc_arm, Gyro_arm, Mag_arm)
- Clean signal (venstre) vs Tremor signal (høyre)
- Metadata: Subject ID, tremor score, RMS, frekvens
- Activity label i tittel

## Bruk

### Metode 1: Interaktiv CLI

Kjør hovedscriptet og følg interaktive prompts:

```bash
python plot_clean_vs_tremor.py
```

Du vil bli bedt om å velge:
1. **Første dataset variant** (clean, mild_mod, eller mod_severe)
2. **Andre dataset variant** (clean, mild_mod, eller mod_severe)
3. **Aktivitet** (ved index 0-11 eller navn, f.eks. "Walking")
4. **Window index** (hvilket sample av aktiviteten, default: 0)
5. **Lagre plots?** (y/n)

### Metode 2: Direkte kjøring med eksempel-script

Rediger parametere i `plot_example.py` og kjør:

```python
# I plot_example.py, modifiser:
VARIANT1 = "clean"           # Første variant
VARIANT2 = "mild_mod"        # Andre variant
ACTIVITY_IDX = 3             # Activity index (3 = Walking)
WINDOW_IDX = 0               # Window index
SAVE_PLOTS = True            # Lagre plots til fil
```

Deretter kjør:

```bash
python plot_example.py
```

### Metode 3: Programmatisk bruk

Importer funksjonene i ditt eget script:

```python
from plot_clean_vs_tremor import plot_all_sensors_comparison

# Plot alle sensorer
figures = plot_all_sensors_comparison(
    variant1_name="s2_w2_fs50_tremor_clean",
    variant2_name="s2_w2_fs50_tremor_mild_mod",
    activity_idx=3,      # Walking
    window_idx=0,
    fs=50,
    save_dir="my_plots"  # eller None for å vise
)
```

## Aktiviteter

| Index | Aktivitet                    |
|-------|------------------------------|
| 0     | Standing still               |
| 1     | Sitting and relaxing         |
| 2     | Lying down                   |
| 3     | Walking                      |
| 4     | Climbing stairs              |
| 5     | Waist bends forward          |
| 6     | Frontal elevation of arms    |
| 7     | Knees bending                |
| 8     | Cycling                      |
| 9     | Jogging                      |
| 10    | Running                      |
| 11    | Jump front & back            |

## Dataset Varianter

- **`clean`** - Ingen tremor (tremor score = 0)
- **`mild_mod`** - Mild til moderate tremor (score 1-2)
- **`mod_severe`** - Moderate til severe tremor (score 3-4)

## Sensorer

Scriptet plotter alle 8 sensorer:

1. **Acc_chest** - Accelerometer på brystet (3 akser: X, Y, Z)
2. **ECG** - Elektrokardiogram på brystet (2 leads)
3. **Acc_ankle** - Accelerometer på ankelen (3 akser)
4. **Gyro_ankle** - Gyroskop på ankelen (3 akser)
5. **Mag_ankle** - Magnetometer på ankelen (3 akser)
6. **Acc_arm** - Accelerometer på armen (3 akser)
7. **Gyro_arm** - Gyroskop på armen (3 akser)
8. **Mag_arm** - Magnetometer på armen (3 akser)

## Output

Plots viser:
- **Venstre kolonne**: Signal fra første variant (f.eks. clean)
- **Høyre kolonne**: Signal fra andre variant (f.eks. tremor)
- **Tittel**: Activity navn og ID
- **Subtitler**: Variant navn, Subject ID, Tremor score, RMS, Frekvens
- **Y-akse**: Sensorverdi med enhet
- **X-akse**: Tid (sekunder)

Hvis plots lagres, opprettes en mappe:
```
plots_{variant1}_vs_{variant2}_act{activity_idx}/
```

Hver sensor lagres som:
```
{sensor_name}_{variant1}_vs_{variant2}_act{activity_idx}_win{window_idx}.png
```

## Eksempler

### Eksempel 1: Sammenlign clean vs mild tremor for Walking
```bash
python plot_example.py
# Med VARIANT1="clean", VARIANT2="mild_mod", ACTIVITY_IDX=3
```

### Eksempel 2: Interaktiv sammenligning
```bash
python plot_clean_vs_tremor.py
# Select: 1 (clean), 2 (mild_mod), Walking, 0, n
```

### Eksempel 3: Sammenlign moderate vs severe tremor for Running
```python
from plot_clean_vs_tremor import plot_all_sensors_comparison
import matplotlib.pyplot as plt

figures = plot_all_sensors_comparison(
    variant1_name="s2_w2_fs50_tremor_mild_mod",
    variant2_name="s2_w2_fs50_tremor_mod_severe",
    activity_idx=10,  # Running
    window_idx=5,
    fs=50,
    save_dir=None
)

plt.show()
```

## Tips

- Bruk `window_idx` for å se forskjellige samples av samme aktivitet
- Clean vs mild_mod gir god visualisering av tremor effect
- Gyro_arm og Acc_arm viser typisk mest tydelig tremor
- ECG og Acc_chest har ikke tremor (tremor-free zones)

## Avhengigheter

- numpy
- matplotlib
- pathlib (standard library)

Installasjon:
```bash
pip install numpy matplotlib
```
