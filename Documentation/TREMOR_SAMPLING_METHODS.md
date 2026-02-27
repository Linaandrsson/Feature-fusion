# Tremor Sampling Methods

## Oversikt

Du har nå to metoder for å generere tremor-parametere i datasettet ditt:

### 1. **Subject-basert metode** (Standard)
Hver subject har faste tremor-karakteristikker som modellerer en Parkinson-pasient.

**Bruk når:**
- Du ønsker realistiske pasient-spesifikke profiler
- Du studerer forskjeller mellom subjects med ulike alvorlighetsgrader
- Du vil ha konsistente parametere per subject

**Konfigurasjon:**
```python
# I DataGenerator_Tremor.py
TREMOR_SAMPLING_METHOD = "subject"
```

**Eksempel output:**
```
Subject 1: Acc=0.10 m/s², Gyro=1.2 deg/s, Freq=5.5 Hz (konsistent for alle vinduer)
Subject 5: Acc=0.45 m/s², Gyro=9.9 deg/s, Freq=4.8 Hz (konsistent for alle vinduer)
```

### 2. **Interval-basert metode** (Ny)
Sampler tremor-parametere uavhengig for hvert vindu fra alvorlighetsområder.

**Bruk når:**
- Du ønsker høy variabilitet i datasettet
- Du gjør augmentation-studier
- Du vil ha mange forskjellige tremor-eksempler

**Konfigurasjon:**
```python
# I DataGenerator_Tremor.py
TREMOR_SAMPLING_METHOD = "interval"
TREMOR_AUGMENT_MODE = "mild_mod"  # eller "mod_severe", "clean"
```

**Eksempel output:**
```
Window 1: Score=1, Acc=0.113 m/s², Gyro=1.4 deg/s, Freq=5.9 Hz
Window 2: Score=2, Acc=0.176 m/s², Gyro=3.9 deg/s, Freq=6.2 Hz
Window 3: Score=2, Acc=0.664 m/s², Gyro=14.6 deg/s, Freq=6.4 Hz
```

## Detaljert sammenligning

| Egenskap | Subject-basert | Interval-basert |
|----------|----------------|-----------------|
| **RMS baseline** | Fra `A_SUBJECT[subject_id]` | Sampled fra `SCORE_RMS_RANGE[score]` |
| **Frekvens** | Fra `FREQ_TREMOR[subject_id]` | Sampled fra `FREQ_RANGE_HZ` (3.5-7.0 Hz) |
| **Konsistens** | Samme baseline per subject | Varierer per vindu |
| **Modulasjon** | body_part × activity × jitter | Kun variation noise |
| **Bruk** | Pasient-studier | Augmentation-studier |

## Hvordan bytte mellom metodene

### I tremor_parkinson_config.py:
```python
# Sett standard metode
DEFAULT_SAMPLING_METHOD = "subject"  # eller "interval"
DEFAULT_AUGMENT_MODE = "mild_mod"    # for interval-metoden
```

### I DataGenerator_Tremor.py:
```python
# Velg metode for dataset-generering
TREMOR_SAMPLING_METHOD = "subject"  # eller "interval"
TREMOR_AUGMENT_MODE = "mild_mod"    # kun relevant for "interval"
```

### Direkte i kode:
```python
from Noise_simulation.Tremor import precompute_tremor_cache_with_parkinson_model

# Subject-basert
tremor_cache = precompute_tremor_cache_with_parkinson_model(
    window_specs=window_specs,
    sensor_column_mapping=SENSORS,
    data_loader_func=load_data,
    fs=50.0,
    sampling_method="subject",  # <-- Velg her
    scenario_seed=42
)

# Interval-basert
tremor_cache = precompute_tremor_cache_with_parkinson_model(
    window_specs=window_specs,
    sensor_column_mapping=SENSORS,
    data_loader_func=load_data,
    fs=50.0,
    sampling_method="interval",  # <-- Velg her
    augment_mode="mild_mod",     # <-- Spesifiser alvorlighet
    scenario_seed=42
)
```

## Augment modes (for interval-metoden)

| Mode | Scores | RMS Range | Beskrivelse |
|------|--------|-----------|-------------|
| `"clean"` | 0 | 0.0 | Ingen tremor |
| `"mild_mod"` | 1-2 | 0.07-0.7 m/s² | Mild til mild-moderat |
| `"mod_severe"` | 3-4 | 0.7-6.0 m/s² | Moderat-alvorlig til alvorlig |

## Metadata-feltier

Begge metodene returnerer metadata per vindu:

```python
meta = {
    'subject_id': int,
    'body_part': str,
    'activity': int,
    'acc_target_rms': float,      # m/s²
    'gyro_target_rms': float,     # deg/s
    'freq_hz': float,             # Hz
    'severity': str,              # "mild", "mild-moderate", etc.
    'sampling_method': str,       # "subject" eller "interval"
    
    # Kun for interval-metoden:
    'augment_mode': str,          # "mild_mod", "mod_severe"
    'score': int,                 # 0-4
}
```

## Anbefalinger

### For trenings-datasett:
- **Subject-basert**: Hvis du vil modellere reelle pasienter med konsistente profiler
- **Interval-basert**: Hvis du vil ha maksimal variasjon for robust modellering

### For augmentation-studier:
- **Interval-basert anbefales**: Gir høyere diversitet

### For validering:
- **Subject-basert anbefales**: Lar deg evaluere per pasient-gruppe

## Eksempel: Generere flere datasett

```python
# Generer 3 datasett med forskjellige konfigurasjoner

# 1. Subject-basert (standard)
TREMOR_SAMPLING_METHOD = "subject"
# Kjør DataGenerator_Tremor.py

# 2. Interval-basert: mild-moderate
TREMOR_SAMPLING_METHOD = "interval"
TREMOR_AUGMENT_MODE = "mild_mod"
# Kjør DataGenerator_Tremor.py

# 3. Interval-basert: moderate-severe
TREMOR_SAMPLING_METHOD = "interval"
TREMOR_AUGMENT_MODE = "mod_severe"
# Kjør DataGenerator_Tremor.py
```

## Testing

Test at begge metodene fungerer:

```bash
# Test config
cd "/Volumes/NO NAME/Master Lina/Code/Noise_simulation"
python tremor_parkinson_config.py

# Test import
python -c "import Tremor; print(f'Default: {Tremor.pk_config.DEFAULT_SAMPLING_METHOD}')"

# Test DataGenerator
cd "/Volumes/NO NAME/Master Lina/Code"
python -c "import DataGenerator_Tremor as DG; print(f'Method: {DG.TREMOR_SAMPLING_METHOD}')"
```

## Feilsøking

**Problem:** "Unknown sampling_method"
- **Løsning:** Bruk kun `"subject"` eller `"interval"`

**Problem:** Interval-metoden gir samme verdier for alle vinduer
- **Løsning:** Sjekk at du ikke har satt samme seed for alle vinduer

**Problem:** Subject-metoden feiler med "Subject X not found"
- **Løsning:** Sjekk at subject_id finnes i `A_SUBJECT` dict i tremor_parkinson_config.py
