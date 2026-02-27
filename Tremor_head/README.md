# Tremor Head Models

Dette er to isolerte modeller for å utforske potensialet til tremor-prediksjonsmetodene, uavhengig av den større feature fusion-pipelinen.

## Oversikt

### Model 1: Regresjon (tremor_regression_train.py)
Predikerer kontinuerlige tremor-parametere:
- **Input**: Rå sensor-vinduer + aktivitets-label
- **Output**: 
  - `tremor_acc_rms` (m/s²) - Accelerometer tremor amplitude
  - `tremor_gyro_rms` (deg/s) - Gyroscope tremor amplitude  
  - `tremor_freq` (Hz) - Tremor frekvens

### Model 2: Klassifikasjon (tremor_classification_train.py)
Predikerer tremor-alvorlighetsgrad:
- **Input**: Rå sensor-vinduer + aktivitets-label
- **Output**: Tremor score (0-4)
  - 0: Ingen tremor (< 0.07 m/s²)
  - 1: Mild (0.07-0.15 m/s²)
  - 2: Mild-Moderat (0.15-0.7 m/s²)
  - 3: Moderat-Alvorlig (0.7-2.5 m/s²)
  - 4: Alvorlig (2.5-6.0 m/s²)

## Arkitektur

Begge modellene bruker samme backbone-struktur:

```
Input (rå sensor-data) → 1D CNN → Flatten
                                      ↓
Activity label → Embedding ─────→ Concat → Shared FC
                                              ↓
Model 1: → 3 separate regresjonshoder (Acc, Gyro, Freq)
Model 2: → Klassifikasjonshode (5 klasser)
```

### CNN Backbone
- Multi-layer 1D convolutions for tidsmessig feature extraction
- BatchNorm + ReLU + MaxPool + Dropout etter hver conv-layer
- Default: 3 layers med [32, 64, 64] filtre

### Activity Embedding
- Liten embedding (default 16-dim) for aktivitets-kontekst
- Konkateneres med CNN features før final layers

## Bruk

### 1. Konfigurasjon

Begge filer har en CONFIGURATION-seksjon øverst hvor du kan spesifisere:

```python
# Velg datasett (kan kombinere flere for data augmentation!)
DATASET_VARIANTS = ["s2_w2_tremor_clean", "s2_w2_tremor_parkinson"]

# Velg sensorer (kan kombinere flere)
SENSORS = ["Acc_arm", "Gyro_arm"]  

# Splitt-strategi
SPLIT_BY_SUBJECT = True  # True = subject-based (generalisering)
TEST_SUBJECTS = [9, 10]
VAL_SUBJECTS = [7, 8]
# Train subjects: [1, 2, 3, 4, 5, 6]
```

**Viktig**: 
- Begge acc og gyro anbefales som input for å unngå "juks" (siden gyro- og acc-targets er koblet via k_g)
- Standard: Begge datasett lastes og merges automatisk! Dette gir:
  - **Mer data** for trening (augmentation)
  - **Bedre balanse** (clean gir score=0, parkinson gir mixed scores)
  - **Mer realistisk** mix av tremor og ikke-tremor samples

### 2. Kjør trening

```bash
# Regresjon
python Tremor_head/tremor_regression_train.py

# Klassifikasjon
python Tremor_head/tremor_classification_train.py
```

### 3. Resultater

Resultater lagres i `tremor_logs/`:
- **Models**: `best_tremor_regression_model.pth` / `best_tremor_classification_model.pth`
- **Logs**: `tremor_regression_experiments.jsonl` / `tremor_classification_experiments.jsonl`
- **Plots**: `tremor_logs/regression_plots/` / `tremor_logs/classification_plots/`

## Regresjon-modellen detaljer

### Loss Function
Weighted Huber Loss (robust mot outliers):

```
L = λ_acc * Huber(acc_pred, acc_true) + 
    λ_gyro * Huber(gyro_pred, gyro_true) + 
    λ_freq * Huber(freq_pred, freq_true)
```

**Default vekter**:
- `λ_acc = 1.0`
- `λ_gyro = 1.0`
- `λ_freq = 0.2` (lavere for å unngå overfokus på frekvens)

### Target Transform
- RMS-verdier bruker `log1p(x) = log(1 + x)` for stabilitet
- Frekvens er lineær (3.5-7 Hz er et snilt område)

### Evalueringsmetrikker
- **MAE** (Mean Absolute Error) per target
- **R²** score per target
- **Spearman correlation** (håndterer ikke-lineære forhold)

## Klassifikasjons-modellen detaljer

### Loss Function
Valgfritt:
1. **Cross-Entropy** med class weights (default)
2. **Focal Loss** for å fokusere på vanskelige eksempler

```python
USE_CLASS_WEIGHTS = True   # Automatisk fra treningsdata
USE_FOCAL_LOSS = False     # Alternativ til CE
```

### Evalueringsmetrikker
- **Accuracy**
- **Macro F1** (viktig for ubalanserte klasser!)
- **Confusion Matrix** (sjekk score 2 vs 3 - ofte vanskeligst)
- **Per-class precision/recall/F1**

## Dataset Merging (Augmentation)

Standard er å laste **begge** datasettene samtidig:

```python
DATASET_VARIANTS = ["s2_w2_tremor_clean", "s2_w2_tremor_parkinson"]
```

Dette gir:
- **Dobbelt så mye data** for trening
- **Bedre klassebalanse**: clean-samplesene gir masse score=0, parkinson gir mixed
- **Realistisk fordeling**: Et ekte datasett ville ha både tremor og ikke-tremor

Datasettene lastes separat og merges før splitting, så train/val/test får en mix.

### Alternativ: Bruk bare ett datasett

Hvis du vil teste med bare ett datasett:
```python
# Bare parkinson
DATASET_VARIANTS = ["s2_w2_tremor_parkinson"]

# Bare clean (for testing)
DATASET_VARIANTS = ["s2_w2_tremor_clean"]
```

## Sanity Checks

### 1. Train på merged dataset, eval på held-out subjects
Modellen lærer forskjellen mellom tremor og ikke-tremor, og generaliserer til nye personer.

### 2. Test på bare clean samples (subset av test set)
Filtrer test set for samples med `y_tremor_score == 0` og sjekk at prediksjoner er nær null.

### 3. Test på bare parkinson samples med høy score
Filtrer test set for samples med `y_tremor_score >= 3` og sjekk at prediksjoner er høye.

## Preprosessering

### Normaliseringsstrategier

Normaliseringen kan konfigureres med `NORMALIZATION_MODE`:

```python
NORMALIZATION_MODE = "clean_only"  # "standard", "none", eller "clean_only"
```

**Tre alternativer:**

1. **`"none"` - INGEN normalisering (sanity check)**
   - Bruker absolutt amplitude direkte
   - Identity transform (mean=0, std=1)
   - **Formål**: Tester "øvre grense" for hvor mye signal som ligger i absolutt amplitude
   - Modellen bør lett separere score-klasser hvis tremor har sterk absolutt signatur

2. **`"standard"` - Standard normalisering**
   - Beregner mean/std fra ALL treningsdata (både clean og tremor)
   - Vanlig z-score normalisering
   - **Problem**: Stats "lærer tremor" og kan flate ut viktig signal

3. **`"clean_only"` - Normaliser med clean-train stats (ANBEFALT)**
   - Beregner mean/std KUN fra clean windows (score=0) i treningsdata
   - **Fordel**: Bevarer tremor som avvik fra normalen
   - Tremor-signal bevares bedre fordi stats ikke inkluderer tremor-variasjonen
   - Dette er ofte best for dette problemet!

### Standard innstillinger

- **Standardisering**: Per-kanal normalisering (avhenger av mode)
- **INGEN filtering**: For å bevare tremor-signalet
- **Input format**: (N, C, L) = (samples, channels, window_length)
  - Window length = 100 (2s @ 50Hz)
  - Channels depends på valgte sensorer (3 for Acc/Gyro/Mag, 2 for ECG)

## Tilgjengelige sensorer

Fra tremor-datasettet:
- `Acc_ankle`, `Acc_arm`, `Acc_chest`
- `Gyro_ankle`, `Gyro_arm`
- `Mag_ankle`, `Mag_arm`
- `ECG` (chest, 2 channels)

## Tips for eksperimentering

### Normaliserings-eksperimenter (viktig!)

**Test 1 - Sanity check:**
```python
NORMALIZATION_MODE = "none"
```
- Ingen normalisering, bruker absolutt amplitude
- Bør gi høy accuracy hvis tremor har sterk absolutt signatur
- Dette er "øvre grense" for ytelse basert på amplitude

**Test 2 - Standard baseline:**
```python
NORMALIZATION_MODE = "standard"
```
- Normaliserer med all treningsdata
- Kan flate ut tremor-signal hvis stats inkluderer tremor

**Test 3 - Anbefalt:**
```python
NORMALIZATION_MODE = "clean_only"
```
- Normaliserer kun med clean samples (score=0)
- Bevarer tremor som avvik fra normalen
- Ofte best for dette problemet!

**Sammenlign resultatene** for å se hvor mye normaliseringen påvirker tremordeteksjon.

### Arkitektur-tuning
- Øk CNN filters for mer kapasitet: `[64, 128, 128]`
- Øk kernel size for lengre temporal context: `kernel_size=7`
- Legg til flere hidden layers i hode
- Eksperimenter med dropout rates

### Loss-tuning (regresjon)
- Hvis freq dominerer: senk `LAMBDA_FREQ` til 0.1 eller 0.05
- Hvis acc/gyro er vanskelig: øk `LAMBDA_ACC` / `LAMBDA_GYRO`
- Hvis mange outliers: øk `HUBER_DELTA`

### Loss-tuning (klassifikasjon)
- Hvis klasseimbalanse: `USE_CLASS_WEIGHTS = True`
- Hvis vanskelige eksempler feiler: `USE_FOCAL_LOSS = True` med `FOCAL_GAMMA = 2.0`

### Data-tuning
- Test med bare Acc: `SENSORS = ["Acc_arm"]`
- Test med bare Gyro: `SENSORS = ["Gyro_arm"]`
- Test multi-sensor: `SENSORS = ["Acc_arm", "Gyro_arm", "Acc_ankle"]`
- Test random split for sanity check: `SPLIT_BY_SUBJECT = False`

## Neste steg

Når modellene er trent og evaluert:

1. **Sammenlign ytelse** mellom regresjon og klassifikasjon
2. **Analyser feiltilfeller** - hvilke samples er vanskelige?
3. **Test generalisering** på clean dataset
4. **Implementer i pipeline** hvis resultatene er lovende
5. **Eksperimenter med multi-task learning** (begge output i én modell)

## Dependencies

```python
# Required:
numpy
torch
scikit-learn
scipy
matplotlib

# Optional (for prettier confusion matrix plots):
seaborn
```

Hvis seaborn ikke er installert, brukes matplotlib fallback automatisk.

## Lisens og info

Basert på tremor-datagenerator og Parkinson tremor-simulering fra hovedprosjektet.
Se `/Volumes/NO NAME/Master Lina/Code/Documentation/TREMOR_Guide.md` for mer info.
