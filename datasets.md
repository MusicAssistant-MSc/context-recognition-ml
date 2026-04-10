# Zbiory danych - przegląd

## Zbiory utworów muzycznych lub MER
### 1. Zbiory wykorzystane w publikacjach z SLR
#### 1.1. MillionSongsDataset
TODO: add description

#### 1.2. MusicBrainzAPI
TODO: add description

#### 1.3. AcousticBrainzAPI
TODO: add description

#### 1.4. Spotify 12m songs (Kaggle)
TODO: add description

#### 1.5. CAL500
Zbiór zawiera 500 plików MP3 z pełnymi utworami oraz 174 rodzaje labeli (kategoriami emocjonalnymi).

#### 1.6. NTWICM
Zbiór zawiera 2648 plików MP3 z pełnymi utworami. Utwory oznaczone są emocjonalnie za pomocą "dyskretnej" wymiarowości.

#### 1.7. Soundtracks
360 wycinków utworów (po 15s-1min, format MP3) wraz z kategorycznym przypisaniem emocji (tension, anger, fear, happy, sad, tender) oraz wymiarowym (valence, energy, tension).

#### 1.8. DEAP
120 wycinków utworów (po 60s, linki do utworów). Emocje reprezentowane wymiarowo (AV, upodobanie, znajomość utworu). Dołączone odczyty biometryczne od 32 uczestników: EEG, ECG, BVP (Blood Volume Pulse, mierzony sensorem PPG), EMG (aktywność elektryczna mięści), EDA (GSR). Wszystkie odczyty mierzone z częstotliwością 512 Hz.

#### 1.9. AMG1608
1608 wycinków utworów (po 30s, format WAV). Utwory oznaczone są emocjonalnie za pomocą "dyskretnej" wymiarowości.

#### 1.10. Emotify
400 wycinków utworów (po 60s, format MP3). Utwory oznaczone są emocjonalnie za pomocą "dyskretnej" wymiarowości.

#### 1.11. DEAM
2058 wycinków utworów (po 45s, format MP3). Utwory oznaczone emocjonalnie za pomocą rozłożonej w czasie wymiarowości VA.

#### 1.12. PMEmo
794 pełnych utworów w formacie MP3. Utwory oznaczone emocjonalnie za pomocą rozłożonej w czasie wymiarowości VA. Dodatkowo do datasetu dołączone odczyty GSR i HR.

#### 1.12. RAVDESS
1012 utworów. Utwory oznaczone kategoriami emocjonalnymi wraz z ich intensywnością.

#### 1.13. EmoMusic

#### 1.14. Spotify API [DEPRECATED]

#### Inne datasety
InCarMusic, GTZAN, Music4All, MTurk, MER60, MediaEval Emotion, AMG240, EMOTIONS, EMOPIA


### 2. Zbiory z innych źródeł
## Zbiory danych biometrycznych
### 1. Zbiory wykorzystane w publikacjach z SLR
#### PAMAP2
TODO: add description

#### PMEmo
TODO: add description

#### MUMR
Dataset utworzony w ramach publikacji, eksperyment przeprowadzony na 30 użytkownikach i na podzbiorze Million Song Dataset oraz Taste Profile Dataset. Zbierane były odtwarzane utwory przez danego użytkownika, jego explicit ratingi VA, czas, pogoda, GPS, HR, kroki i rodzaj aktywności.

Źródło:
https://github.com/JiayuLi-997/MUMR-Ubiquitous_Recommendation

#### WISDM
ponad 1 mln rekordów z userem, labelem aktywności i danymi z 3 osi akcelerometru

#### WESAD
Wearable Stress and Affect Detection
Zebrany od 15 uczestników, eksperyment opierał się na odtwarzaniu wideo o różnym poziomie stresu. Zebrano tętno, ECG, EDA, EMG, temperaturę i dane z akcelerometru (urządzenie Empatica E4). Odczytom przypisano etykiety z poziomem stresu/podekscytowania oraz VA.

#### DEAP
120 wycinków utworów (po 60s, linki do utworów). Emocje reprezentowane wymiarowo (AV, upodobanie, znajomość utworu). Dołączone odczyty biometryczne od 32 uczestników: EEG, ECG, BVP (Blood Volume Pulse, mierzony sensorem PPG), EMG (aktywność elektryczna mięści), EDA (GSR). Wszystkie odczyty mierzone z częstotliwością 512 Hz.

#### SBRHAPTDS
Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set

#### Inne datasety
FER2013, AffectNet, Sentiment140


### 2. Zbiory z innych źródeł
#### MANHOB-HCI
Eksperyment (odtwarzanie wideo) na 32 uczestnikach, zbierane VA oraz etykieta emocji. Dołączone odczyty biometryczne EEG, ECG i EDA (GSR), wszystkie zbierane z częstotliwością 1024 Hz.

#### DECAF
Eksperyment (odtwarzanie wideo i utworów muzycznych ze zbioru DEAP) na 30 uczestnikach, zbierane VA. Dołączone odczyty biometryczne MEG, ECG i EOG, wszystkie zbierane z częstotliwością 1000 Hz.

#### ASCERTAIN
Eksperyment (odtwarzanie wideo) na 58 uczestnikach, zbierane VA, zaangażowanie i upodobanie. Dołączone odczyty biometryczne EEG, ECG i EDA.

#### DAPPER
Eksperyment przeprowadzony w naturalnych warunkach na 88 uczestnikach. Zbierano informacje o kontekście (lokalizacja, środowisko, typ aktywności, wyniki testu charakteru TIPI, VA). Do danych dołączono dane biometryczne: HR, akcelerometr (20 Hz), EDA (GSR, 40 Hz), zbierane z własnoręcznie skonstruowanej opaski.

#### ECSMP
Eksperyment przeprowadzony na 73 uczestnikach. Zbierano od nich wywiad w postaci kwestionariuszy ERQ, SDS, POMS, PSQI i CESAF. Dołączono odczyty biometryczne: EEG, ECG, PPG, EDA (GSR), temperaturę i dane z akcelerometru.

#### EmoWear
Eksperyment przeprowadzony na 49 uczestnikach. Zbierano VA i upodobanie podczas oglądania 38 1-minutowych wideo. Dołączono odczyty biometryczne: ECG, EDA (GSR), dane z akcelerometru i żyroskopu.

#### Inne datasety
BIRAFFE, CASE, SWELL-KW, MPED
