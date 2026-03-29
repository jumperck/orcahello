# Fine-tuning Dataloader Design Brief

Design decisions and context for building the training dataloader (Phase 2), building on top of the HF datasets created by the dataset toolkit (Phase 1).

## Model & Audio Pipeline (InferenceSystem)

The existing inference model is a **PyTorch ResNet50 binary classifier** for SRKW call detection.

**Audio preprocessing pipeline** (`src/model/audio_frontend.py`):

```
Audio file
  → load + downmix mono + resample to 20kHz     (load_processed_waveform)
  → mel spectrogram (256 bins, 16kHz SR*)        (featurize_waveform)
  → pad/crop to 312 frames (4.0s)                (standardize)
  → model input: (1, 256, 312)
```

\*The mel spectrogram uses `sample_rate=16000` despite audio being at 20kHz — a quirk preserved from the original FastAI training for inference parity. Fine-tuning must preserve this.

**Key config values** (`InferenceSystem/model/config.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `audio.resample_rate` | 20000 | Waveform sample rate |
| `spectrogram.sample_rate` | 16000 | Mel filterbank SR (parity quirk) |
| `spectrogram.n_fft` | 2560 | |
| `spectrogram.hop_length` | 256 | |
| `spectrogram.mel_n_filters` | 256 | |
| `spectrogram.mel_f_max` | 10000.0 | |
| `model.input_pad_s` | 4.0 | Fixed input duration |
| Target frames | 312 | = `input_pad_s × resample_rate / hop_length` |

**Public functions to reuse** from `audio_frontend.py`:

- `featurize_waveform(waveform, sample_rate, spectrogram_config)` → mel spectrogram tensor
- `standardize(spectrogram, model_config, spectrogram_config)` → pad/crop to target frames

## Input: HF Datasets from Phase 1

The dataset toolkit (`ModelDevelopment/dataset_toolkit/`) produces two HF dataset levels. The dataloader operates on the **segment-level dataset**.

### Segment dataset schema (`segment_dataset/`)

| Column | Type | Description |
|--------|------|-------------|
| `audio` | `Audio()` | Segment waveform (Arrow-backed, memory-mapped) |
| `label` | `int64` | Binary: 0 (negative) / 1 (positive) |
| `tag` | `string` | `srkw_positive` or `srkw_negative` |
| `source_id` | `string` | Recording this segment came from |
| `start_s` | `float32` | Start time in source recording |
| `end_s` | `float32` | End time in source recording |

Segments vary in duration (2s–60s). Large segments are pre-broken into ≤`max_segment_s` chunks (default 10s) during dataset creation.

Audio preprocessing (resample to 20kHz, downmix, normalize) can be materialized into the dataset as a one-time step via `preprocess_audio_column()`, or left for the dataloader to handle at train time.

## Dataloader Design

### Architecture: `set_transform` + `DataLoader`

The dataloader uses HF datasets' `set_transform` for lazy, per-access transforms. This means:

- **Arrow-backed audio stays memory-mapped on disk** — working memory is just `batch_size × ~2MB`
- **Spectrogram is recomputed each epoch** — this is desirable because random crop gives different training views each time (implicit augmentation)
- **No caching** — `set_transform` results are never written to disk

```python
ds = load_from_disk("segment_dataset")
ds = ds.train_test_split(test_size=0.15, seed=42)

ds["train"].set_transform(make_transform(config, augmentations=train_aug))
ds["test"].set_transform(make_transform(config, augmentations=None))

train_loader = DataLoader(ds["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(ds["test"], batch_size=16)
```

### Transform pipeline

`set_transform` receives already-decoded audio (`{"array": np.ndarray, "sampling_rate": int}`) per batch. The transform function:

```
Decoded audio (numpy array, 20kHz mono)
  → random crop or zero-pad to target_duration_s
  → featurize_waveform()          # mel spectrogram
  → standardize()                 # pad/crop to exact 312 frames
  → optional augmentations        # on spectrogram tensor
  → return {"pixel_values": tensor, "label": tensor}
```

### Crop/pad strategy

Given a `target_duration_s` (e.g. 4.0s to match model input):

- **Segment longer than target**: random crop — pick a random offset within the segment
- **Segment shorter than target**: keep full segment — `standardize()` zero-pads to 312 frames

The random crop offset varies each access (each epoch), providing data augmentation for free.

### Augmentations (composable)

Augmentations operate on the spectrogram tensor `(1, 256, n_frames)` after `featurize_waveform`, composed via `torchvision.transforms.Compose` or equivalent:

```python
augmentations = Compose([
    SpecAugment(freq_mask=20, time_mask=40),   # torchaudio FrequencyMasking + TimeMasking
    GainJitter(max_db=4.0),                     # random dB offset
])
```

Train split gets augmentations; validation does not.

### Skeleton

```python
def make_transform(config, target_duration_s=4.0, augmentations=None):
    spec_cfg = config["spectrogram"]
    model_cfg = {**config["model"], "resample_rate": config["audio"]["resample_rate"],
                 "input_pad_s": target_duration_s}

    def transform(batch):
        features_list, labels_list = [], []
        for audio, label in zip(batch["audio"], batch["label"]):
            waveform = torch.from_numpy(audio["array"]).unsqueeze(0).float()
            sr = audio["sampling_rate"]

            # Random crop if longer than target, else keep full (standardize pads)
            segment = _sample_segment(waveform, sr, target_duration_s)

            features, _, _ = featurize_waveform(segment, sr, spec_cfg)
            features = standardize(features, model_cfg, spec_cfg)

            if augmentations:
                features = augmentations(features)

            features_list.append(features)
            labels_list.append(label)

        return {
            "pixel_values": torch.stack(features_list),
            "label": torch.tensor(labels_list, dtype=torch.long),
        }
    return transform
```

## Open questions for Phase 2 implementation

- **Waveform-level augmentations?** The current design augments spectrograms only. Time-domain augmentations (noise injection, time stretch) would go before `featurize_waveform`.
- **Class balancing**: should the dataloader handle class imbalance (e.g. `WeightedRandomSampler`), or is that handled upstream in dataset creation?
- **Multi-worker loading**: `set_transform` is compatible with `DataLoader(num_workers=N)` but the waveform cache (if any) needs to be process-safe.
- **Variable target duration**: the design supports a configurable `target_duration_s`. Should this match the model's `input_pad_s` (4.0s) exactly, or is there a use case for training with different window sizes?
