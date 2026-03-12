"""
Lightweight finetuning toolkit for OrcaHello SRKW Detector V1.

Reuses the existing model_v1 inference model and audio_frontend preprocessing
pipeline. No new heavy dependencies — pure PyTorch training on top of what's
already in requirements.txt.

Modules:
    dataset   - PyTorch Dataset that segments 1-min WAVs into model-ready spectrograms
    trainer   - Training loop with backbone freezing, LR scheduling, early stopping
    evaluate  - Per-segment and per-file metrics (precision, recall, F1, confusion matrix)
    finetune  - CLI entry point that ties everything together
"""
