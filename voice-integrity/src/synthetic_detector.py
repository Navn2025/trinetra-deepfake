import argparse

import torch
import soundfile as sf
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

MODEL_NAME = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

_processor = None
_model = None
_device = None


def load_model(device: str | None = None):
    """Loaded once and reused across predict() calls."""
    global _processor, _model, _device
    if _model is None:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # AutoProcessor would try to build a full Wav2Vec2Processor (feature
        # extractor + CTC tokenizer, meant for speech-to-text) and crash on
        # some repos -- this one ships a plain preprocessor_config.json
        # (Wav2Vec2FeatureExtractor), no vocab/tokenizer files, since it's a
        # classifier with no text output. AutoFeatureExtractor sidesteps that.
        _processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _processor, _model


def predict(audio_path) -> dict[str, float]:
    """Returns {label: probability} for every class the model reports (this
    checkpoint: "real"/"fake"). Input must be 16kHz mono."""
    audio_input, sample_rate = sf.read(str(audio_path))
    return predict_array(audio_input, sample_rate)


def predict_array(audio_input, sample_rate: int) -> dict[str, float]:
    """Same as predict(), but for audio already in memory -- skips the
    write-to-disk-then-read-back round trip callers would otherwise pay
    (server.py's segment scoring calls this directly for exactly that
    reason: 12 segments x 2 models x a disk round trip each was real,
    measurable overhead for clips that are already fully in memory)."""
    processor, model = load_model()
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)  # Convert to mono if stereo
    if sample_rate != 16000:
        raise ValueError(f"Expected sample rate of 16000 Hz, but got {sample_rate} Hz.")
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    return {
        model.config.id2label.get(i, f"label_{i}"): p.item()
        for i, p in enumerate(probabilities)
    }


def predict_batch(windows: list[tuple]) -> list[dict[str, float]]:
    """Same result as calling predict_array() once per window, but as one
    batched forward pass instead of N sequential ones -- see
    xlsr_sls_detector.predict_batch()'s docstring for why this (not just
    removing I/O overhead) is the real fix for segment-scoring cost.
    processor() already batches a list of raw arrays with padding, so this
    is a straightforward extension of predict_array(). Empty list returns
    []."""
    if not windows:
        return []
    processor, model = load_model()
    arrays = []
    for audio_input, sample_rate in windows:
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected sample rate of 16000 Hz, but got {sample_rate} Hz.")
        arrays.append(audio_input)

    inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)  # (N, num_labels)

    return [
        {model.config.id2label.get(i, f"label_{i}"): p.item() for i, p in enumerate(row)}
        for row in probabilities
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Gustking wav2vec2-XLSR deepfake audio classifier (fast, in-domain accuracy ~93%)."
    )
    parser.add_argument("audio_file", help="Path to a 16kHz mono WAV file to analyze")
    args = parser.parse_args()

    print("Loading the pre-trained model...")
    load_model()
    print(f"Model Loaded. Analyzing audio file: {args.audio_file}")

    labels = predict(args.audio_file)
    print("Model Labels")
    for label, probability in labels.items():
        print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
