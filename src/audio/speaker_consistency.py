# src/audio/speaker_consistency.py

from pathlib import Path
from typing import List, Dict, Optional

import subprocess
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoFeatureExtractor, WavLMForXVector


# -----------------------------
# Global model (loaded once)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/wavlm-base-plus-sv"

FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
SPEAKER_MODEL = WavLMForXVector.from_pretrained(MODEL_NAME).to(DEVICE)
SPEAKER_MODEL.eval()


class SpeakerConsistencyAnalyzer:
    """
    Integrates your speaker-consistency logic into the project structure.

    Typical usage:
        analyzer = SpeakerConsistencyAnalyzer(tmp_dir="logs/audio_tmp")
        audio_summary = analyzer.analyze(audio_path)
    """

    def __init__(
        self,
        tmp_dir: str | Path = "tmp/audio",
        sample_rate: int = 16000,
        chunk_sec: int = 3,
        similarity_threshold: float = 0.85,
    ):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sample_rate
        self.chunk_sec = chunk_sec
        self.similarity_threshold = similarity_threshold

    # --------- audio I/O helpers ---------

    def _maybe_extract_audio(self, input_path: str) -> str:
        """
        If the input is already an audio file (e.g., .wav, .mp3),
        we just return it. If it's a video, we use ffmpeg to extract audio.
        """
        p = Path(input_path)
        ext = p.suffix.lower()

        # crude check: treat these as 'audio enough'
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        if ext in audio_exts:
            return str(p)

        # otherwise, extract audio with ffmpeg
        out_path = self.tmp_dir / "audio_extracted.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(p),
            "-ac", "1",
            "-ar", str(self.sr),
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        return str(out_path)

    def _load_audio(self, audio_path: str) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return audio

    def _split_into_chunks(
        self,
        audio: np.ndarray,
        min_fraction: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Splits audio into fixed-length chunks (chunk_sec).
        Drops very short tail chunks.
        """
        chunk_size = int(self.chunk_sec * self.sr)
        chunks: List[np.ndarray] = []
        n_samples = len(audio)

        for start in range(0, n_samples, chunk_size):
            end = start + chunk_size
            if end > n_samples:
                if (n_samples - start) < min_fraction * chunk_size:
                    break
                else:
                    end = n_samples
            chunk = audio[start:end]
            if len(chunk) > 0:
                chunks.append(chunk)

        return chunks

    # --------- embeddings ---------

    def _get_embedding_for_chunk(self, chunk: np.ndarray) -> np.ndarray:
        inputs = FEATURE_EXTRACTOR(
            [chunk],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = SPEAKER_MODEL(**inputs)
            emb = outputs.embeddings  # (1, D)
            emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb.squeeze(0).cpu().numpy()  # (D,)

    def _get_embeddings_for_chunks(self, chunks: List[np.ndarray]) -> np.ndarray:
        embs = []
        for ch in chunks:
            if len(ch) == 0:
                continue
            embs.append(self._get_embedding_for_chunk(ch))

        if not embs:
            raise ValueError("No valid audio chunks to embed.")

        return np.vstack(embs)  # (n_chunks, D)

    # --------- analysis logic ---------

    def _analyze_speaker_consistency(self, embeddings: np.ndarray) -> Dict:
        n_chunks = embeddings.shape[0]
        if n_chunks < 2:
            return {
                "num_chunks": int(n_chunks),
                "avg_similarity": None,
                "min_similarity": None,
                "speaker_change_flag": None,
                "speaker_consistency_score": None,
                "chunk_similarities": None,
                "warning": "Not enough chunks to analyze speaker consistency.",
            }

        # Use centroid instead of only chunk 0 as reference
        centroid = embeddings.mean(axis=0, keepdims=True)  # (1, D)
        sims = cosine_similarity(centroid, embeddings)[0]  # (n_chunks,)

        avg_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))

        speaker_change_flag = bool(min_sim < self.similarity_threshold)

        return {
            "num_chunks": int(n_chunks),
            "avg_similarity": avg_sim,
            "min_similarity": min_sim,
            "similarity_threshold": float(self.similarity_threshold),
            "speaker_change_flag": speaker_change_flag,
            # 0–1 score
            "speaker_consistency_score": avg_sim,
            "chunk_similarities": [float(x) for x in sims],
        }

    # --------- public API ---------

    def analyze(self, input_media_path: str) -> Dict:
        """
        High-level entrypoint:
          1. If needed, extract audio to mono 16k
          2. Load waveform
          3. Split into chunks
          4. Compute embeddings
          5. Compute speaker consistency metrics
        """
        try:
            audio_path = self._maybe_extract_audio(input_media_path)
        except Exception as e:
            return {
                "error": f"Failed to extract/load audio: {str(e)}",
                "num_chunks": 0,
                "speaker_consistency_score": None,
            }

        audio = self._load_audio(audio_path)
        if len(audio) == 0:
            return {
                "error": "Audio file is empty.",
                "num_chunks": 0,
                "speaker_consistency_score": None,
            }

        chunks = self._split_into_chunks(audio)
        if len(chunks) < 1:
            return {
                "error": "No audio content detected.",
                "num_chunks": 0,
                "speaker_consistency_score": None,
            }

        embeddings = self._get_embeddings_for_chunks(chunks)
        result = self._analyze_speaker_consistency(embeddings)
        return result
