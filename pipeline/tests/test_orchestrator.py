from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.models import Segment
from pipeline.orchestrator import process_recording
from pipeline.config import PipelineConfig


class TestProcessRecording:
    def _make_test_recording(self, path: str) -> None:
        sr = 16000
        duration = 5.0
        samples = int(sr * duration)
        ch0 = np.random.randn(samples).astype(np.float32) * 0.1
        ch1 = np.random.randn(samples).astype(np.float32) * 0.1
        sf.write(path, np.column_stack([ch0, ch1]), sr)

    @patch("pipeline.orchestrator.ASRTranscriber")
    @patch("pipeline.orchestrator.SileroVAD")
    def test_should_produce_transcript_without_diarization(
        self,
        mock_vad_cls: MagicMock,
        mock_asr_cls: MagicMock,
        temp_dir: str,
    ) -> None:
        input_path = str(Path(temp_dir) / "test.wav")
        output_path = str(Path(temp_dir) / "transcript.txt")
        self._make_test_recording(input_path)

        mock_vad = mock_vad_cls.return_value
        mock_vad.detect_speech.return_value = [
            Segment(start=0.0, end=2.0, speaker="placeholder", text=""),
        ]

        mock_asr = mock_asr_cls.return_value
        mock_asr.transcribe_segments.side_effect = lambda path, segs, td: [
            Segment(start=s.start, end=s.end, speaker=s.speaker, text="mock text")
            for s in segs
        ]

        config = PipelineConfig(enable_diarization=False)
        result = process_recording(input_path, output_path, config)

        assert "mock text" in result
        assert Path(output_path).exists()
        assert "You" in Path(output_path).read_text()

    @patch("pipeline.orchestrator.ASRTranscriber")
    @patch("pipeline.orchestrator.SpeakerDiarizer")
    @patch("pipeline.orchestrator.SileroVAD")
    def test_should_use_diarizer_when_enabled(
        self,
        mock_vad_cls: MagicMock,
        mock_diarizer_cls: MagicMock,
        mock_asr_cls: MagicMock,
        temp_dir: str,
    ) -> None:
        input_path = str(Path(temp_dir) / "test.wav")
        output_path = str(Path(temp_dir) / "transcript.txt")
        self._make_test_recording(input_path)

        mock_vad = mock_vad_cls.return_value
        mock_vad.detect_speech.return_value = [
            Segment(start=0.0, end=2.0, speaker="placeholder", text=""),
        ]

        mock_diarizer = mock_diarizer_cls.return_value
        mock_diarizer.diarize.return_value = [
            Segment(start=0.5, end=3.0, speaker="SPEAKER_00", text=""),
        ]

        mock_asr = mock_asr_cls.return_value
        mock_asr.transcribe_segments.side_effect = lambda path, segs, td: [
            Segment(start=s.start, end=s.end, speaker=s.speaker, text="said something")
            for s in segs
        ]

        config = PipelineConfig(hf_token="fake-token", enable_diarization=True)
        result = process_recording(input_path, output_path, config)

        mock_diarizer_cls.assert_called_once_with("fake-token")
        mock_diarizer.diarize.assert_called_once()
        mock_diarizer.unload.assert_called_once()
        assert "SPEAKER_00" in result
