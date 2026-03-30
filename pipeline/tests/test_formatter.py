from pipeline.formatter import format_transcript, _format_time
from pipeline.models import Segment


class TestFormatTime:
    def test_zero(self) -> None:
        assert _format_time(0.0) == "00:00:00"

    def test_seconds_only(self) -> None:
        assert _format_time(45.0) == "00:00:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_time(125.0) == "00:02:05"

    def test_hours(self) -> None:
        assert _format_time(3661.0) == "01:01:01"

    def test_fractional_seconds_truncate(self) -> None:
        assert _format_time(5.9) == "00:00:05"


class TestFormatTranscript:
    def test_single_segment(self) -> None:
        segments = [Segment(start=5.0, end=8.0, speaker="You", text="hello world")]
        result = format_transcript(segments)
        assert result == "[00:00:05] You: hello world"

    def test_should_sort_by_start_time(self) -> None:
        segments = [
            Segment(start=10.0, end=12.0, speaker="Other", text="second"),
            Segment(start=2.0, end=5.0, speaker="You", text="first"),
        ]
        result = format_transcript(segments)
        lines = result.split("\n")
        assert lines[0] == "[00:00:02] You: first"
        assert lines[1] == "[00:00:10] Other: second"

    def test_should_skip_empty_text(self) -> None:
        segments = [
            Segment(start=0.0, end=1.0, speaker="You", text=""),
            Segment(start=2.0, end=3.0, speaker="Other", text="hello"),
        ]
        result = format_transcript(segments)
        assert result == "[00:00:02] Other: hello"

    def test_should_skip_whitespace_only_text(self) -> None:
        segments = [
            Segment(start=0.0, end=1.0, speaker="You", text="   "),
            Segment(start=2.0, end=3.0, speaker="Other", text="hello"),
        ]
        result = format_transcript(segments)
        assert result == "[00:00:02] Other: hello"

    def test_empty_segments_list(self) -> None:
        result = format_transcript([])
        assert result == ""

    def test_multiple_speakers(self) -> None:
        segments = [
            Segment(start=0.0, end=3.0, speaker="SPEAKER_00", text="hello"),
            Segment(start=3.5, end=6.0, speaker="You", text="hi there"),
            Segment(start=6.5, end=9.0, speaker="SPEAKER_01", text="welcome"),
        ]
        result = format_transcript(segments)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "SPEAKER_00" in lines[0]
        assert "You" in lines[1]
        assert "SPEAKER_01" in lines[2]
