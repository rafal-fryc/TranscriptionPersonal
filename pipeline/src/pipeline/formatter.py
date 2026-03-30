from pipeline.models import Segment


def format_transcript(segments: list[Segment]) -> str:
    """Format segments into a timestamped, speaker-labeled transcript."""
    sorted_segments = sorted(segments, key=lambda s: s.start)
    lines: list[str] = []
    for seg in sorted_segments:
        if not seg.text.strip():
            continue
        timestamp = _format_time(seg.start)
        lines.append(f"[{timestamp}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
