from pipeline.models import Segment


class TestSegment:
    def test_should_store_fields(self) -> None:
        seg = Segment(start=1.5, end=4.0, speaker="You", text="hello world")
        assert seg.start == 1.5
        assert seg.end == 4.0
        assert seg.speaker == "You"
        assert seg.text == "hello world"

    def test_should_calculate_duration(self) -> None:
        seg = Segment(start=10.0, end=25.5, speaker="A", text="")
        assert seg.duration == 15.5

    def test_duration_should_be_zero_for_empty_segment(self) -> None:
        seg = Segment(start=5.0, end=5.0, speaker="A", text="")
        assert seg.duration == 0.0
