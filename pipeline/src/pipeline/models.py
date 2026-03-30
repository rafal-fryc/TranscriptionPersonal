from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start
