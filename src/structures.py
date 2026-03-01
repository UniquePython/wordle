from enum import Enum

from wordle import Feedback


# Per letter: what do we know about this letter?
class LetterState(Enum):
    UNKNOWN = 0
    ABSENT = 1
    PRESENT = 2  # somewhere in word, position unclear


# For this specific (letter, position) pair, what do I know?
class PositionState(Enum):
    UNKNOWN = 0
    CONFIRMED = 1
    ELIMINATED = 2


# The full state
class State:
    letter_states: list[LetterState]  # length 26
    position_states: list[list[PositionState]]  # 26 x word_len

    def __init__(self, word_len: int):
        self.letter_states = [LetterState.UNKNOWN] * 26
        self.position_states = list(
            [PositionState.UNKNOWN] * word_len for _ in range(26)
        )

    def update(self, word: str, feedback: list[Feedback]) -> None:
        for (pos, ch), f in zip(enumerate(word), feedback, strict=True):
            idx = ord(ch) - ord("A")
            if not self.letter_states[idx] == LetterState.PRESENT:
                self.letter_states[idx] = (
                    LetterState.ABSENT if f == Feedback.ABSENT else LetterState.PRESENT
                )
            if f == Feedback.PRESENT:
                self.position_states[idx][pos] = PositionState.ELIMINATED
            if f == Feedback.CORRECT:
                self.position_states[idx][pos] = PositionState.CONFIRMED

    def to_vector(self) -> list[int]:
        return [ls.value for ls in self.letter_states] + [
            i.value for j in self.position_states for i in j
        ]
