import random
from collections import Counter
from enum import Enum
from pathlib import Path


class Feedback(Enum):
    ABSENT = 0
    PRESENT = 1
    CORRECT = 2


class Wordle:
    def __init__(self, wordlist: Path, wordlen: int, max_tries: int) -> None:
        self.wordlist = wordlist
        self.wordlen = wordlen
        self.max_tries = max_tries

        self._validate_inputs()
        self.words = self._load_words()
        self.answer = random.choice(self.words)

        self.tries_left = max_tries
        self.won = False

    def _validate_inputs(self) -> None:
        if not self.wordlist.exists():
            raise FileNotFoundError(f"Could not find {self.wordlist}")
        if not self.wordlist.is_file():
            raise IsADirectoryError(f"{self.wordlist} is not a file")

        if self.wordlen <= 0:
            raise ValueError("Word length must be greater than 0")
        if self.max_tries <= 0:
            raise ValueError("Number of tries must be at least 1")

    def _load_words(self) -> list[str]:
        words: list[str] = []

        with self.wordlist.open() as infile:
            for line in infile:
                word = line.strip().upper()
                if len(word) == self.wordlen:
                    words.append(word)

        if not words:
            raise ValueError("No words match the given word length")

        return words

    def guess(self, word: str) -> list[Feedback]:
        if self.won:
            raise RuntimeError("Game already won")
        if self.tries_left <= 0:
            raise RuntimeError("No tries left")

        word = word.upper().strip()

        if len(word) != self.wordlen:
            raise ValueError(f"Guess must be {self.wordlen} letters long")
        if word not in self.words:
            raise ValueError("Word not in word list")

        self.tries_left -= 1

        feedback = [Feedback.ABSENT] * self.wordlen
        answer_counts = Counter(self.answer)

        # First pass: correct positions
        for i, char in enumerate(word):
            if char == self.answer[i]:
                feedback[i] = Feedback.CORRECT
                answer_counts[char] -= 1

        # Second pass: present but wrong position
        for i, char in enumerate(word):
            if feedback[i] == Feedback.CORRECT:
                continue
            if answer_counts[char] > 0:
                feedback[i] = Feedback.PRESENT
                answer_counts[char] -= 1

        if word == self.answer:
            self.won = True

        return feedback
