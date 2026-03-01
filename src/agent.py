from torch.nn import Linear, Sequential, ReLU
import torch

from structures import State


class Agent:
    def __init__(self, word_len: int, wordlist: list[str]):
        self.word_len = word_len
        self.wordlist = wordlist
        self.model = Sequential(
            Linear(26 + (26 * word_len), 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, len(self.wordlist)),
        )

    def select_action(self, state: State, valid_words: list[str]) -> str:
        with torch.no_grad():
            out = self.model(torch.tensor(state.to_vector(), dtype=torch.float32))
        valid_set = set(valid_words)
        mask = torch.tensor([0 if word in valid_set else -torch.inf for word in self.wordlist])
        masked_out = out + mask
        return self.wordlist[torch.argmax(masked_out).item()]