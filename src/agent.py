from torch.nn import Linear, Sequential, ReLU
import torch

from structures import State
from pathlib import Path


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def select_action(self, state: State, valid_words: list[str], training: bool = False) -> tuple[str, torch.Tensor] | str:
        if not valid_words:
            valid_words = self.wordlist
        
        logits = self.model(
            torch.tensor(state.to_vector(), dtype=torch.float32).to(self.device)
        )

        valid_set = set(valid_words)
        mask = torch.tensor(
            [0.0 if word in valid_set else -torch.inf for word in self.wordlist],
            dtype=torch.float32,
        ).to(self.device)

        masked_logits = logits + mask

        if not training:
            idx = torch.argmax(masked_logits).item()
            return self.wordlist[idx]

        dist = torch.distributions.Categorical(logits=masked_logits)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)

        return self.wordlist[idx.item()], log_prob
    
    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))