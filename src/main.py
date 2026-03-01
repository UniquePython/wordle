from collections import Counter
from wordle import Feedback, Wordle
from agent import Agent
from structures import State
from pathlib import Path
import torch


def is_consistent(word: str, guess: str, feedback: list[Feedback]) -> bool:
    answer_counts = Counter(
        ch for ch, f in zip(guess, feedback)
        if f in (Feedback.CORRECT, Feedback.PRESENT)
    )
    
    for (pos, ch), f in zip(enumerate(guess), feedback, strict=True):
        if f == Feedback.CORRECT:
            if word[pos] != ch:
                return False
        if f == Feedback.PRESENT:
            if ch not in word or word[pos] == ch:
                return False
        if f == Feedback.ABSENT:
            if word.count(ch) > answer_counts[ch]:
                return False
    return True


def filter_valid_words(words: list[str], guess: str, feedback: list[Feedback]) -> list[str]:
    valid = []
    for word in words:
        if is_consistent(word, guess, feedback):
            valid.append(word)
    return valid


def compute_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

total_guesses = 0

def train(agent: Agent, wordlist: Path, word_len: int, max_tries: int, episodes: int):
    global total_guesses
    for episode in range(episodes):
        wordle = Wordle(wordlist, word_len, max_tries)
        state = State(word_len)
        valid_words = wordle.words.copy()
        log_probs = []
        rewards = []

        while not wordle.won and wordle.tries_left > 0:
            word, log_prob = agent.select_action(state, valid_words, True)
            feedback = wordle.guess(word)

            # reward = how many words were eliminated
            words_before = len(valid_words)
            valid_words = filter_valid_words(valid_words, word, feedback)
            if not valid_words:
                valid_words = wordle.words.copy()
            words_after = len(valid_words)
            reward = (words_before - words_after) / len(wordle.words)

            # bonus for winning
            if wordle.won:
                reward += 1.0

            state.update(word, feedback)
            log_probs.append(log_prob)
            rewards.append(reward)

        # update weights
        returns = compute_returns(rewards)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = sum(-lp * G for lp, G in zip(log_probs, returns))

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        
        guesses_taken = max_tries - wordle.tries_left
        total_guesses += guesses_taken
        
        if episode % 100 == 0:
            avg = total_guesses / (episode + 1)
            print(f"Episode {episode}, guesses: {guesses_taken}, running avg: {avg:.2f}, won: {wordle.won}")


def play(agent: Agent, wordlist: Path, word_len: int = 5, max_tries: int = 6) -> None:
    wordle = Wordle(wordlist, word_len, max_tries)
    state = State(word_len)
    valid_words = wordle.words.copy()

    print(f"Answer: {wordle.answer}\n")

    while not wordle.won and wordle.tries_left > 0:
        word = agent.select_action(state, valid_words)
        feedback = wordle.guess(word)

        symbols = {Feedback.ABSENT: "⬛", Feedback.PRESENT: "🟨", Feedback.CORRECT: "🟩"}
        print(f"Guess: {word}  {''.join(symbols[f] for f in feedback)}")

        valid_words = filter_valid_words(valid_words, word, feedback)
        if not valid_words:
            valid_words = wordle.words.copy()
        state.update(word, feedback)

    if wordle.won:
        print(f"\nSolved in {max_tries - wordle.tries_left} guesses")
    else:
        print(f"\nFailed. Answer was {wordle.answer}")


def main() -> None:
    words = Wordle.load_words(Path("wordle-La.txt"), 5)
    agent = Agent(5, words)
    
    model_path = Path("wordle_agent.pt")
    if model_path.exists():
        agent.load(model_path)
        print("Loaded existing model")
    
    train(agent, Path("wordle-La.txt"), 5, 6, 5000)
    agent.save(model_path)
    print("Model saved")
    
    for i in range(5):
        print(f"\n===Game {i}===\n")
        play(agent, Path("wordle-La.txt"))


if __name__ == "__main__":
    main()
