class SnakeGame:
    def __init__(self, width=1280, height=720, cell_size=10):
        ...

    def reset(self):
        """Reset the game state and return the initial observation."""
        ...

    def step(self, action):
        """
        Perform an action.
        Returns: next_state, reward, done
        """
        ...

    def get_state(self):
        """Return a representation of the current state."""
        ...
