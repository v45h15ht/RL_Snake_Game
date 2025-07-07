# evaluate_dqn.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utilities.dqn_model import DQN
from utilities.environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_and_save_gif(model_path="policy_net.pth", save_path="snake_game.gif"):
    input_dim = 12
    output_dim = 4

    model = DQN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = Environment()
    game = env.game

    state = np.array(env.step(0)['observations'], dtype=np.float32)

    frames = []

    while True:
        with torch.no_grad():
            q_values = model(torch.tensor(state, device=device))
            action = q_values.argmax().item()

        step = env.step(action)
        state = np.array(step['observations'], dtype=np.float32) if step['observations'] else None

        # Build grid frame (2 = food, 1 = snake, 0 = empty)
        grid = np.zeros((game.length, game.width), dtype=np.uint8)
        for x, y in game.snake_list:
            grid[y, x] = 1
        fx, fy = game.food
        grid[fy, fx] = 2

        frames.append(grid.copy())

        if step['done']:
            print(f"Game over. Score: {game.score}")
            break

    # Create animation
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(frames[0], cmap='viridis', vmin=0, vmax=2)
    plt.axis('off')

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

    # 🔄 Save as GIF using pillow
    ani.save(save_path, writer='pillow')
    print(f"GIF saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_and_save_gif()
