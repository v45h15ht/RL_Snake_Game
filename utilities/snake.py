class SnakeGame:
    def __init__(self, length=20, width=20):
        self.length = length
        self.width = width
        self.snake_list = [(self.length//2, self.width//2)]
        self.food = None
        self.direction = (1, 0)
        self.score = 0
        self.spawn_food()
    
    def spawn_food(self):
        import random
        while True:
            x = random.randint(0, self.length-1)
            y = random.randint(0, self.width-1)
            if (x, y) not in self.snake_list:
                self.food = (x, y)
                break
    
    def change_direction(self, new_direction):
        opp_dir = {
            (0, 1): (0, -1),
            (0, -1): (0, 1),
            (1, 0): (-1, 0),
            (-1, 0): (1, 0)
        }
        if new_direction != opp_dir[self.direction]:
            self.direction = new_direction
    
    def snake_move(self):
        head_x, head_y = self.snake_list[0]
        dir_x, dir_y = self.direction
        new_x, new_y = (head_x + dir_x, head_y + dir_y)

        if (new_x, new_y) == self.food:
            self.score += 1
            self.snake_list.insert(0, (new_x, new_y))
            self.spawn_food()
        else:
            self.snake_list.insert(0, (new_x, new_y))
            self.snake_list.pop()
    
    def is_game_over(self):
        head_x, head_y = self.snake_list[0]
        if head_x < 0 or head_y < 0 or head_x >= self.length or head_y >= self.width or len(self.snake_list) != len(set(self.snake_list)):
            return True
        return False

    def game_state(self):
        return {
            "score": self.score,
            "snake": self.snake_list,
            "food": self.food,
            "direction": self.direction,
            "game_over": self.is_game_over()
        }
    
    def render(self, delay=0.1):
        import matplotlib.pyplot as plt
        import numpy as np
        grid = np.zeros((self.length, self.width), dtype=np.uint8)

        for x, y in self.snake_list:
            grid[y, x] = 1  # Snake body

        fx, fy = self.food
        grid[fy, fx] = 2  # Food

        plt.imshow(grid, cmap='gray_r')
        plt.title(f"Score: {self.score}")
        plt.axis('off')
        plt.pause(delay)
        plt.clf()