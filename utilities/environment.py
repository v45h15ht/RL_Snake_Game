from utilities import snake

class Environment:
    def __init__(self):
        self.observation_space = 12
        self.action_space = 4
        self.game = snake.SnakeGame()

    def is_over(self, final_x, final_y, length, width, snake_list):
        if final_x < 0 or final_y < 0 or final_x >= length or final_y >= width or (final_x, final_y) in snake_list:
            return 1
        return 0

    def dangers(self):
        cur_state = self.game.game_state()
        snake_list = cur_state['snake']
        head_x, head_y = snake_list[0]
        length = self.game.length
        width = self.game.width

        up = self.is_over(head_x, head_y + 1, length, width, snake_list)
        down = self.is_over(head_x, head_y - 1, length, width, snake_list)
        left = self.is_over(head_x - 1, head_y, length, width, snake_list)
        right = self.is_over(head_x + 1, head_y, length, width, snake_list)

        return up, down, left, right

    def food_locations(self, food, snakeList):
        sx, sy = snakeList[0]
        fx, fy = food

        up = 1 if fy > sy else 0
        down = 1 if fy < sy else 0
        left = 1 if fx < sx else 0
        right = 1 if fx > sx else 0

        distance = (sx-fx)**2 + (sy-fy)**2

        return up, down, left, right, distance


    def step(self, action):
        direction = {
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0) 
        }
        self.game.change_direction(direction[action])
        snakeList = self.game.snake_list
        food = self.game.food
        foodDistance = self.food_locations(food, snakeList)[-1]

        self.game.snake_move()

        if self.game.game_state()['game_over']:
            return {
                "observations": None,
                "rewards": -10,
                "done": 1
            }

        newSnakeList = self.game.snake_list
        newFoodDistance = self.food_locations(food, newSnakeList)[-1]

        reward = 0
        if newFoodDistance == 0:
            reward = 10
        elif newFoodDistance < foodDistance:
            reward = 0.1
        else:
            reward = -0.1

        obs_direction = {
            0: (1, 0, 0, 0),
            1: (0, 0, 0, 1),
            2: (0, 1, 0, 0),
            3: (0, 0, 1, 0)
        }

        danger_up, danger_down, danger_left, danger_right = self.dangers()
        dir_up, dir_down, dir_left, dir_right = obs_direction[action]
        food_up, food_down, food_left, food_right, _ = self.food_locations(food, newSnakeList)

        observations = [danger_up, danger_down, danger_left, danger_right, dir_up, dir_down, dir_left, dir_right, food_up, food_down, food_left, food_right]
        rewards = reward
        done = 0

        return {
            "observations": observations,
            "rewards": rewards,
            "done": done
        }


