from environment import grid_world
from agent import AGENT


WORLD_HEIGHT = 5
WORLD_WIDTH = 10

env = grid_world(WORLD_HEIGHT,WORLD_WIDTH,
                 GOAL = [[WORLD_HEIGHT-1, WORLD_WIDTH-1]],
                 OBSTACLES=[[0,2], [1,2], [2,2], [0,4], [2,4], [4,4], [2,6], [3,6], [4,6], [2,7], [2,8]])

agent = AGENT(env,is_upload=False)
agent.deep_Q_learning(discount=1.0, alpha=0.01, max_seq_len=500, epsilon=0.2, epsilon_decay_period=50000, decay_rate=0.8)


