import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from network import MLP
import random
import collections
from tqdm import tqdm


from visualize_train import draw_value_image, draw_policy_image


# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]

TRAINING_EPISODE_NUM = 20000
buffer_limit  = 50000
batch_size    = 64


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_flag_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_flag = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_flag_lst.append([done_flag])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_flag_lst)

    def size(self):
        return len(self.buffer)


class AGENT:
    def __init__(self, env, is_upload=False):

        # **************************************************
        #                Initialize
        # **************************************************

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = self.env.size()
        self.state = [0,0]
        self.device = 'cpu'  # if torch.cuda.is_available() else 'cpu'
        self.memory = ReplayBuffer()


        # **************************************************
        #                Network
        # **************************************************
        print('==> Building model..')
        self.Q_net = MLP()
        self.Q_net = self.Q_net.to(self.device)
        self.Q_net_target = MLP()
        if is_upload:
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            weight = torch.load('./checkpoint/ckpt.pth')
            self.Q_net.load_state_dict(weight)
        else:
            pass
        self.Q_net_target.load_state_dict(self.Q_net.state_dict())

        # Loss, Optimizer, Scheduler setting
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.Q_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)



    def initialize_episode(self):
        HEIGHT, WIDTH = self.env.size()
        while True:
            i = np.random.randint(HEIGHT)
            j = np.random.randint(WIDTH)
            state = [i, j]
            if (state in self.env.goal) or (state in self.env.obstacles):
                continue
            break

        return state



    def deep_Q_learning(self, discount=1.0, alpha=0.01, max_seq_len=500, epsilon=0.1, epsilon_decay_period=20000,decay_rate=0.9):

        HEIGHT, WIDTH = self.env.size()
        V_values = np.zeros((HEIGHT, WIDTH))
        train_loss = 0
        reward_sum = 0
        num_of_timeout_episode = 0
        num_of_valid_episode = 0


        for episode in tqdm(range(TRAINING_EPISODE_NUM)):
            state = self.initialize_episode()
            done = False
            timeout = False
            seq_len = 0.
            history = []

            #  Episode generation
            while not (done or timeout):
                # Next state and action generation
                action = self.get_action(state, epsilon)
                movement = ACTIONS[action]
                next_state, reward, done = self.env.interaction(state, movement)
                done_flag = 0.0 if done else 1.0
                history.append((state, action, reward, next_state, done_flag))
                state = next_state
                seq_len += 1
                if (seq_len >= max_seq_len):
                    timeout = True


            if timeout:
                num_of_timeout_episode += 1
            else:    # ifnot timeout:
                num_of_valid_episode += 1
                for transition in history[::-1]:
                    state, action, reward, next_state, done_flag = transition
                    self.memory.put((state, action, reward, next_state, done_flag))
                    reward_sum += reward


            # MLP learning
            if self.memory.size() > 2000:
                for i in range(100): # 학습 100번 반복
                    #***********************************************************
                    s, a, r, s_prime, done_flag = self.memory.sample(batch_size) # ReplayBuffer의 sample function을 이용해 minibatch 생성
                    q_out = self.Q_net(s) # current state s에서의 Q 값 추출
                    output = q_out.gather(1, a) # 선택한 action a에 대한 Q 값 추출
                    # next state s'에서 최대가 되는 Q 값을 계산하고 차원을 맞추기 위해 unsqueeze
                    max_q_prime = self.Q_net_target(s_prime).max(1)[0].unsqueeze(1)
                    target = r + discount * max_q_prime * done_flag # target Q 값: R + discout 된 최대 Q 값 * 종료 여부
                    
                    self.optimizer.zero_grad() # optimizer gradient 초기화
                    loss = self.criterion(output, target) # predicted Q 값과 target Q 값 간의 loss 계산
                    loss.backward() # loss를 이용해 backward gradient 계산
                    self.optimizer.step() # optimizer를 이용해 weight update
                    train_loss += loss.item() # 실험 후 train loss log를 보기 위해 loss 저장
                    #***********************************************************


            if episode % 500 == 0:
                print("")
                print("Num of episodes={:}, Num of timeout episodes = {:d}, Num of valid episodes = {:d}, epsilon={:.4f}, "
                      "loss={:.4f}, return for each episode={:.4f}".format(episode,num_of_timeout_episode,
                        num_of_valid_episode,epsilon, train_loss, reward_sum/500))

                for param_group in self.optimizer.param_groups:
                    print('Current learning rate is {:f}'.format(param_group['lr']))

                # Reset for next monitoring interval
                train_loss = 0
                reward_sum = 0
                num_of_timeout_episode = 0
                num_of_valid_episode = 0
                self.Q_net_target.load_state_dict(self.Q_net.state_dict())

                for i in range(HEIGHT):
                    for j in range(WIDTH):
                        V_values[i, j] = np.max(self.Q_net(torch.tensor([i, j], dtype=torch.float)).detach().numpy())

                draw_value_image(1, V_values, self.env)



            if episode % epsilon_decay_period == 0:
               epsilon *= decay_rate


        weight = self.Q_net.state_dict()
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(weight, './checkpoint/ckpt.pth')

        return V_values



    def get_action(self, state, epsilon):  # epsilon-greedy
        Q = self.Q_net(torch.tensor(np.array(state))).detach().numpy()
        if np.random.rand() < epsilon:
            action = np.random.choice(len(ACTIONS))
        else:
            action = np.argmax(Q)
        return action


    def get_V_value(self, state):
        Q = self.Q_net(torch.tensor(np.array(state))).detach().numpy()
        return np.max(Q)
