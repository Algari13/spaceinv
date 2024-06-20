import random
import time
import turtle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib.pyplot as plt

# Parámetros de la IA
EPSILON_START = 1.0  # Exploration rate inicial
EPSILON_END = 0.01   # Exploration rate final
EPSILON_DECAY = 0.995  # Decaimiento de epsilon
GAMMA = 0.9    # Discount factor
ALPHA = 0.001  # Learning rate
BATCH_SIZE = 64
TARGET_UPDATE_INTERVAL = 1  # Intervalo de actualización de la red objetivo
NUM_EPISODES = 10000  # Número de episodios de entrenamiento

# Definir el dispositivo para PyTorch (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros del juego
FRAME_RATE = 30
TIME_FOR_1_FRAME = 1 / FRAME_RATE
CANNON_STEP = 10
LASER_LENGTH = 20
LASER_SPEED = 20
ALIEN_SPAWN_INTERVAL = 1.2
ALIEN_SPEED = 3.5

# Inicializar la pantalla del juego
window = turtle.Screen()
window.tracer(0)
window.setup(0.5, 0.75)
window.bgcolor(0.2, 0.2, 0.2)
window.title("The Real Python Space Invaders")

LEFT = -window.window_width() / 2
RIGHT = window.window_width() / 2
TOP = window.window_height() / 2
BOTTOM = -window.window_height() / 2
FLOOR_LEVEL = 0.9 * BOTTOM
GUTTER = 0.025 * window.window_width()

# Crear cañón láser
cannon = turtle.Turtle()
cannon.penup()
cannon.color(1, 1, 1)
cannon.shape("square")
cannon.setposition(0, FLOOR_LEVEL)
cannon.cannon_movement = 0  # -1, 0 or 1 for left, stationary, right

# Crear turtle para escribir texto
text = turtle.Turtle()
text.penup()
text.hideturtle()
text.setposition(LEFT * 0.8, TOP * 0.8)
text.color(1, 1, 1)

# Variables globales para gráficos
scores = []
epsilons = []

def draw_cannon():
    cannon.clear()
    cannon.turtlesize(1, 4)  # Base
    cannon.stamp()
    cannon.sety(FLOOR_LEVEL + 10)
    cannon.turtlesize(1, 1.5)  # Next tier
    cannon.stamp()
    cannon.sety(FLOOR_LEVEL + 10)
    cannon.turtlesize(0.8, 0.3)  # Tip of cannon
    cannon.stamp()
    cannon.sety(FLOOR_LEVEL)

def create_laser():
    global laser
    if laser is None:
        laser = turtle.Turtle()
        laser.penup()
        laser.color(1, 0, 0)
        laser.hideturtle()
        laser.setposition(cannon.xcor(), cannon.ycor())
        laser.setheading(90)
        # Move laser to just above cannon tip
        laser.forward(20)
        # Prepare to draw the laser
        laser.pendown()
        laser.pensize(5)
        laser.showturtle()

def move_laser():
    global laser
    if laser:
        laser.clear()
        laser.forward(LASER_SPEED)
        # Draw the laser
        laser.forward(LASER_LENGTH)
        laser.forward(-LASER_LENGTH)
        if laser.ycor() > TOP:
            remove_laser()

def remove_laser():
    global laser
    if laser:
        laser.clear()
        laser.hideturtle()
        laser = None

def create_alien():
    alien = turtle.Turtle()
    alien.penup()
    alien.turtlesize(1.5)
    alien.setposition(
        random.randint(
            int(LEFT + GUTTER),
            int(RIGHT - GUTTER),
        ),
        TOP,
    )
    alien.shape("turtle")
    alien.setheading(-90)
    alien.color(random.random(), random.random(), random.random())
    aliens.append(alien)

def remove_sprite(sprite, sprite_list):
    sprite.clear()
    sprite.hideturtle()
    window.update()
    sprite_list.remove(sprite)
    turtle.turtles().remove(sprite)

class ConvDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_dim[1] * input_dim[2], 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = torch.tensor(batch[0], dtype=torch.float32).to(device)
        actions = torch.tensor(batch[1], dtype=torch.long).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_states = torch.tensor(batch[3], dtype=torch.float32).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float32).to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

class Agent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.lr = ALPHA
        self.memory = PrioritizedReplayBuffer(10000)

        self.policy_net = ConvDQN(state_space, len(action_space)).to(device)
        self.target_net = ConvDQN(state_space, len(action_space)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        beta = min(1.0, 0.4 + (1.0 - 0.4) * (episode / NUM_EPISODES))
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE, beta=beta)

        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = (weights * nn.functional.mse_loss(q_values, target_q_values, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, loss.detach().cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def get_state():
    state = np.zeros((3, int(window.window_height()), int(window.window_width())))
    cannon_x = int(cannon.xcor() - LEFT)
    state[0, int(FLOOR_LEVEL):int(FLOOR_LEVEL)+10, cannon_x:cannon_x+10] = 1
    for alien in aliens:
        alien_x = int(alien.xcor() - LEFT)
        alien_y = int(TOP - alien.ycor())
        state[1, alien_y:alien_y+10, alien_x:alien_x+10] = 1
    if laser:
        laser_x = int(laser.xcor() - LEFT)
        laser_y = int(TOP - laser.ycor())
        state[2, laser_y:laser_y+LASER_LENGTH, laser_x:laser_x+5] = 1
    return state

def get_reward(alien_hit, done, closest_alien):
    if done:
        return -10
    if alien_hit:
        return 30 + 5 * (1 - closest_alien.ycor() / (TOP - FLOOR_LEVEL))
    return -1

def reset_environment():
    global laser, aliens
    if laser:
        remove_laser()
    for alien in aliens.copy():
        remove_sprite(alien, aliens)
    aliens = []
    cannon.setposition(0, FLOOR_LEVEL)
    draw_cannon()

# Inicializar el agente
state_space_dim = (3, int(window.window_height()), int(window.window_width()))  # [3 channels, height, width]
action_space = [0, 1, 2]  # 0: Izquierda, 1: Derecha, 2: Disparar
agent = Agent(state_space_dim, action_space)

# Inicializar variables globales
laser = None
aliens = []

# Bucle de entrenamiento
scores = []  # Lista para almacenar los puntajes por episodio
epsilons = []  # Lista para almacenar los valores de epsilon por episodio

for episode in range(NUM_EPISODES):
    reset_environment()
    
    alien_timer = 0
    game_timer = time.time()
    score = 0
    game_running = True
    last_shoot_time = 0

    while game_running:
        timer_this_frame = time.time()

        time_elapsed = time.time() - game_timer
        text.clear()
        text.write(
            f"Time: {time_elapsed:5.1f}s\nScore: {score:5}",
            font=("Courier", 20, "bold"),
        )

        current_state = get_state()
        action = agent.choose_action(current_state)

        if action == 0 and LEFT + GUTTER < cannon.xcor() - CANNON_STEP:
            cannon.setx(cannon.xcor() - CANNON_STEP)
        elif action == 1 and cannon.xcor() + CANNON_STEP < RIGHT - GUTTER:
            cannon.setx(cannon.xcor() + CANNON_STEP)
        elif action == 2 and time.time() - last_shoot_time > 0.5:
            create_laser()
            last_shoot_time = time.time()
        
        draw_cannon()

        alien_hit = False  # Inicializar alien_hit
        move_laser()
        for alien in aliens.copy():
            if laser and laser.distance(alien) < 20:
                remove_laser()
                remove_sprite(alien, aliens)
                score += 1
                alien_hit = True  # Marcar que el alienígena fue golpeado
                break

        if time.time() - alien_timer > ALIEN_SPAWN_INTERVAL:
            create_alien()
            alien_timer = time.time()

        for alien in aliens.copy():
            alien.sety(alien.ycor() - ALIEN_SPEED)
            if alien.ycor() < FLOOR_LEVEL:
                game_running = False
                break

        next_state = get_state()
        closest_alien = min(aliens, key=lambda alien: alien.ycor()) if aliens else None
        reward = get_reward(alien_hit, not game_running, closest_alien)  # Pasar alien_hit como argumento
        done = not game_running

        agent.remember(current_state, action, reward, next_state, done)
        agent.learn()

        if done:
            break

        window.update()

        time.sleep(max(0, TIME_FOR_1_FRAME - (time.time() - timer_this_frame)))

        if len(agent.memory.buffer) > 1000:
            agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_net()

    # Almacenar puntaje y epsilon al final de cada episodio
    scores.append(score)
    epsilons.append(agent.epsilon)

    if episode % 10 == 0:
        print(f'Episode {episode}, Score: {score}, Epsilon: {agent.epsilon}')

# Graficar los resultados al final del entrenamiento
plt.figure(figsize=(12, 6))

# Graficar puntajes
plt.subplot(1, 2, 1)
plt.plot(scores)
plt.title('Score por Episodio')
plt.xlabel('Episodio')
plt.ylabel('Score')

# Graficar valores de epsilon
plt.subplot(1, 2, 2)
plt.plot(epsilons)
plt.title('Valor de Epsilon por Episodio')
plt.xlabel('Episodio')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.show()

window.bye()
