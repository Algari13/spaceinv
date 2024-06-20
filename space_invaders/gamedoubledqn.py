import random
import time
import turtle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Parámetros de la IA
EPSILON_START = 1.0  # Tasa de exploración inicial
EPSILON_END = 0.01   # Tasa de exploración final
EPSILON_DECAY = 0.995  # Decaimiento de epsilon
GAMMA = 0.9    # Factor de descuento
ALPHA = 0.001  # Tasa de aprendizaje
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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
        states = np.array(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.long).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_states = np.array(batch[3])
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

        self.policy_net = DQN(state_space, len(action_space)).to(device)
        self.target_net = DQN(state_space, len(action_space)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        beta = min(1.0, 0.4 + (1.0 - 0.4) * (episode / NUM_EPISODES))
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE, beta=beta)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()  # detach() para evitar el error
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        (weights * loss).mean().backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, loss.detach().cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Inicialización del entorno del juego
# Debes completar esta parte con las funciones y variables necesarias para el juego

# Función principal de ejecución del juego
def main():
    global cannon, laser, aliens, episode
    episode = 0
    cannon_movement = 0
    aliens = []

    state_space = 4  # Define la dimensión del espacio de estados
    action_space = [0, 1, 2]  # Define el espacio de acciones

    agent = Agent(state_space, action_space)

    while episode < NUM_EPISODES:
        # Debes completar el ciclo de ejecución del juego
        pass

    turtle.done()

if __name__ == "__main__":
    main()

