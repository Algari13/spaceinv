import random
import time
import turtle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib.pyplot as plt
from PIL import ImageGrab

# Parámetros de la IA
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99
ALPHA = 0.0001
BATCH_SIZE = 64
TARGET_UPDATE_INTERVAL = 10
NUM_EPISODES = 1000
MEMORY_SIZE = 10000
CHECKPOINT_INTERVAL = 100

# Parámetros del juego
FRAME_RATE = 30
TIME_FOR_1_FRAME = 1 / FRAME_RATE
CANNON_STEP = 10
LASER_LENGTH = 20
LASER_SPEED = 20
ALIEN_SPAWN_PROBABILITY = 0.02
ALIEN_SPEED = 3.5
MAX_ALIENS = 5

# Inicializar la pantalla del juego
window = turtle.Screen()
window.tracer(0)
window.setup(width=600, height=800)
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

# Puntaje y estado del juego
score = 0
game_over = False

def draw_cannon():
    cannon.clear()
    cannon.turtlesize(1, 4)  # Base
    cannon.stamp()
    cannon.sety(FLOOR_LEVEL + 10)
    cannon.turtlesize(1, 1.5)  # Next tier
    cannon.stamp()
    cannon.sety(FLOOR_LEVEL + 20)
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
        laser.setposition(cannon.xcor(), cannon.ycor() + 10)
        laser.setheading(90)
        laser.showturtle()

def move_laser():
    global laser, score, game_over
    if laser:
        laser.forward(LASER_SPEED)
        if laser.ycor() > TOP:
            laser.hideturtle()
            laser = None
        else:
            # Detectar colisiones con aliens
            for alien in aliens:
                if laser.distance(alien) < 20:
                    alien.hideturtle()
                    aliens.remove(alien)
                    laser.hideturtle()
                    laser = None
                    score += 10
                    break
            if not aliens:
                game_over = True
                text.goto(0, 0)
                text.write("GAME OVER", align="center", font=("Arial", 24, "normal"))

def move_cannon():
    new_x = cannon.xcor() + CANNON_STEP * cannon.cannon_movement
    if LEFT < new_x < RIGHT:
        cannon.setx(new_x)

def update_text():
    text.clear()
    text.write(f"Score: {score}", align="left", font=("Arial", 16, "normal"))

def spawn_alien():
    if len(aliens) < MAX_ALIENS and random.random() < ALIEN_SPAWN_PROBABILITY:
        alien = turtle.Turtle()
        alien.penup()
        alien.color(0, 1, 0)
        alien.shape("square")
        alien.setposition(random.randint(LEFT + 20, RIGHT - 20), TOP - 40)
        aliens.append(alien)

def move_aliens():
    global game_over
    for alien in aliens:
        alien.sety(alien.ycor() - ALIEN_SPEED)
        if alien.ycor() < FLOOR_LEVEL:
            game_over = True
            text.goto(0, 0)
            text.write("GAME OVER", align="center", font=("Arial", 24, "normal"))

def reset_game():
    global score, game_over, aliens, laser
    score = 0
    game_over = False
    for alien in aliens:
        alien.hideturtle()
    aliens = []
    if laser:
        laser.hideturtle()
        laser = None
    cannon.setposition(0, FLOOR_LEVEL)
    update_text()
    text.clear()

# Eventos de teclado
def go_left():
    cannon.cannon_movement = -1

def go_right():
    cannon.cannon_movement = 1

def stop():
    cannon.cannon_movement = 0

def fire():
    create_laser()

window.listen()
window.onkeypress(go_left, "Left")
window.onkeypress(go_right, "Right")
window.onkeyrelease(stop, "Left")
window.onkeyrelease(stop, "Right")
window.onkeypress(fire, "space")

# Inicializar variables de juego
laser = None
aliens = []

# Definir la red neuronal y la memoria de experiencia
class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(8, 24)  # 8 entradas: 4 posiciones de aliens, 1 de cañón, 1 de láser, 1 de puntaje, 1 si hay láser activo
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)  # 3 salidas: izquierda, derecha, disparar

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Inicializar agente y memoria
agent = DQNAgent()
target_agent = DQNAgent()
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=ALPHA)
memory = ReplayMemory(MEMORY_SIZE)

# Funciones para seleccionar acción y optimizar modelo
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return agent(torch.tensor(state, dtype=torch.float32)).argmax().item()
    else:
        return random.randrange(3)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = zip(*transitions)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = [torch.tensor(x, dtype=torch.float32) for x in batch]

    state_action_values = agent(state_batch).gather(1, action_batch.long().unsqueeze(1)).squeeze(1)
    next_state_values = target_agent(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon = EPSILON_START

# Variables para el seguimiento del aprendizaje
episode_durations = []
scores = []

plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

def plot_durations():
    ax[0].clear()
    ax[0].plot(episode_durations)
    ax[0].set_title('Duración de los episodios')
    ax[0].set_xlabel('Episodio')
    ax[0].set_ylabel('Duración')
    
    ax[1].clear()
    ax[1].plot(scores)
    ax[1].set_title('Puntuaciones')
    ax[1].set_xlabel('Episodio')
    ax[1].set_ylabel('Puntuación')

    plt.pause(0.001)

def get_state():
    cannon_x = cannon.xcor()
    laser_y = laser.ycor() if laser else 0
    num_aliens = len(aliens)
    aliens_positions = []
    for i in range(4):
        if i < num_aliens:
            aliens_positions.append(aliens[i].ycor())
        else:
            aliens_positions.append(0)
    laser_active = int(laser is not None)
    return [cannon_x, laser_y] + aliens_positions + [score, laser_active]

# Bucle principal del juego
for episode in range(NUM_EPISODES):
    if game_over:
        reset_game()

    state = get_state()
    action = select_action(state, epsilon)
    if action == 0:
        go_left()
    elif action == 1:
        go_right()
    elif action == 2:
        fire()

    move_cannon()
    move_laser()
    spawn_alien()
    move_aliens()
    draw_cannon()
    update_text()

    next_state = get_state()
    reward = score
    done = game_over

    memory.push(state, action, reward, next_state, done)
    optimize_model()

    state = next_state
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if done:
        episode_durations.append(episode)
        scores.append(score)
        plot_durations()

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_agent.load_state_dict(agent.state_dict())

    if episode % CHECKPOINT_INTERVAL == 0:
        torch.save(agent.state_dict(), f'checkpoint_{episode}.pth')

    window.update()
    time.sleep(TIME_FOR_1_FRAME)

plt.ioff()
plt.show()
