import random
import time
import turtle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

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
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.lr = ALPHA
        self.memory = collections.deque(maxlen=10000)

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
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def get_state():
    cannon_x = (cannon.xcor() - LEFT) / (RIGHT - LEFT)
    if aliens:
        closest_alien = min(aliens, key=lambda alien: alien.ycor())
        alien_x = (closest_alien.xcor() - LEFT) / (RIGHT - LEFT)
        alien_y = (TOP - closest_alien.ycor()) / (TOP - FLOOR_LEVEL)
    else:
        alien_x = 0.5
        alien_y = 0.5
    return np.array([cannon_x, alien_x, alien_y])

def get_reward(alien_hit, done, closest_alien):
    if done:
        return -10
    if alien_hit:
        if closest_alien is not None:
            return 30 + 5 * (1 - closest_alien.ycor() / (TOP - FLOOR_LEVEL))
        else:
            return 30
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
state_space_dim = 3  # Ejemplo: [posición del cañón, x del alien más cercano, y del alien más cercano]
action_space = [0, 1, 2]  # 0: Izquierda, 1: Derecha, 2: Disparar
agent = Agent(state_space_dim, action_space)

# Inicializar variables globales
laser = None
aliens = []

# Bucle de entrenamiento
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

        alien_hit = False
        move_laser()
        for alien in aliens.copy():
            if laser and laser.distance(alien) < 20:
                remove_laser()
                remove_sprite(alien, aliens)
                score += 1
                alien_hit = True
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
        reward = get_reward(alien_hit, not game_running, closest_alien)
        done = not game_running

        agent.remember(current_state, action, reward, next_state, done)
        agent.learn()

        if done:
            break

        window.update()

        time.sleep(max(0, TIME_FOR_1_FRAME - (timer_this_frame)))

        if len(agent.memory) > 1000:
            agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_net()

    if episode % 10 == 0:
        print(f'Episode {episode}, Score: {score}, Epsilon: {agent.epsilon}')

window.bye()

