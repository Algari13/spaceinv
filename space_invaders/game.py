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
EPSILON_DECAY = 0.999
GAMMA = 0.99
ALPHA = 0.0001
BATCH_SIZE = 64
TARGET_UPDATE_INTERVAL = 10
NUM_EPISODES = 10000
MEMORY_SIZE = 10000

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
        laser.forward(20)
        laser.pendown()
        laser.pensize(5)
        laser.showturtle()

def move_laser():
    global laser
    if laser:
        laser.clear()
        laser.forward(LASER_SPEED)
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

class DQN_CNN(nn.Module):
    def __init__(self, input_channels, action_space):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Agent:
    def __init__(self, input_channels, action_space):
        self.input_channels = input_channels
        self.action_space = action_space
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.lr = ALPHA
        self.memory = collections.deque(maxlen=MEMORY_SIZE)

        self.policy_net = DQN_CNN(input_channels, action_space).to(device)
        self.target_net = DQN_CNN(input_channels, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, file_name):
        torch.save(self.policy_net.state_dict(), file_name)

    def load_model(self, file_name):
        self.policy_net.load_state_dict(torch.load(file_name))
        self.target_net.load_state_dict(torch.load(file_name))

def get_state():
    canvas = turtle.getcanvas()
    canvas.postscript(file="frame.eps")
    img = ImageGrab.grab(bbox=(window.window_width() // 2, window.window_height() // 2, window.window_width(), window.window_height()))
    img = img.convert('L')
    img = img.resize((84, 84))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

def get_reward(alien_hit, done, closest_alien):
    if done:
        return -100 
    if alien_hit:
        return 30 + (TOP - closest_alien.ycor()) / TOP * 30
    return -10

def reset_environment():
    global laser, aliens
    if laser:
        remove_laser()
    for alien in aliens.copy():
        remove_sprite(alien, aliens)
    aliens = []
    cannon.setposition(0, FLOOR_LEVEL)
    draw_cannon()

# Inicializar agente
state_space_dim = 1  # Para imágenes en escala de grises
action_space = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(state_space_dim, action_space)

laser = None
aliens = []

# Monitorización
episode_rewards = []

# Configurar gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(episode_rewards)
ax.set_xlim(0, NUM_EPISODES)
ax.set_ylim(-100, 1000)
plt.xlabel('Episode')
plt.ylabel('Reward')

# Entrenamiento del agente
for episode in range(NUM_EPISODES):
    reset_environment()
    state = get_state()
    total_reward = 0
    done = False
    alien_hit = False
    last_alien_spawn_time = time.time()

    while not done:
        action = agent.choose_action(state)

        # Realizar acción
        if action == 0:
            cannon.cannon_movement = -1
        elif action == 1:
            cannon.cannon_movement = 1
        else:
            cannon.cannon_movement = 0

        if cannon.cannon_movement != 0:
            new_x = cannon.xcor() + cannon.cannon_movement * CANNON_STEP
            if LEFT + GUTTER <= new_x <= RIGHT - GUTTER:
                cannon.setx(new_x)

        if action == 2 and laser is None:
            create_laser()

        move_laser()

        for alien in aliens.copy():
            alien.forward(ALIEN_SPEED)
            if laser and alien.distance(laser) < 20:
                alien_hit = True
                remove_laser()
                remove_sprite(alien, aliens)
            if alien.ycor() <= FLOOR_LEVEL:
                done = True
                break

        if time.time() - last_alien_spawn_time > ALIEN_SPAWN_INTERVAL:
            create_alien()
            last_alien_spawn_time = time.time()

        closest_alien = min(aliens, key=lambda a: a.ycor()) if aliens else None
        reward = get_reward(alien_hit, done, closest_alien)
        total_reward += reward

        next_state = get_state()
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        agent.learn()

        window.update()
        time.sleep(TIME_FOR_1_FRAME)

    episode_rewards.append(total_reward)
    line.set_ydata(episode_rewards)
    fig.canvas.draw()

    if agent.epsilon > EPSILON_END:
        agent.epsilon *= EPSILON_DECAY

    if episode % TARGET_UPDATE_INTERVAL == 0:
        agent.update_target_net()

    if episode % 100 == 0:
        print(f"Episode {episode} - Reward: {total_reward}")

# Guardar modelo final
agent.save_model("dqn_model.pth")

# Desactivar gráfica en tiempo real
plt.ioff()
plt.show()

window.mainloop()

