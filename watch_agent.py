import sys
import pygame
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from env_flappy import FlappyBirdEnv

WIDTH, HEIGHT = 400, 600

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = FlappyBirdEnv()
    if sys.argv[1] == "ppo":
        model = PPO.load("ppo_flappy")
    else:
        model = DQN.load("dqn_flappy")

    obs, _ = env.reset()
    running = True
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if done:
            obs, _ = env.reset()
            done = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # drawing
        screen.fill((135, 206, 235))  # sky

        # bird
        bird_x_px = int(env.bird_x * WIDTH)
        bird_y_px = int((1.0 - env.y) * HEIGHT)  # invert y for pixel coords
        pygame.draw.circle(screen, (255, 255, 0), (bird_x_px, bird_y_px), 10)

        # pipe
        pipe_x_px = int(env.pipe_x * WIDTH)
        gap_y_px  = int((1.0 - env.pipe_gap_y) * HEIGHT)
        gap_half_px = int(env.gap_half * HEIGHT)

        # top pipe
        pygame.draw.rect(screen, (0, 200, 0), (pipe_x_px, 0, 60, gap_y_px - gap_half_px))
        # bottom pipe
        pygame.draw.rect(screen, (0, 200, 0), (pipe_x_px, gap_y_px + gap_half_px, 60, HEIGHT))

        # score text
        font = pygame.font.SysFont(None, 24)
        img = font.render(f"Score: {info.get('score', 0)}", True, (0, 0, 0))
        screen.blit(img, (10, 10))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
