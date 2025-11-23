"""
Manual Flappy Bird Game
Play the game yourself using the SPACE key to flap!
"""
import pygame
from env_flappy import FlappyBirdEnv


class FlappyBirdRenderer:
    """Pygame renderer for the Flappy Bird environment"""
    
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird - Manual Play")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 32)
        
        # Colors
        self.BG_COLOR = (135, 206, 250)  # Sky blue
        self.BIRD_COLOR = (225, 50, 110)  # Pink
        self.PIPE_COLOR = (34, 139, 34)  # Green
        self.GROUND_COLOR = (222, 184, 135)  # Tan
        self.GAP_COLOR = (100, 149, 237)  # Cornflower blue
        
    def render(self, env):
        """Render the current game state"""
        self.screen.fill(self.BG_COLOR)
        
        # Draw ground and ceiling
        pygame.draw.rect(self.screen, self.GROUND_COLOR, 
                        (0, int(self.height * 0.95), self.width, int(self.height * 0.05)))
        
        # Draw bird
        bird_screen_x = int(env.bird_x * self.width)
        bird_screen_y = int((1 - env.y) * self.height)  # Invert y-axis
        bird_radius = 15
        pygame.draw.circle(self.screen, self.BIRD_COLOR, 
                          (bird_screen_x, bird_screen_y), bird_radius)
        # Eye
        eye_x = bird_screen_x + 5
        eye_y = bird_screen_y - 3
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), 3)
        
        # Draw pipe
        pipe_screen_x = int(env.pipe_x * self.width)
        pipe_width = 60
        gap_screen_y = int((1 - env.pipe_gap_y) * self.height)
        gap_half_screen = int(env.gap_half * self.height)
        
        # Top pipe
        top_pipe_height = gap_screen_y - gap_half_screen
        if top_pipe_height > 0:
            pygame.draw.rect(self.screen, self.PIPE_COLOR,
                           (pipe_screen_x - pipe_width // 2, 0, 
                            pipe_width, top_pipe_height))
            # Pipe cap
            pygame.draw.rect(self.screen, (40, 160, 40),
                           (pipe_screen_x - pipe_width // 2 - 5, top_pipe_height - 20,
                            pipe_width + 10, 20))
        
        # Bottom pipe
        bottom_pipe_top = gap_screen_y + gap_half_screen
        bottom_pipe_height = int(self.height * 0.95) - bottom_pipe_top
        if bottom_pipe_height > 0:
            pygame.draw.rect(self.screen, self.PIPE_COLOR,
                           (pipe_screen_x - pipe_width // 2, bottom_pipe_top,
                            pipe_width, bottom_pipe_height))
            # Pipe cap
            pygame.draw.rect(self.screen, (40, 160, 40),
                           (pipe_screen_x - pipe_width // 2 - 5, bottom_pipe_top,
                            pipe_width + 10, 20))
        
        # Draw score
        score_text = self.font.render(f"Score: {env.score}", True, (255, 255, 255))
        score_shadow = self.font.render(f"Score: {env.score}", True, (0, 0, 0))
        self.screen.blit(score_shadow, (22, 22))
        self.screen.blit(score_text, (20, 20))
        
        # Draw instructions
        inst_text = self.small_font.render("SPACE to flap | ESC to quit", True, (255, 255, 255))
        inst_shadow = self.small_font.render("SPACE to flap | ESC to quit", True, (0, 0, 0))
        self.screen.blit(inst_shadow, (22, self.height - 42))
        self.screen.blit(inst_text, (20, self.height - 40))
        
        pygame.display.flip()
        
    def render_game_over(self, score):
        """Render game over screen"""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font.render("GAME OVER", True, (255, 0, 0))
        score_text = self.font.render(f"Final Score: {score}", True, (255, 255, 255))
        restart_text = self.small_font.render("Press R to restart | ESC to quit", True, (255, 255, 255))
        
        self.screen.blit(game_over_text, 
                        (self.width // 2 - game_over_text.get_width() // 2, 
                         self.height // 2 - 100))
        self.screen.blit(score_text,
                        (self.width // 2 - score_text.get_width() // 2,
                         self.height // 2 - 20))
        self.screen.blit(restart_text,
                        (self.width // 2 - restart_text.get_width() // 2,
                         self.height // 2 + 60))
        
        pygame.display.flip()
        
    def close(self):
        """Clean up pygame"""
        pygame.quit()


def play_game():
    """Main game loop for manual play"""
    env = FlappyBirdEnv()

    # Check if display is available
    try:
        renderer = FlappyBirdRenderer()
    except Exception as e:
        print(f"Error initializing display: {e}")
        print("Make sure you have a display available (X11, Wayland, etc.)")
        return

    running = True
    game_over = False
    
    # Reset the environment
    obs, info = env.reset()
    
    try:
        while running:
            # Handle events
            action = 0  # Default: no flap
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and not game_over:
                        action = 1  # Flap!
                    elif event.key == pygame.K_r and game_over:
                        # Restart game
                        obs, info = env.reset()
                        game_over = False
            
            if not game_over:
                # Step the environment
                obs, reward, done, truncated, info = env.step(action)
                
                # Render current state
                renderer.render(env)
                
                # Check if game over
                if done:
                    game_over = True
                    renderer.render_game_over(env.score)
            
            # Control frame rate
            renderer.clock.tick(30)  # 30 FPS
    
    finally:
        renderer.close()
    
    print(f"\nGame ended! Final score: {env.score}")


if __name__ == "__main__":
    print("Starting Flappy Bird - Manual Play")
    print("Controls:")
    print("  SPACE - Flap")
    print("  R     - Restart (after game over)")
    print("  ESC   - Quit")
    print("\nGood luck!\n")
    
    play_game()

