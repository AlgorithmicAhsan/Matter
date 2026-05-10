"""
Test Environment for Action System
Simple 3D visualization using Pygame to test avatar movement with keyboard controls.
"""

import sys
import pygame
import numpy as np
from typing import Tuple
from actions import ActionController, ActionType, KeyboardController


# ── Configuration ────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Colors
BG_COLOR = (20, 20, 30)
GRID_COLOR = (40, 40, 50)
AVATAR_COLOR = (0, 255, 100)
TRAIL_COLOR = (0, 150, 255)
TEXT_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (255, 200, 0)

# Grid settings
GRID_SIZE = 50  # Size of each grid cell in world units
GRID_SPACING = 1.0  # World units between grid lines


class SimpleCamera:
    """Top-down camera for visualization."""
    
    def __init__(self, zoom: float = 30.0):
        self.zoom = zoom  # pixels per world unit
        self.offset = np.array([WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2])
    
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates (x, z) to screen coordinates."""
        screen_x = int(self.offset[0] + world_pos[0] * self.zoom)
        screen_y = int(self.offset[1] - world_pos[2] * self.zoom)  # Flip Z for screen
        return (screen_x, screen_y)
    
    def adjust_zoom(self, delta: float):
        """Zoom in/out."""
        self.zoom = max(10.0, min(100.0, self.zoom + delta))


class TestEnvironment:
    """
    Simple test environment for the action system.
    Top-down view with keyboard controls.
    """
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Action System Test - WASD + Space/Shift/Ctrl + Q/E")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Action system
        self.action_controller = ActionController()
        self.keyboard = KeyboardController(self.action_controller)
        
        # Camera
        self.camera = SimpleCamera(zoom=30.0)
        
        # Visualization
        self.trail: list = []  # Position history for trail
        self.max_trail_length = 100
        
        # Simulation
        self.running = True
        self.gravity = -9.8  # m/s^2
        self.ground_level = 0.0
        
        # Stats
        self.frame_count = 0
    
    def handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Map pygame keys to our key strings
                key = self._get_key_string(event.key)
                if key:
                    self.keyboard.key_down(key)
                
                # Camera zoom
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.camera.adjust_zoom(5)
                elif event.key == pygame.K_MINUS:
                    self.camera.adjust_zoom(-5)
                
                # Reset position
                if event.key == pygame.K_r:
                    self.action_controller.position = np.array([0.0, 0.0, 0.0])
                    self.action_controller.rotation = 0.0
                    self.trail.clear()
            
            elif event.type == pygame.KEYUP:
                key = self._get_key_string(event.key)
                if key:
                    self.keyboard.key_up(key)
    
    def _get_key_string(self, pygame_key) -> str:
        """Convert pygame key constant to our key string."""
        key_map = {
            pygame.K_w: 'w',
            pygame.K_a: 'a',
            pygame.K_s: 's',
            pygame.K_d: 'd',
            pygame.K_q: 'q',
            pygame.K_e: 'e',
            pygame.K_SPACE: ' ',
            pygame.K_LSHIFT: 'shift',
            pygame.K_RSHIFT: 'shift',
            pygame.K_LCTRL: 'ctrl',
            pygame.K_RCTRL: 'ctrl',
        }
        return key_map.get(pygame_key, '')
    
    def update(self, dt: float):
        """Update simulation."""
        # Update action controller
        velocity, rotation_delta = self.action_controller.update(dt)
        
        # Apply rotation
        self.action_controller.rotation += rotation_delta
        
        # Apply velocity
        self.action_controller.position += velocity * dt
        
        # Apply gravity if not grounded
        if not self.action_controller.is_grounded:
            self.action_controller.position[1] += self.gravity * dt * 0.5
        
        # Ground collision
        if self.action_controller.position[1] <= self.ground_level:
            self.action_controller.position[1] = self.ground_level
            self.action_controller.is_grounded = True
        else:
            self.action_controller.is_grounded = False
        
        # Update trail
        self.trail.append(self.action_controller.position.copy())
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
    
    def draw_grid(self):
        """Draw a grid on the ground plane."""
        for i in range(-GRID_SIZE, GRID_SIZE + 1):
            # Vertical lines (along Z-axis)
            start_world = np.array([i * GRID_SPACING, 0, -GRID_SIZE * GRID_SPACING])
            end_world = np.array([i * GRID_SPACING, 0, GRID_SIZE * GRID_SPACING])
            start_screen = self.camera.world_to_screen(start_world)
            end_screen = self.camera.world_to_screen(end_world)
            
            color = HIGHLIGHT_COLOR if i == 0 else GRID_COLOR
            pygame.draw.line(self.screen, color, start_screen, end_screen, 1)
            
            # Horizontal lines (along X-axis)
            start_world = np.array([-GRID_SIZE * GRID_SPACING, 0, i * GRID_SPACING])
            end_world = np.array([GRID_SIZE * GRID_SPACING, 0, i * GRID_SPACING])
            start_screen = self.camera.world_to_screen(start_world)
            end_screen = self.camera.world_to_screen(end_world)
            
            color = HIGHLIGHT_COLOR if i == 0 else GRID_COLOR
            pygame.draw.line(self.screen, color, start_screen, end_screen, 1)
    
    def draw_trail(self):
        """Draw movement trail."""
        if len(self.trail) < 2:
            return
        
        for i in range(1, len(self.trail)):
            alpha = i / len(self.trail)  # Fade effect
            color = (
                int(TRAIL_COLOR[0] * alpha),
                int(TRAIL_COLOR[1] * alpha),
                int(TRAIL_COLOR[2] * alpha)
            )
            
            start = self.camera.world_to_screen(self.trail[i-1])
            end = self.camera.world_to_screen(self.trail[i])
            pygame.draw.line(self.screen, color, start, end, 2)
    
    def draw_avatar(self):
        """Draw avatar as a triangle showing position and orientation."""
        pos_2d = self.camera.world_to_screen(self.action_controller.position)
        
        # Draw direction indicator (triangle)
        size = 15
        angle_rad = np.deg2rad(self.action_controller.rotation)
        
        # Triangle points (pointing in direction of rotation)
        points = [
            (pos_2d[0] + size * np.sin(angle_rad), 
             pos_2d[1] - size * np.cos(angle_rad)),  # Front point
            (pos_2d[0] + size * 0.5 * np.sin(angle_rad + 2.6), 
             pos_2d[1] - size * 0.5 * np.cos(angle_rad + 2.6)),  # Left
            (pos_2d[0] + size * 0.5 * np.sin(angle_rad - 2.6), 
             pos_2d[1] - size * 0.5 * np.cos(angle_rad - 2.6)),  # Right
        ]
        
        # Draw filled triangle
        pygame.draw.polygon(self.screen, AVATAR_COLOR, points)
        
        # Draw circle for body
        pygame.draw.circle(self.screen, AVATAR_COLOR, pos_2d, 8)
        
        # Height indicator (if jumping)
        if self.action_controller.position[1] > 0.1:
            height_offset = int(self.action_controller.position[1] * 10)
            shadow_pos = (pos_2d[0], pos_2d[1] + height_offset)
            pygame.draw.circle(self.screen, (100, 100, 100), shadow_pos, 8, 1)
            pygame.draw.line(self.screen, (150, 150, 150), pos_2d, shadow_pos, 1)
    
    def draw_ui(self):
        """Draw UI overlay with stats and controls."""
        y_offset = 10
        line_height = 25
        
        # Position and rotation
        pos = self.action_controller.position
        texts = [
            f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
            f"Rotation: {self.action_controller.rotation:.1f}°",
            f"Grounded: {self.action_controller.is_grounded}",
            f"",
            f"Active Actions:",
        ]
        
        for text in texts:
            surface = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(surface, (10, y_offset))
            y_offset += line_height
        
        # Active actions
        active = self.action_controller.active_actions
        if active:
            for action_type in active:
                text = f"  • {action_type.name}"
                surface = self.small_font.render(text, True, HIGHLIGHT_COLOR)
                self.screen.blit(surface, (10, y_offset))
                y_offset += 20
        else:
            surface = self.small_font.render("  • IDLE", True, TEXT_COLOR)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20
        
        # Controls help (bottom-left)
        help_texts = [
            "Controls:",
            "WASD - Move",
            "Q/E - Turn",
            "Space - Jump",
            "Shift - Sprint",
            "Ctrl - Crouch",
            "+/- - Zoom",
            "R - Reset position",
        ]
        
        y_offset = WINDOW_HEIGHT - len(help_texts) * 22 - 10
        for text in help_texts:
            surface = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 22
        
        # FPS (top-right)
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        surface = self.font.render(fps_text, True, TEXT_COLOR)
        self.screen.blit(surface, (WINDOW_WIDTH - 100, 10))
    
    def render(self):
        """Render the scene."""
        self.screen.fill(BG_COLOR)
        
        self.draw_grid()
        self.draw_trail()
        self.draw_avatar()
        self.draw_ui()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        print("=== Action System Test Environment ===")
        print("Controls:")
        print("  WASD - Move forward/left/back/right")
        print("  Q/E - Turn left/right")
        print("  Space - Jump")
        print("  Shift - Sprint")
        print("  Ctrl - Crouch")
        print("  +/- - Zoom in/out")
        print("  R - Reset position")
        print("\nClose window or press ESC to quit.")
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Convert to seconds
            
            self.handle_events()
            self.update(dt)
            self.render()
            
            self.frame_count += 1
        
        pygame.quit()


def main():
    env = TestEnvironment()
    env.run()


if __name__ == "__main__":
    main()