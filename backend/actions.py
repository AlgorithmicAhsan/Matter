"""
Action System for Avatar Movement
Defines discrete, game-like actions similar to game engine input systems.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class ActionType(Enum):
    """Discrete actions the avatar can perform."""
    IDLE = "idle"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    JUMP = "jump"
    CROUCH = "crouch"
    SPRINT = "sprint"


@dataclass
class ActionConfig:
    """Configuration parameters for each action type."""
    # Movement speeds (units per second)
    walk_speed: float = 2.0
    sprint_speed: float = 4.0
    strafe_speed: float = 1.5
    backward_speed: float = 1.2
    
    # Rotation (degrees per second)
    turn_speed: float = 120.0
    
    # Jump parameters
    jump_velocity: float = 5.0
    jump_cooldown: float = 0.5  # seconds
    
    # Crouch
    crouch_height_multiplier: float = 0.6
    crouch_speed_multiplier: float = 0.5
    
    # Sprint
    sprint_multiplier: float = 2.0


@dataclass
class ActionState:
    """Current state of an action being performed."""
    action: ActionType
    intensity: float = 1.0  # 0.0 to 1.0, for blend/partial actions
    duration: float = 0.0   # how long this action has been active
    cooldown_remaining: float = 0.0
    
    def update(self, dt: float):
        """Update action timers."""
        self.duration += dt
        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0, self.cooldown_remaining - dt)
    
    def can_activate(self) -> bool:
        """Check if action can be activated (not on cooldown)."""
        return self.cooldown_remaining <= 0


class ActionController:
    """
    Manages avatar actions and converts them to movement vectors.
    Handles action state, cooldowns, and conflicting inputs.
    """
    
    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()
        
        # Active actions (can have multiple active at once, e.g., forward + strafe)
        self.active_actions: dict[ActionType, ActionState] = {}
        
        # Avatar state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.rotation = 0.0  # degrees, Y-axis rotation
        self.is_grounded = True
        self.is_crouching = False
        self.is_sprinting = False
        
        # Jump cooldown tracker
        self.jump_cooldown = 0.0
    
    def activate_action(self, action: ActionType, intensity: float = 1.0):
        """Activate an action with optional intensity."""
        # Handle jump cooldown
        if action == ActionType.JUMP:
            if self.jump_cooldown > 0 or not self.is_grounded:
                return  # Can't jump
        
        # Deactivate conflicting actions
        if action == ActionType.SPRINT and ActionType.CROUCH in self.active_actions:
            self.deactivate_action(ActionType.CROUCH)
        elif action == ActionType.CROUCH and ActionType.SPRINT in self.active_actions:
            self.deactivate_action(ActionType.SPRINT)
        
        # Add or update action
        if action not in self.active_actions:
            self.active_actions[action] = ActionState(action, intensity)
        else:
            self.active_actions[action].intensity = intensity
    
    def deactivate_action(self, action: ActionType):
        """Stop performing an action."""
        if action in self.active_actions:
            del self.active_actions[action]
    
    def update(self, dt: float):
        """
        Update all active actions and compute movement.
        dt: delta time in seconds
        Returns: (velocity_vector, rotation_delta)
        """
        # Update cooldowns
        if self.jump_cooldown > 0:
            self.jump_cooldown = max(0, self.jump_cooldown - dt)
        
        # Update action durations
        for state in self.active_actions.values():
            state.update(dt)
        
        # Compute movement vector (in avatar's local space)
        velocity = np.array([0.0, 0.0, 0.0])  # x (right), y (up), z (forward)
        rotation_delta = 0.0
        
        # Determine base speed
        base_speed = self.config.walk_speed
        if ActionType.SPRINT in self.active_actions and not self.is_crouching:
            base_speed *= self.config.sprint_multiplier
            self.is_sprinting = True
        else:
            self.is_sprinting = False
        
        if ActionType.CROUCH in self.active_actions:
            base_speed *= self.config.crouch_speed_multiplier
            self.is_crouching = True
        else:
            self.is_crouching = False
        
        # Forward/backward movement
        if ActionType.MOVE_FORWARD in self.active_actions:
            intensity = self.active_actions[ActionType.MOVE_FORWARD].intensity
            velocity[2] += base_speed * intensity
        
        if ActionType.MOVE_BACKWARD in self.active_actions:
            intensity = self.active_actions[ActionType.MOVE_BACKWARD].intensity
            velocity[2] -= self.config.backward_speed * intensity
        
        # Strafing
        if ActionType.STRAFE_RIGHT in self.active_actions:
            intensity = self.active_actions[ActionType.STRAFE_RIGHT].intensity
            velocity[0] += self.config.strafe_speed * intensity
        
        if ActionType.STRAFE_LEFT in self.active_actions:
            intensity = self.active_actions[ActionType.STRAFE_LEFT].intensity
            velocity[0] -= self.config.strafe_speed * intensity
        
        # Rotation
        if ActionType.TURN_RIGHT in self.active_actions:
            intensity = self.active_actions[ActionType.TURN_RIGHT].intensity
            rotation_delta += self.config.turn_speed * dt * intensity
        
        if ActionType.TURN_LEFT in self.active_actions:
            intensity = self.active_actions[ActionType.TURN_LEFT].intensity
            rotation_delta -= self.config.turn_speed * dt * intensity
        
        # Jump (instant vertical velocity)
        if ActionType.JUMP in self.active_actions and self.is_grounded:
            velocity[1] = self.config.jump_velocity
            self.jump_cooldown = self.config.jump_cooldown
            self.is_grounded = False
            self.deactivate_action(ActionType.JUMP)  # Jump is one-shot
        
        # Convert local velocity to world space
        velocity_world = self._rotate_vector(velocity, self.rotation)
        
        return velocity_world, rotation_delta
    
    def _rotate_vector(self, vec: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate a vector around Y-axis by angle_deg."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix around Y-axis
        rotated = vec.copy()
        rotated[0] = vec[0] * cos_a + vec[2] * sin_a
        rotated[2] = -vec[0] * sin_a + vec[2] * cos_a
        
        return rotated
    
    def get_state_dict(self) -> dict:
        """Get current state for debugging/logging."""
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation,
            "active_actions": [a.name for a in self.active_actions.keys()],
            "is_grounded": self.is_grounded,
            "is_crouching": self.is_crouching,
            "is_sprinting": self.is_sprinting,
            "jump_cooldown": self.jump_cooldown
        }


# ── Keyboard Input Mapping ──────────────────────────────────────────────────

class KeyboardController:
    """
    Maps keyboard inputs to actions for testing.
    Standard FPS controls: WASD + Space + Shift + Ctrl
    """
    
    # Key mappings (can be customized)
    KEY_BINDINGS = {
        'w': ActionType.MOVE_FORWARD,
        's': ActionType.MOVE_BACKWARD,
        'a': ActionType.STRAFE_LEFT,
        'd': ActionType.STRAFE_RIGHT,
        'q': ActionType.TURN_LEFT,
        'e': ActionType.TURN_RIGHT,
        ' ': ActionType.JUMP,      # Space
        'shift': ActionType.SPRINT,
        'ctrl': ActionType.CROUCH,
    }
    
    def __init__(self, action_controller: ActionController):
        self.controller = action_controller
        self.pressed_keys = set()
    
    def key_down(self, key: str):
        """Handle key press event."""
        key = key.lower()
        self.pressed_keys.add(key)
        
        if key in self.KEY_BINDINGS:
            action = self.KEY_BINDINGS[key]
            self.controller.activate_action(action)
    
    def key_up(self, key: str):
        """Handle key release event."""
        key = key.lower()
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        
        if key in self.KEY_BINDINGS:
            action = self.KEY_BINDINGS[key]
            self.controller.deactivate_action(action)
    
    def update(self):
        """Maintain continuous actions for held keys."""
        for key in self.pressed_keys:
            if key in self.KEY_BINDINGS:
                action = self.KEY_BINDINGS[key]
                # Re-activate to ensure it stays active
                self.controller.activate_action(action)


if __name__ == "__main__":
    # Quick test
    controller = ActionController()
    
    print("Testing action system...")
    print("\nInitial state:", controller.get_state_dict())
    
    # Simulate some actions
    controller.activate_action(ActionType.MOVE_FORWARD)
    controller.activate_action(ActionType.SPRINT)
    
    for i in range(5):
        velocity, rotation = controller.update(dt=0.1)
        controller.position += velocity * 0.1
        controller.rotation += rotation
        
        print(f"\nFrame {i}:")
        print(f"  Velocity: {velocity}")
        print(f"  Position: {controller.position}")
        print(f"  Rotation: {controller.rotation:.1f}°")
    
    print("\nFinal state:", controller.get_state_dict())