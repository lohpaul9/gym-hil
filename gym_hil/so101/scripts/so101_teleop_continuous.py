#!/usr/bin/env python3
"""
SO-101 Continuous Teleoperation Mode

Simple continuous control - no episodes, no time limits.
Just control the robot with the keyboard until you quit.
"""

import gymnasium as gym
import numpy as np
import gym_hil  # noqa: F401

GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def main():
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}           SO-101 CONTINUOUS TELEOPERATION{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()
    print(f"{CYAN}Keyboard Controls:{RESET}")
    print(f"  Arrow Keys:  Move in X-Y plane")
    print(f"  W/S:         Move up/down in Z")
    print(f"  O/C:         Open/Close gripper")
    print(f"  Space:       Start/Stop intervention")
    print(f"  R:           Reset robot to home")
    print(f"  ESC/Q:       Exit")
    print()
    print(f"{GREEN}✓ Using IK-based control for precise Cartesian movements{RESET}")
    print()

    # Create environment using the standard registration system
    env = gym.make(
        "gym_hil/SO101PickCubeKeyboard-v0",
        render_mode="human",
        control_dt=0.1,
    )

    # Remove the TimeLimit wrapper to allow continuous operation
    # Unwrap until we find TimeLimit, then skip it
    from gymnasium.wrappers import TimeLimit
    if isinstance(env, TimeLimit):
        env = env.env

    print(f"{GREEN}✓ Environment created - press Space to start controlling!{RESET}")
    print()

    try:
        obs, info = env.reset()
        step_count = 0

        while True:
            # Step with None - wrapper handles keyboard input
            obs, reward, terminated, truncated, info = env.step(None)
            step_count += 1

            # Show step count every 100 steps
            if step_count % 100 == 0:
                print(f"Steps: {step_count}", end='\r')

            # Handle reset request
            if info.get('rerecord_episode', False):
                print(f"\n{CYAN}Resetting robot...{RESET}")
                obs, info = env.reset()
                step_count = 0
                continue

            # Check for exit
            if terminated or truncated:
                # Check if it's a manual exit (ESC pressed)
                break

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{YELLOW}Error: {e}{RESET}")
    finally:
        print()
        print(f"{CYAN}Total steps: {step_count}{RESET}")
        print(f"{GREEN}Teleoperation ended{RESET}")
        print()
        env.close()


if __name__ == "__main__":
    main()
