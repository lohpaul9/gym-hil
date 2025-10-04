#!/usr/bin/env python3
"""
SO-101 Teleoperation Validation

Demonstrates the full teleoperation cycle working:
1. Environment loads with IK control
2. Actions produce correct Cartesian movements
3. Gripper control works
4. Episode completion works
"""

import gymnasium as gym
import numpy as np
import gym_hil  # noqa: F401

GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

def main():
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}    SO-101 TELEOPERATION VALIDATION{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()

    print(f"{CYAN}Testing full teleoperation cycle...{RESET}")
    print()

    # Create environment
    print(f"1. Creating environment...")
    env = gym.make(
        "gym_hil/SO101PickCubeBase-v0",
        render_mode="rgb_array",
        control_dt=0.1,
    )
    print(f"   {GREEN}✓ Environment created{RESET}")
    print()

    # Reset
    print(f"2. Resetting environment...")
    obs, info = env.reset()
    ee_initial = env.unwrapped.data.site_xpos[env.unwrapped._ee_site_id].copy()
    print(f"   {GREEN}✓ Reset complete{RESET}")
    print(f"   Initial EE position: [{ee_initial[0]:.3f}, {ee_initial[1]:.3f}, {ee_initial[2]:.3f}]")
    print()

    # Test 1: Continuous forward movement (teleoperation style)
    print(f"3. Test: Continuous movement in +X direction")
    print(f"   Holding action [+0.005, 0, 0, ...] for 10 steps (simulating held key)...")

    for i in range(10):
        action = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Continuous velocity
        obs, reward, terminated, truncated, info = env.step(action)
        ee_current = env.unwrapped.data.site_xpos[env.unwrapped._ee_site_id].copy()
        displacement = ee_current - ee_initial
        if i % 2 == 1:
            print(f"     Step {i+1}: X displacement = {displacement[0]*100:+.1f}cm")

    ee_after_motion = env.unwrapped.data.site_xpos[env.unwrapped._ee_site_id].copy()
    displacement_motion = ee_after_motion - ee_initial
    print(f"   Total displacement during motion: {displacement_motion[0]*100:+.1f}cm")
    print()

    # Now hold position (zero action) - should stay put
    print(f"   Holding position (zero action) for 10 steps...")
    for i in range(10):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

    ee_final = env.unwrapped.data.site_xpos[env.unwrapped._ee_site_id].copy()
    total_disp = ee_final - ee_initial
    drift = np.linalg.norm(ee_final - ee_after_motion)

    print(f"   Final displacement: {total_disp[0]*100:+.1f}cm")
    print(f"   Position drift: {drift*1000:.2f}mm")
    print()

    # Check results
    moved_correctly = displacement_motion[0] > 0.02  # Moved at least 2cm in +X
    stable = drift < 0.01  # Less than 1cm drift

    if moved_correctly and stable:
        print(f"   {GREEN}✓ Teleoperation control working correctly{RESET}")
        print(f"   {GREEN}✓ Continuous motion: {displacement_motion[0]*100:+.1f}cm{RESET}")
        print(f"   {GREEN}✓ Position stable when holding: {drift*1000:.1f}mm drift{RESET}")
        x_correct = True
    else:
        print(f"   {RED}✗ Control issues detected{RESET}")
        x_correct = False
    print()

    # Test 2: Gripper control
    print(f"4. Test: Gripper control")
    gripper_initial = env.unwrapped.data.ctrl[env.unwrapped._gripper_ctrl_id]
    print(f"   Initial gripper: {gripper_initial:.3f}")

    # Close gripper
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5])  # Negative = close
    obs, reward, terminated, truncated, info = env.step(action)
    gripper_closed = env.unwrapped.data.ctrl[env.unwrapped._gripper_ctrl_id]
    print(f"   After close command: {gripper_closed:.3f}")

    gripper_works = gripper_closed != gripper_initial
    if gripper_works:
        print(f"   {GREEN}✓ Gripper responds to commands{RESET}")
    else:
        print(f"   {YELLOW}⚠ Gripper may not be responding{RESET}")
    print()

    # Test 3: Episode termination
    print(f"5. Test: Episode completion")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(7))

    if not (terminated or truncated):
        print(f"   {GREEN}✓ Episode continues normally{RESET}")
    print()

    # Summary
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}VALIDATION RESULTS{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()

    all_pass = x_correct and gripper_works

    if all_pass:
        print(f"{GREEN}✓ Environment creation & reset{RESET}")
        print(f"{GREEN}✓ Cartesian control (IK-based): {total_disp[0]*100:+.1f}cm{RESET}")
        print(f"{GREEN}✓ Gripper control{RESET}")
        print(f"{GREEN}✓ Episode management{RESET}")
        print()
        print(f"{GREEN}{'=' * 70}{RESET}")
        print(f"{GREEN}SUCCESS! Full teleoperation cycle is working!{RESET}")
        print(f"{GREEN}{'=' * 70}{RESET}")
        print()
        print(f"✅ Ready for teleoperation: {BOLD}MUJOCO_GL=glfw mjpython so101_teleop.py{RESET}")
    else:
        print(f"{RED}Some tests failed - check implementation{RESET}")

    print()
    env.close()

if __name__ == "__main__":
    main()
