#!/usr/bin/env python3
"""
test_grippers.py — dual-arm end-effector test for the G02, run on the robot.

Checks that BOTH grippers respond, on both command channels, before any grasp run:

  1. read       — report each arm's end-effector joints, so a missing/wrong tool
                  group is visible before anything moves
  2. open/close — sweep left then right through _gripper()'s joint_servo_control
                  stream (the path open_gripper/close_gripper actually use)
  3. open/close — same sweep through test_gripper()'s move_ee_pos() channel, so
                  the two channels can be compared on the same hardware

Nothing but the end-effector joints is commanded — the arms stay where they are.
Run with the arms clear of the table and nothing in the grippers.

Usage:
    python3 test_grippers.py                 # full test, both arms, both channels
    python3 test_grippers.py --arm left      # one arm only
    python3 test_grippers.py --skip-ee-pos   # servo channel only
"""
import argparse
import sys
import time

import agibot_gdk as gdk

import robot_control as rc


def read_side(robot, arm: str) -> tuple[list, list] | None:
    """(names, positions) of one arm's end-effector joints."""
    try:
        side = robot.get_end_state()["left_end_state" if arm == "left"
                                     else "right_end_state"]
    except Exception as e:
        print(f"  [{arm}] end-state read failed: {e}")
        return None
    names = list(side.get("names", []))
    pos   = [float(s["position"]) for s in side.get("end_states", [])]
    if not names:
        print(f"  [{arm}] no end-effector joints reported")
        return None
    print(f"  [{arm}] {len(names)} EE joint(s):")
    for n, p in zip(names, pos):
        print(f"        {n:42s} pos={p:+.3f}")
    return names, pos


def sweep(robot, gripper_type: str, arm: str, label: str) -> None:
    print(f"\n--- {label}: {arm} ---")
    t0 = time.time()
    rc.open_gripper(robot, gripper_type, arm)
    time.sleep(0.6)
    after_open = read_side(robot, arm)
    rc.close_gripper(robot, gripper_type, arm)
    time.sleep(0.6)
    after_close = read_side(robot, arm)

    if after_open and after_close:
        po, pc = after_open[1][0], after_close[1][0]
        print(f"  [{arm}] open={po:+.3f} → close={pc:+.3f}  "
              f"(Δ={abs(pc - po):.3f} rad)")
        if abs(pc - po) < 0.05:
            print(f"  [{arm}] *** WARNING: finger barely moved — check tool group ***")
    print(f"  [{label}] {time.time() - t0:.1f}s")


def sweep_ee_pos(robot, gripper_type: str, arm: str) -> None:
    """Raw move_ee_pos() channel, bypassing the servo stream entirely."""
    print(f"\n--- move_ee_pos only: {arm} ---")
    for pos, what in ((rc.GRIPPER_OPEN_POS[gripper_type][0], "open"),
                      (rc.GRIPPER_CLOSE_POS[gripper_type][0], "close")):
        print(f"  → {what} ({pos:+.3f})")
        rc.test_gripper(robot, pos, arm, gripper_type)
        time.sleep(0.6)
        read_side(robot, arm)


def main():
    ap = argparse.ArgumentParser(description="G02 dual-arm gripper test")
    ap.add_argument("--arm", choices=["both", "left", "right"], default="both")
    ap.add_argument("--gripper", choices=rc._GRIPPER_CHOICES, default="omnipicker")
    ap.add_argument("--skip-ee-pos", action="store_true",
                    help="skip the move_ee_pos() channel, test only the servo stream")
    args = ap.parse_args()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]

    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        print("gdk_init() failed"); return 1
    time.sleep(1)

    robot = gdk.Robot()
    time.sleep(2)

    try:
        print(f"Gripper test  gripper={args.gripper}  arms={arms}")
        print("\n=== 1. end-effector joints as reported ===")
        for arm in arms:
            read_side(robot, arm)

        # The arms are left exactly where they are; only EE joints are driven.
        for arm in arms:
            sweep(robot, args.gripper, arm, "servo stream (joint_servo_control)")

        if not args.skip_ee_pos:
            for arm in arms:
                sweep_ee_pos(robot, args.gripper, arm)

        print("\n=== done ===")
        for arm in arms:
            print(f"final {arm}:")
            read_side(robot, arm)
        return 0
    finally:
        try:
            gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
