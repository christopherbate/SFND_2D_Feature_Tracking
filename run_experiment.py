#!/usr/bin/python3
import subprocess

prog = "cmake-build-debug/2D_feature_tracking"

keypoint_types = ["SHITOMASI", "HARRIS",
                  "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
desc_types = ["BRIEF", "ORB", "BRISK", "FREAK", "AKAZE", "SIFT"]

for kpt in keypoint_types:
    for desc in desc_types:
        # Akaze only usd with akaze keypoint
        if desc == "AKAZE" and kpt != "AKAZE":
            continue

        subprocess.run([prog, kpt, desc])
