"""Run TotalSegmentator in its own process: python -m lesionagent.backends.ts_run IMAGE OUT TASK FAST(0|1).

In-process TotalSegmentator can hang inside threaded hosts (Agent SDK tool threads, GUI event loops) because
nnU-Net's preprocessing spawns worker processes; a separate process isolates that and returns all GPU memory
afterwards.
"""
import sys


def main():
    image, out, task, fast = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(image, out, ml=True, fast=fast, task=task, device="gpu", quiet=True)


if __name__ == "__main__":
    main()
