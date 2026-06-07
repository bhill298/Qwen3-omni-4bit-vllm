import json
import os
import sys


def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        print(
            "Usage: python generate_tasks.py <media_file1> <prompt_file1> [<media_file2> <prompt_file2> ...]"
        )
        sys.exit(1)

    tasks = []
    for i in range(1, len(sys.argv), 2):
        media = sys.argv[i].replace("\\", "/")
        prompt = sys.argv[i + 1].replace("\\", "/")

        tasks.append({"media": media, "prompt": prompt})

    print(json.dumps(tasks, indent=2))


if __name__ == "__main__":
    main()
