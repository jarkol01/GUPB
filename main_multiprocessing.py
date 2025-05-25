import multiprocessing
import sys
from gupb.__main__ import main

def worker_function(x):
    sys.argv = [
        "python -m gupb",
    ]

    main(prog_name='python -m gupb')

if __name__ == "__main__":
    inputs = [1, 2, 3, 4, 5, 6, 7, 8]  # Eight tasks

    with multiprocessing.Pool(processes=6) as pool:
        results = pool.map(worker_function, inputs)

    print("Results:", results)