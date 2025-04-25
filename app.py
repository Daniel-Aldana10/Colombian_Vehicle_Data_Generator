#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from data_generator.vehicle_data_generator import generate_dataset

def prompt_for_rows():
    try:
        val = input("How many records would you like to generate? ")
        return int(val)
    except ValueError:
        print("Invalid input; defaulting to 1000 records.", file=sys.stderr)
        return 1000

def run_single_generation(n, output_path, write_header=True):
    start_time = time.time()
    df = generate_dataset(n=n)
    elapsed = time.time() - start_time

    df.to_csv(
        output_path,
        mode='w' if write_header else 'a',
        index=False,
        header=write_header
    )
    print(f"→ Generated {n} records in {elapsed:.2f} seconds → {output_path}")
    return elapsed

def benchmark_to_single_file(n_total, output_path, step=500):
    results = []
    write_header = True

    for n in range(step, n_total + 1, step):
        t = run_single_generation(n, output_path, write_header=write_header)
        results.append((n, t))
        write_header = False

    return results

def plot_results(results, save_to=None):
    sizes, times = zip(*results)
    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Number of Records")
    plt.ylabel("Generation Time (seconds)")
    plt.title("Benchmark: Generation Time vs. Dataset Size")
    plt.grid(True)
    if save_to:
        plt.savefig(save_to)
        print(f"Chart saved to {save_to}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Colombian Vehicle Dataset Generator"
    )
    parser.add_argument(
        '-n', '--rows',
        type=int,
        help='Total number of records to generate (interactive if omitted)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='vehicle_data.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '-b', '--benchmark',
        action='store_true',
        help='Run benchmark (append blocks of size STEP to a single CSV)'
    )
    parser.add_argument(
        '-s', '--step',
        type=int,
        default=500,
        help='Step size for benchmark increments (default: 500)'
    )
    args = parser.parse_args()

    total = args.rows if args.rows is not None else prompt_for_rows()

    if args.benchmark:
        results = benchmark_to_single_file(
            n_total=total,
            output_path=args.output,
            step=args.step
        )
        plot_results(results, save_to="benchmark.png")
    else:
        run_single_generation(
            n=total,
            output_path=args.output,
            write_header=True
        )

if __name__ == "__main__":
    main()
