#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from data_generator.vehicle_data_generator import generate_dataset
import psutil
import os
import threading


def prompt_for_rows():
    try:
        val = input("How many records would you like to generate? ")
        return int(val)
    except ValueError:
        print("Invalid input; defaulting to 1000 records.", file=sys.stderr)
        return 1000


def run_single_generation(n, output_path, write_header=True, monitor_memory=False, vehicle_type=None):
    """
    Run a single generation of data with optional memory monitoring
    """
    # Start memory monitoring in a separate thread if requested
    if monitor_memory:
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_memory_usage,
            args=(stop_monitoring, 5)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

    start_time = time.time()

    # Call generate_dataset with the correct parameters
    generate_dataset(
        n=n,
        output_path=output_path,
        sub_type=vehicle_type
    )

    elapsed = time.time() - start_time

    # Stop memory monitoring if active
    if monitor_memory:
        stop_monitoring.set()
        monitor_thread.join(timeout=1)

    print(f"\n→ Generated {n:,} records in {elapsed:.2f} seconds → {output_path}")
    print(f"  Average speed: {n / elapsed:.1f} records/second")

    return elapsed


def benchmark_to_single_file(n_total, output_path, step=500, monitor_memory=False, vehicle_type=None):
    """
    Run benchmark by incrementally increasing the number of records
    """
    results = []
    write_header = True

    for n in range(step, n_total + 1, step):
        print(f"\n=== Benchmarking with {n:,} records ===")
        t = run_single_generation(
            n=n,
            output_path=output_path,
            write_header=write_header,
            monitor_memory=monitor_memory,
            vehicle_type=vehicle_type
        )
        results.append((n, t))
        write_header = False  # Disable header after first iteration

    return results


def plot_results(results, save_to=None):
    """
    Generate a chart with benchmark results
    """
    sizes, times = zip(*results)
    plt.figure(figsize=(10, 6))

    # Total time plot
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o', color='blue')
    plt.xlabel("Number of Records")
    plt.ylabel("Total Time (seconds)")
    plt.title("Total Time vs. Dataset Size")
    plt.grid(True)

    # Records per second plot
    plt.subplot(1, 2, 2)
    records_per_second = [size / time for size, time in results]
    plt.plot(sizes, records_per_second, marker='o', color='green')
    plt.xlabel("Number of Records")
    plt.ylabel("Records per Second")
    plt.title("Performance vs. Dataset Size")
    plt.grid(True)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
        print(f"Charts saved to {save_to}")
    else:
        plt.show()


def monitor_memory_usage(stop_event, interval=5):
    """
    Function to monitor memory usage during generation
    """
    process = psutil.Process(os.getpid())
    max_memory = 2000

    while not stop_event.is_set():
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        max_memory = max(max_memory, memory_usage_mb)
        print(f"[Monitor] Memory usage: {memory_usage_mb:.2f} MB (Maximum: {max_memory:.2f} MB)")

        # Check for memory warnings
        if memory_usage_mb > 1000:  # Warning if using more than 1GB
            print(f"[Monitor] ⚠️ WARNING: High memory usage ({memory_usage_mb:.2f} MB)")

        stop_event.wait(interval)

    print(f"[Monitor] Maximum memory usage during execution: {max_memory:.2f} MB")


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
    parser.add_argument(
        '-m', '--monitor',
        action='store_true',
        help='Monitor memory usage during generation'
    )
    parser.add_argument(
        '-t', '--type',
        type=str,
        default=None,
        choices=['particular', 'servicio_publico', 'diplomatico',
                'remolque', 'carga_especial', 'suv', 'camioneta'],
        help='Vehicle type filter'
    )
    args = parser.parse_args()

    total = args.rows if args.rows is not None else prompt_for_rows()

    if args.benchmark:
        print(f"Starting benchmark for generating up to {total:,} records...")
        results = benchmark_to_single_file(
            n_total=total,
            output_path=args.output,
            step=args.step,
            monitor_memory=args.monitor,
            vehicle_type=args.type
        )
        plot_results(results, save_to="benchmark.png")
    else:
        print(f"Starting generation of {total:,} records{' with memory monitoring' if args.monitor else ''}...")
        run_single_generation(
            n=total,
            output_path=args.output,
            write_header=True,
            monitor_memory=args.monitor,
            vehicle_type=args.type
        )


if __name__ == "__main__":
    main()