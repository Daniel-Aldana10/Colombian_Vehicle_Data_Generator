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


def run_single_generation(n, output_path, write_header=True, monitor_memory=False, vehicle_type=None,
                          bloom_error_rate=0.01):
    """
    Ejecuta una generación única de datos con opción de monitoreo de memoria
    """
    # Iniciar monitoreo de memoria en un hilo separado si se solicita
    if monitor_memory:
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_memory_usage,
            args=(stop_monitoring, 5)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

    start_time = time.time()

    # Llamar a generate_dataset con el total de registros y la tasa de error configurada
    generate_dataset(
        n=n,
        output_path=output_path,
        sub_type=vehicle_type,
        monitor_memory=monitor_memory
    )

    elapsed = time.time() - start_time

    # Detener el monitoreo de memoria si está activo
    if monitor_memory:
        stop_monitoring.set()
        monitor_thread.join(timeout=1)

    print(f"\n→ Generados {n:,} registros en {elapsed:.2f} segundos → {output_path}")
    print(f"  Velocidad promedio: {n / elapsed:.1f} registros/segundo")

    return elapsed


def benchmark_to_single_file(n_total, output_path, step=500, monitor_memory=False, vehicle_type=None, bloom_error_rate=0.01):
    """
    Ejecuta benchmark de generación incrementando el número de registros
    """
    results = []
    write_header = True

    for n in range(step, n_total + 1, step):
        print(f"\n=== Benchmarking con {n:,} registros ===")
        t = run_single_generation(
            n=n,
            output_path=output_path,
            write_header=write_header,
            monitor_memory=monitor_memory,
            vehicle_type=vehicle_type,
            bloom_error_rate=bloom_error_rate
        )
        results.append((n, t))
        write_header = False  # Disable header after first iteration

    return results


def plot_results(results, save_to=None):
    """
    Genera un gráfico con los resultados del benchmark
    """
    sizes, times = zip(*results)
    plt.figure(figsize=(10, 6))

    # Gráfico de tiempo total
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o', color='blue')
    plt.xlabel("Número de Registros")
    plt.ylabel("Tiempo Total (segundos)")
    plt.title("Tiempo Total vs. Tamaño del Dataset")
    plt.grid(True)

    # Gráfico de registros por segundo
    plt.subplot(1, 2, 2)
    records_per_second = [size / time for size, time in results]
    plt.plot(sizes, records_per_second, marker='o', color='green')
    plt.xlabel("Número de Registros")
    plt.ylabel("Registros por Segundo")
    plt.title("Rendimiento vs. Tamaño del Dataset")
    plt.grid(True)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
        print(f"Gráficos guardados en {save_to}")
    else:
        plt.show()


def monitor_memory_usage(stop_event, interval=5):
    """
    Función para monitorear el uso de memoria durante la generación
    """
    process = psutil.Process(os.getpid())
    max_memory = 2000

    while not stop_event.is_set():
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        max_memory = max(max_memory, memory_usage_mb)
        print(f"[Monitor] Uso de memoria: {memory_usage_mb:.2f} MB (Máximo: {max_memory:.2f} MB)")

        # Verificar si hay advertencias de memoria
        if memory_usage_mb > 1000:  # Advertencia si se usan más de 1GB
            print(f"[Monitor] ⚠️ ADVERTENCIA: Uso de memoria alto ({memory_usage_mb:.2f} MB)")

        stop_event.wait(interval)

    print(f"[Monitor] Uso máximo de memoria durante la ejecución: {max_memory:.2f} MB")


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
    # Nuevo parámetro para la tasa de error del BloomFilter
    parser.add_argument(
        '-e', '--error-rate',
        type=float,
        default=0.01,
        help='BloomFilter error rate (0.01 recommended, lower values use more memory)'
    )
    args = parser.parse_args()

    total = args.rows if args.rows is not None else prompt_for_rows()

    if args.benchmark:
        print(f"Iniciando benchmark para generación de hasta {total:,} registros...")
        results = benchmark_to_single_file(
            n_total=total,
            output_path=args.output,
            step=args.step,
            monitor_memory=args.monitor,
            vehicle_type=args.type,
            bloom_error_rate=args.error_rate
        )
        plot_results(results, save_to="benchmark.png")
    else:
        print(f"Iniciando generación de {total:,} registros{'con monitoreo de memoria' if args.monitor else ''}...")
        run_single_generation(
            n=total,
            output_path=args.output,
            write_header=True,
            monitor_memory=args.monitor,
            vehicle_type=args.type,
            bloom_error_rate=args.error_rate
        )


if __name__ == "__main__":
    main()