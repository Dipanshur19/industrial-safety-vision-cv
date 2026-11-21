# Benchmark Report

Generated: `2026-11-21T22:19:05.052361+00:00`

Mode: `real_model`

This benchmark used a real model checkpoint.

Hardware:

- Platform: `Windows-11-10.0.26300-SP0`
- Processor: `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`
- Python: `3.14.4`

| Backend | Device | Input size | Mean latency | P50 latency | P95 latency | FPS | Model size MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch_ultralytics | cpu | 640 | 70.259 ms | 69.569 ms | 72.973 ms | 14.23 | 5.353702545166016 |

Benchmark configuration:

- Warmup runs: `2`
- Benchmark runs: `5`

Real model benchmark command:

```bash
python scripts/run_benchmark.py --model yolo11n.pt --image docs/assets/demo_input.jpg
```

Mock pipeline benchmark command:

```bash
python scripts/run_benchmark.py --mock --image docs/assets/demo_input.jpg
```
