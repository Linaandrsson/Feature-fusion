"""Extract embeddings for all configured sensors and variants."""

import subprocess
import sys
from pathlib import Path

from config_extraction import SENSORS, variant_names


def main():
    script_dir = Path(__file__).parent.resolve()
    extract_script = script_dir / "extract_features.py"
    total_jobs = len(variant_names) * len(SENSORS)

    print(f"\n{'=' * 80}")
    print("Extract embeddings for all sensors and variants")
    print(f"Sensors: {len(SENSORS)}, Variants: {len(variant_names)}, Total jobs: {total_jobs}")
    print(f"{'=' * 80}\n")

    success = []
    failed = []
    job_idx = 0

    for variant in variant_names:
        print(f"\n{'-' * 80}")
        print(f"Variant: {variant}")
        print(f"{'-' * 80}")
        for sensor_name in SENSORS.keys():
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Sensor={sensor_name}, Variant={variant}")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(extract_script),
                        "--sensor",
                        sensor_name,
                        "--variant",
                        variant,
                    ],
                    check=True,
                    capture_output=False,
                    text=True,
                )
                success.append((sensor_name, variant))
            except subprocess.CalledProcessError:
                print(f"  Failed: sensor={sensor_name}, variant={variant}")
                failed.append((sensor_name, variant))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Success: {len(success)}/{total_jobs}")
    if failed:
        print(f"Failed: {len(failed)}/{total_jobs}")
        for sensor_name, variant in failed:
            print(f"  - sensor={sensor_name}, variant={variant}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
