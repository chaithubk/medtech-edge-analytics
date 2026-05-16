#!/usr/bin/env python3
"""
Generate realistic PhysioNet-format sample PSV files for testing.

Creates 10 patient records (5 healthy, 5 sepsis) with realistic vital sign
time series data matching ICU monitoring patterns.

PhysioNet 2019 Sepsis Challenge format:
- Pipe-separated values (|)
- One row per hour of monitoring
- Columns: HR, O2Sat, SBP, DBP, Temp, Resp, WBC, Lactate, Creatinine, SepsisLabel
- Missing values represented as empty strings
"""

from pathlib import Path

import numpy as np


def create_healthy_patient(patient_id):
    """Generate a healthy patient PSV file (no sepsis)."""
    hours = 24  # 24 hours of monitoring

    lines = ["HR|O2Sat|SBP|DBP|Temp|Resp|WBC|Lactate|Creatinine|SepsisLabel"]

    for hour in range(hours):
        # Healthy vitals with small random variation
        hr = np.random.normal(72, 5)  # Heart rate: 60-85 bpm
        o2sat = np.random.normal(98, 1)  # O2 saturation: 95-100%
        sbp = np.random.normal(118, 8)  # Systolic BP: 110-130 mmHg
        dbp = np.random.normal(76, 5)  # Diastolic BP: 70-85 mmHg
        temp = np.random.normal(37.0, 0.3)  # Temperature: 36.5-37.5°C
        resp = np.random.normal(16, 2)  # Respiratory rate: 12-20 breaths/min
        wbc = np.random.normal(7.0, 1.0)  # WBC: 4.5-11 K/uL
        lactate = np.random.normal(1.5, 0.3)  # Lactate: 0.5-2.0 mmol/L
        creatinine = np.random.normal(0.9, 0.2)  # Creatinine: 0.7-1.3 mg/dL
        sepsis_label = 0  # No sepsis

        line = f"{hr:.1f}|{o2sat:.1f}|{sbp:.1f}|{dbp:.1f}|{temp:.1f}|{resp:.1f}|{wbc:.1f}|{lactate:.2f}|{creatinine:.2f}|{sepsis_label}"
        lines.append(line)

    return "\n".join(lines)


def create_sepsis_patient(patient_id):
    """Generate a sepsis patient PSV file with progressive deterioration."""
    hours = 36  # 36 hours of monitoring

    lines = ["HR|O2Sat|SBP|DBP|Temp|Resp|WBC|Lactate|Creatinine|SepsisLabel"]

    for hour in range(hours):
        # Progressive deterioration pattern (sepsis development)
        # Hour 0-12: Early signs
        # Hour 12-24: Worsening
        # Hour 24-36: Severe sepsis

        progression = hour / hours  # 0 to 1

        # Heart rate increases (tachycardia)
        hr = 72 + (progression * 40) + np.random.normal(0, 3)

        # O2 saturation drops
        o2sat = 98 - (progression * 8) + np.random.normal(0, 1)

        # Blood pressure drops (septic shock pattern)
        sbp = 118 - (progression * 30) + np.random.normal(0, 3)
        dbp = 76 - (progression * 20) + np.random.normal(0, 2)

        # Temperature elevated (fever) then may drop (severe sepsis)
        if progression < 0.7:
            temp = 37.0 + (progression / 0.7 * 2.5) + np.random.normal(0, 0.2)
        else:
            temp = 39.5 - ((progression - 0.7) / 0.3 * 1.5) + np.random.normal(0, 0.2)

        # Respiratory rate increases (tachypnea)
        resp = 16 + (progression * 12) + np.random.normal(0, 1)

        # WBC elevated (immune response)
        wbc = 7.0 + (progression * 8) + np.random.normal(0, 0.5)

        # Lactate increases (tissue hypoperfusion)
        lactate = 1.5 + (progression * 3.5) + np.random.normal(0, 0.2)

        # Creatinine increases (renal dysfunction)
        creatinine = 0.9 + (progression * 1.2) + np.random.normal(0, 0.1)

        # SepsisLabel becomes 1 around hour 12-15 (when criteria met)
        sepsis_label = 1 if hour >= 12 else 0

        # Ensure values stay in realistic ranges
        hr = max(40, min(180, hr))
        o2sat = max(70, min(100, o2sat))
        sbp = max(40, min(180, sbp))
        dbp = max(20, min(120, dbp))
        temp = max(35, min(41, temp))
        resp = max(5, min(60, resp))
        wbc = max(0.5, min(30, wbc))
        lactate = max(0.5, min(20, lactate))
        creatinine = max(0.4, min(10, creatinine))

        line = f"{hr:.1f}|{o2sat:.1f}|{sbp:.1f}|{dbp:.1f}|{temp:.1f}|{resp:.1f}|{wbc:.1f}|{lactate:.2f}|{creatinine:.2f}|{sepsis_label}"
        lines.append(line)

    return "\n".join(lines)


def main():
    """Generate all sample PSV files."""

    output_dir = Path("data/physionet_sample/training_setA")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating PhysioNet sample dataset...")
    print()

    # Generate 5 healthy patients
    for i in range(1, 6):
        patient_id = f"p{i:05d}"
        content = create_healthy_patient(i)
        filepath = output_dir / f"{patient_id}.psv"
        filepath.write_text(content)
        print(f"✓ Created healthy patient: {filepath.name}")

    print()

    # Generate 5 sepsis patients
    for i in range(6, 11):
        patient_id = f"p{i:05d}"
        content = create_sepsis_patient(i)
        filepath = output_dir / f"{patient_id}.psv"
        filepath.write_text(content)
        print(f"✓ Created sepsis patient: {filepath.name}")

    print()
    print(f"Sample dataset created at: {output_dir}")
    print(f"Total patients: 10 (5 healthy, 5 sepsis)")
    print()
    print("Data characteristics:")
    print("  - Healthy patients: normal vitals, sepsis label = 0")
    print("  - Sepsis patients: progressive deterioration, sepsis label = 1 at hour 12+")
    print("  - Format: PhysioNet pipe-separated values (PSV)")
    print("  - Duration: 24-36 hours per patient")
    print()


if __name__ == "__main__":
    main()
