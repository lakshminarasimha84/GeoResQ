# --- Multimodal Hazard Engine (satellite + terrain, ground sensor optional) ---

def classify_hazard(coldness: float, moisture: float, ruggedness: float, ground_rain_rate: float = 0.0):
    """
    coldness: 0.0 to 1.0 (from TIR-1 brightness temp drop)
    moisture: 0.0 to 1.0 (from Water Vapor band)
    ruggedness: 0.0 to 1.0 (from CartoDEM gradient)
    ground_rain_rate: mm/hr from ground AWS / telemetry (optional — defaults to 0 if not wired in yet)
    """
    # 1. Atmospheric severity score (Satellite ConvLSTM)
    sat_score = (coldness * 0.6) + (moisture * 0.4)

    # 2. Ground vulnerability score
    # High ruggedness = flash runoff; flat terrain = poor drainage/ponding
    terrain_factor = ruggedness if ruggedness > 0.25 else 0.5

    # 3. Multimodal fusion score (Satellite + Terrain + Ground Gauge)
    sensor_normalized = min(ground_rain_rate / 100.0, 1.0)
    fusion_score = (0.50 * sat_score) + (0.20 * terrain_factor) + (0.30 * sensor_normalized)

    # Dynamic classification
    if fusion_score >= 0.70 or ground_rain_rate >= 60.0:
        category = "FLASH FLOOD / CLOUDBURST"
        severity = "CRITICAL"
    elif fusion_score >= 0.45 or ground_rain_rate >= 20.0:
        category = "SEVERE THUNDERSTORM"
        severity = "WARNING"
    elif fusion_score >= 0.25:
        category = "MODERATE RAINFALL"
        severity = "WATCH"
    else:
        category = "NORMAL / LOW RISK"
        severity = "ADVISORY"

    return category, severity, round(fusion_score, 2)


def generate_operator_summary(category, severity, fusion_score, coldness, moisture, ruggedness, sensor_val: float = 0.0):
    """
    Builds an evidence-linked summary + structured payload for the operator dashboard.
    sensor_val defaults to 0.0 so this works even before ground sensors are wired in.
    """
    drivers = []
    if coldness >= 0.50:
        drivers.append(f"rapid cloud-top cooling ({coldness*100:.0f}%) indicating vertical updrafts")
    if moisture >= 0.45:
        drivers.append(f"high mid-tropospheric water vapor ({moisture*100:.0f}%)")
    if sensor_val > 15.0:
        drivers.append(f"ground telemetry surge (+{sensor_val} mm/hr)")
    elif ruggedness < 0.15:
        drivers.append(f"flat terrain basin ({ruggedness:.2f}) causing high water accumulation risk")

    drivers_text = ", ".join(drivers) if drivers else "baseline seasonal atmospheric patterns"

    summary_text = (
        f"[{severity}] {category} flagged with {int(fusion_score * 100)}% overall confidence. "
        f"Primary triggers: {drivers_text}."
    )

    evidence_payload = {
        "hazard_category": category,
        "severity": severity,
        "confidence_score": fusion_score,
        "summary": summary_text,
        "underlying_evidence": {
            "satellite_cloud_coldness": coldness,
            "satellite_moisture_index": moisture,
            "cartodem_ruggedness": ruggedness,
            "ground_sensor_reading": f"{sensor_val} mm/hr",
        },
        "operator_action": (
            "Issue evacuation advisory for low-lying areas"
            if severity == "CRITICAL"
            else "Monitor next 30-min MOSDAC update"
        ),
    }

    return evidence_payload


if __name__ == "__main__":
    # Quick standalone test using the exact values from your terminal run.
    # ground_rain_rate / sensor_val left at 0.0 since ground data isn't wired in yet.
    import json

    coldness, moisture, ruggedness = 0.54, 0.48, 0.08

    category, severity, fusion_score = classify_hazard(coldness, moisture, ruggedness)
    result = generate_operator_summary(category, severity, fusion_score, coldness, moisture, ruggedness)

    print(json.dumps(result, indent=2))
