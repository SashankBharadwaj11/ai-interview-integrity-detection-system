
def compute_severity(alert_type: str, ctx: dict) -> str:
    """
    Returns one of: 'info', 'low', 'medium', 'high', 'critical'
    based on alert type and contextual metrics.
    """
    if alert_type == "FACE_MISSING":
        # seconds without face
        duration = ctx.get("duration_sec", 0.0)
        if duration > 30:
            return "critical"
        elif duration > 10:
            return "high"
        elif duration > 3:
            return "medium"
        else:
            return "low"

    if alert_type == "GAZE_AWAY":
        dur = ctx.get("duration_sec", 0.0)
        direction = ctx.get("direction", "center")
        if dur > 5:
            return "high"
        elif dur > 2:
            return "medium"
        else:
            return "low"

    if alert_type == "MULTI_FACE":
        n = ctx.get("num_faces", 0)
        if n >= 3:
            return "critical"
        elif n == 2:
            return "high"
        else:
            return "medium"

    if alert_type == "OBJECT_DETECTED":
        label = ctx.get("label", "")
        if label in ("cell phone", "phone", "mobile"):
            return "critical"
        elif label in ("book", "notebook"):
            return "high"
        else:
            return "medium"

    return "medium"
