from config import cursor

def process_sensor_readings(image_id: str):
    """
    Returns recent sensor readings linked to a crop image.
    """
    cursor.execute("""
        SELECT sensor_id, value, metadata, reading_time
        FROM public.sensor_readings
        WHERE linked_image_id = %s
        ORDER BY reading_time DESC
        LIMIT 5
    """, (image_id,))
    
    readings = cursor.fetchall()
    processed = []
    for sensor_id, value, metadata, time in readings:
        processed.append({
            "sensor_id": str(sensor_id),
            "value": float(value),
            "metadata": metadata,
            "time": str(time)
        })
    return processed
