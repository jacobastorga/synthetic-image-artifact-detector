import json
from typing import List, Dict, Any

class AnnotationExporter:
    """Exports detected image defects into standard CVAT and Labelbox JSON schemas."""

    @staticmethod
    def to_cvat_json(image_id: int, image_name: str, width: int, height: int,
                     anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Converts anomaly bounding boxes into CVAT 1.1 JSON format."""
        shapes = []
        for idx, item in enumerate(anomalies):
            x, y, w, h = item["bbox"]
            shapes.append({
                "id": idx + 1,
                "type": "rectangle",
                "label": item["type"],
                "occluded": False,
                "points": [x, y, x + w, y + h],
                "attributes": [
                    {"name": "severity", "value": item.get("severity", "moderate")},
                    {"name": "score", "value": str(item.get("peak_gradient", item.get("gradient_loss", 0.0)))}
                ]
            })

        return {
            "version": 1.1,
            "image": {
                "id": image_id,
                "name": image_name,
                "width": width,
                "height": height
            },
            "annotations": shapes
        }

    @staticmethod
    def to_labelbox_ndjson(data_row_id: str, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts defects into Labelbox v2 NDJSON annotation objects."""
        records = []
        for item in anomalies:
            x, y, w, h = item["bbox"]
            records.append({
                "data_row": {"id": data_row_id},
                "schema_id": "defect-bounding-box-schema",
                "bbox": {"top": y, "left": x, "height": h, "width": w},
                "name": item["type"],
                "classifications": [
                    {"name": "severity", "answer": item.get("severity", "moderate")}
                ]
            })
        return records
