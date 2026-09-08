"""Libyan Municipal Waste Management AI System.

A single-file Streamlit application for citizen-submitted waste reports:
image upload, YOLOv8 detection, EXIF GPS enrichment, municipal mapping, and
severity-based dispatch prioritisation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import exifread
import folium
import streamlit as st
from PIL import Image
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from ultralytics import YOLO


APP_TITLE = "Libyan Municipal Waste Management AI System"
MODEL_NAME = "best.pt"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_NAME
CONFIDENCE_THRESHOLD = 0.25
FALLBACK_LATITUDE = 31.2089
FALLBACK_LONGITUDE = 16.5886
LIBYA_MAP_CENTER = (27.2, 17.3)

DATASET_LINKS = [
    ("TACO Dataset website", "https://tacodataset.org"),
    ("TACO Dataset GitHub", "https://github.com/pedropro/TACO"),
    ("HDX Libya GIS Data", "https://data.humdata.org/group/lby"),
    ("AerialWaste Dataset", "https://zenodo.org/records/7034382"),
    ("Waste Datasets Review", "https://github.com/AgaMiko/waste-datasets-review"),
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="WM",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading YOLOv8 detection model...")
def load_model() -> YOLO:
    """Load the uploaded waste-specific YOLO model once per session."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Custom model not found at {MODEL_PATH}. Upload it as best.pt."
        )
    return YOLO(str(MODEL_PATH))


def ratio_to_float(value: Any) -> float:
    """Convert an EXIF Ratio or numeric value to a float."""

    if hasattr(value, "num") and hasattr(value, "den"):
        if value.den == 0:
            raise ValueError("EXIF ratio has a zero denominator")
        return float(value.num) / float(value.den)
    return float(value)


def parse_gps_coordinate(values: Any, reference: Any) -> float:
    """Convert EXIF DMS values to signed decimal degrees."""

    if isinstance(values, (list, tuple)):
        values = list(values)
    else:
        raise ValueError("EXIF GPS coordinate is not a DMS sequence")

    degrees, minutes, seconds = [ratio_to_float(item) for item in values]
    decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if isinstance(reference, (list, tuple)):
        reference = reference[0] if reference else ""
    direction = str(reference).strip().upper()
    if direction in {"S", "W"}:
        decimal_degrees *= -1
    return decimal_degrees


def extract_gps(image_bytes: bytes) -> tuple[float | None, float | None]:
    """Read GPS latitude and longitude from image EXIF metadata."""

    try:
        tags = exifread.process_file(
            BytesIO(image_bytes),
            details=False,
        )
        latitude = tags.get("GPS GPSLatitude")
        latitude_reference = tags.get("GPS GPSLatitudeRef")
        longitude = tags.get("GPS GPSLongitude")
        longitude_reference = tags.get("GPS GPSLongitudeRef")

        if not all(
            (
                latitude,
                latitude_reference,
                longitude,
                longitude_reference,
            )
        ):
            return None, None

        return (
            parse_gps_coordinate(
                latitude.values,
                latitude_reference.values,
            ),
            parse_gps_coordinate(
                longitude.values,
                longitude_reference.values,
            ),
        )
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None, None


def severity_from_count(detection_count: int) -> int:
    """Map the number of detected objects to a 1–5 dispatch severity."""

    if detection_count == 0:
        return 1
    if detection_count <= 2:
        return 2
    if detection_count <= 5:
        return 3
    if detection_count <= 9:
        return 4
    return 5


def detection_summary(result: Any) -> list[dict[str, Any]]:
    """Extract a compact, serialisable detection list from a YOLO result."""

    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    names = getattr(result, "names", {})
    for box in boxes:
        class_index = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        if isinstance(names, dict):
            label = names.get(class_index, f"class {class_index}")
        else:
            label = str(class_index)
        detections.append(
            {
                "label": str(label),
                "confidence": round(confidence, 2),
            }
        )
    return detections


def build_heatmap(reports: list[dict[str, Any]]) -> folium.Map:
    """Build a Libya-centered weighted HeatMap from all submitted reports."""

    municipal_map = folium.Map(
        location=list(LIBYA_MAP_CENTER),
        zoom_start=5,
        min_zoom=4,
        max_bounds=True,
        control_scale=True,
        tiles="CartoDB positron",
    )

    heat_data = [
        [
            report["latitude"],
            report["longitude"],
            max(0.2, report["severity"] / 5),
        ]
        for report in reports
    ]

    if heat_data:
        HeatMap(
            heat_data,
            name="Waste report intensity",
            radius=28,
            blur=24,
            min_opacity=0.35,
            max_zoom=10,
            gradient={
                0.2: "#60a5fa",
                0.4: "#22c55e",
                0.6: "#facc15",
                0.8: "#f97316",
                1.0: "#dc2626",
            },
        ).add_to(municipal_map)

    severity_colors = {
        1: "green",
        2: "lightgreen",
        3: "orange",
        4: "red",
        5: "darkred",
    }
    for report in reports:
        popup_html = (
            f"<b>{report['report']}</b><br>"
            f"Detected items: {report['detected_items']}<br>"
            f"Severity: {report['severity']}/5<br>"
            f"GPS source: {report['gps_source']}<br>"
            f"Coordinates: {report['latitude']:.5f}, "
            f"{report['longitude']:.5f}"
        )
        folium.Marker(
            location=[report["latitude"], report["longitude"]],
            tooltip=(
                f"{report['report']} · "
                f"Severity {report['severity']}/5"
            ),
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(
                color=severity_colors.get(report["severity"], "red"),
                icon="trash",
                prefix="fa",
            ),
        ).add_to(municipal_map)

    folium.LayerControl(collapsed=False).add_to(municipal_map)
    return municipal_map


def file_signature(image_bytes: bytes) -> str:
    """Return a stable ID so Streamlit reruns do not duplicate reports."""

    return hashlib.sha256(image_bytes).hexdigest()[:12]


def add_report_if_new(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Persist a report in session state and return the canonical record."""

    if "reports" not in st.session_state:
        st.session_state.reports = []
    if "processed_signatures" not in st.session_state:
        st.session_state.processed_signatures = set()

    signature = report["signature"]
    if signature not in st.session_state.processed_signatures:
        st.session_state.reports.append(report)
        st.session_state.processed_signatures.add(signature)
        return report

    return next(
        saved_report
        for saved_report in st.session_state.reports
        if saved_report["signature"] == signature
    )


def render_sidebar() -> None:
    """Render project context, architecture, and public dataset references."""

    with st.sidebar:
        st.header("Project context")
        st.write(
            "A computer-vision workflow for turning citizen waste photos into "
            "geolocated, severity-ranked municipal dispatch reports."
        )

        st.subheader("Architecture")
        st.markdown(
            """
            1. **Citizen Portal** — image intake from JPG/PNG reports.
            2. **YOLOv8 inference** — waste detection with `best.pt`.
            3. **GPS enrichment** — EXIF coordinates when available.
            4. **Municipal Dashboard** — Libya-centered Folium HeatMap.
            5. **Dispatch queue** — session-persistent priority table.
            """
        )

        st.subheader("Detection settings")
        st.caption(
            f"Model: `{MODEL_NAME}`  \n"
            f"Confidence threshold: `{CONFIDENCE_THRESHOLD:.2f}`  \n"
            "HeatMap intensity is weighted by severity."
        )

        st.subheader("Dataset references")
        for label, url in DATASET_LINKS:
            st.markdown(f"- [{label}]({url})")

        st.subheader("GPS fallback")
        st.caption(
            f"{FALLBACK_LATITUDE:.4f}° N, {FALLBACK_LONGITUDE:.4f}° E "
            "(Sirte / Tripoli corridor reference)"
        )


def main() -> None:
    render_sidebar()

    st.title(APP_TITLE)
    st.caption(
        "AI-assisted citizen reporting and municipal dispatch prioritisation "
        "for cleaner Libyan communities."
    )

    citizen_column, municipal_column = st.columns([1, 1.25], gap="large")

    active_report: dict[str, Any] | None = None

    with citizen_column:
        st.header("Citizen Portal")
        st.write("Upload a street-level image to create a waste report.")
        uploaded_file = st.file_uploader(
            "Waste report image",
            type=["jpg", "jpeg", "png"],
            help="JPG and PNG images are supported. EXIF GPS data is used when present.",
        )

        if uploaded_file is None:
            st.info(
                "Upload a JPG or PNG image to run YOLOv8 detection and add a "
                "report to the municipal dispatch queue."
            )
        else:
            image_bytes = uploaded_file.getvalue()
            report_id = file_signature(image_bytes)

            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except (OSError, ValueError) as error:
                st.error(f"Could not read this image: {error}")
                image = None

            if image is not None:
                try:
                    model = load_model()
                    results = model(
                        image,
                        conf=CONFIDENCE_THRESHOLD,
                        verbose=False,
                    )
                    result = results[0]
                    detections = detection_summary(result)
                    annotated_image = Image.fromarray(result.plot()[..., ::-1])
                except Exception as error:
                    st.error(
                        "The YOLOv8 model could not process this image. "
                        f"Details: {error}"
                    )
                    detections = []
                    annotated_image = image

                latitude, longitude = extract_gps(image_bytes)
                has_exif_gps = latitude is not None and longitude is not None
                if not has_exif_gps:
                    latitude, longitude = FALLBACK_LATITUDE, FALLBACK_LONGITUDE

                severity = severity_from_count(len(detections))
                report = {
                    "signature": report_id,
                    "report": uploaded_file.name,
                    "submitted_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                    "detected_items": len(detections),
                    "severity": severity,
                    "latitude": round(latitude, 5),
                    "longitude": round(longitude, 5),
                    "gps_source": "EXIF GPS" if has_exif_gps else "Fallback",
                    "status": "Ready for dispatch",
                }
                active_report = add_report_if_new(report)

                st.image(
                    annotated_image,
                    caption="YOLOv8 annotated detection",
                    use_container_width=True,
                )

                metric_one, metric_two, metric_three = st.columns(3)
                metric_one.metric("Detected items", active_report["detected_items"])
                metric_two.metric(
                    "Severity index",
                    f"{active_report['severity']}/5",
                )
                metric_three.metric(
                    "GPS source",
                    active_report["gps_source"],
                )

                if detections:
                    st.write("Detected classes")
                    st.dataframe(
                        detections,
                        column_config={
                            "label": "Class",
                            "confidence": st.column_config.NumberColumn(
                                "Confidence",
                                format="%.2f",
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "No objects were detected above the confidence threshold. "
                        "The report remains in the queue at severity 1."
                    )

    with municipal_column:
        st.header("Municipal Dashboard")

        if active_report is None:
            if st.session_state.get("reports"):
                st.info(
                    "The map shows all reports collected in this session. "
                    "Upload another photo to add a new heat point."
                )
            else:
                st.info(
                    "The map is centered on Libya. Upload a report to add "
                    "a weighted heat point and dispatch marker."
                )
        else:
            st.success(
                f"Added {active_report['report']} to the Libya dispatch "
                f"HeatMap at severity {active_report['severity']}/5."
            )

        all_reports = st.session_state.get("reports", [])
        municipal_map = build_heatmap(all_reports)
        st_folium(municipal_map, use_container_width=True, height=430)

        location_one, location_two = st.columns(2)
        location_one.metric("Reports mapped", len(all_reports))
        location_two.metric(
            "Highest severity",
            (
                f"{max(report['severity'] for report in all_reports)}/5"
                if all_reports
                else "—"
            ),
        )

    st.divider()
    st.header("Priority Dispatch Queue")
    st.caption("Reports are ranked from highest to lowest severity in this session.")

    reports = sorted(
        st.session_state.get("reports", []),
        key=lambda item: (-item["severity"], item["submitted_at"]),
    )
    if reports:
        dispatch_rows = [
            {
                "Priority": index,
                "Report": item["report"],
                "Detected items": item["detected_items"],
                "Severity": f"{item['severity']}/5",
                "GPS source": item["gps_source"],
                "Latitude": item["latitude"],
                "Longitude": item["longitude"],
                "Submitted": item["submitted_at"],
                "Status": item["status"],
            }
            for index, item in enumerate(reports, start=1)
        ]
        st.dataframe(
            dispatch_rows,
            column_config={
                "Priority": st.column_config.NumberColumn("Priority", width="small"),
                "Detected items": st.column_config.NumberColumn(
                    "Detected items",
                    width="small",
                ),
                "Severity": st.column_config.TextColumn("Severity", width="small"),
                "Latitude": st.column_config.NumberColumn(
                    "Latitude",
                    format="%.5f",
                ),
                "Longitude": st.column_config.NumberColumn(
                    "Longitude",
                    format="%.5f",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No reports have been submitted yet.")

    st.caption(
        "Operational note: `best.pt` is the uploaded custom waste model. "
        "HeatMap weights use each report's 1–5 severity index; reports "
        "without EXIF GPS use the configured fallback coordinates."
    )


if __name__ == "__main__":
    main()