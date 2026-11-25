"""Export the original No-Show view with precomputed, synthetic examples."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from flask import Flask, render_template

ROOT = Path(__file__).resolve().parent
# This build imports the original model without starting the Flask app factory.
from app.modules.no_show import model_utils


def export(destination: Path) -> None:
    app = Flask(__name__, template_folder=str(ROOT / "app" / "templates"))
    cases = [
        dict(patient_name="Örnek 1", poliklinik="Genel Muayene", age=64, sms_received=1, days_waiting=4, previous_noshow=0),
        dict(patient_name="Örnek 2", poliklinik="Kontrol", age=34, sms_received=1, days_waiting=28, previous_noshow=1),
        dict(patient_name="Örnek 3", poliklinik="Görüntüleme", age=42, sms_received=0, days_waiting=72, previous_noshow=3),
    ]
    with TemporaryDirectory() as temporary:
        # Always train the original fallback on synthetic data. Never load an existing pickle.
        model_utils.MODEL_PATH = str(Path(temporary) / "synthetic.pkl")
        samples = []
        for index, case in enumerate(cases, 1):
            result = model_utils.predict_no_show(case)
            if result.get("error"):
                raise RuntimeError(result["error"])
            samples.append(dict(case, **result, id=index, date_created=datetime(2026, 11, 25, tzinfo=timezone.utc)))

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "app/static/css/style.css", destination / "style.css")
    with app.test_request_context():
        for sample in samples:
            name = "index.html" if sample["id"] == 1 else f"ornek-{sample['id']}.html"
            html = render_template(
                "no_show.html", demo=True, selected_id=sample["id"],
                high_risk_appointments=samples, prediction_result=sample,
            )
            (destination / name).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
