"""
DetectAI - Full Pipeline Integration Test
Tests:
1. /health
2. / (UI rendering)
3. /stats
4. /history
5. /predict with sample_face.jpg (full face detection, feature extraction, Grad-CAM, ELA, and RF inference)
6. /report/{id} (PDF forensic report generation)
"""

import os
from pathlib import Path
from fastapi.testclient import TestClient
import app as main_app

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_FACE = BASE_DIR / "sample_face.jpg"


def run_full_pipeline_tests():
    with TestClient(main_app.app) as client:
        print("[*] Testing /health...")
        res = client.get("/health")
        assert res.status_code == 200
        print("    Health:", res.json())

        print("[*] Testing / (UI Index)...")
        res = client.get("/")
        assert res.status_code == 200
        assert "DetectAI" in res.text

        print("[*] Testing /predict with sample_face.jpg...")
        assert SAMPLE_FACE.exists(), "sample_face.jpg missing!"
        with open(SAMPLE_FACE, "rb") as f:
            res_pred = client.post(
                "/predict",
                files={"file": ("sample_face.jpg", f, "image/jpeg")},
                data={"sensitivity": 50},
            )

        assert res_pred.status_code == 200, f"Predict failed: {res_pred.text}"
        data = res_pred.json()
        print("    Prediction Verdict:", data["label"], f"({data['confidence']}%)")
        print("    CNN fake probability:", data["cnn_probability"])
        print("    RF fake probability:", data["rf_probability"])
        print("    Combined fake probability:", data["fake_probability"])
        print("    Forensic radar features:", data["forensic_features"])
        print("    Heatmap image generated:", len(data["heatmap_image"]) > 50)
        print("    ELA image generated:", len(data["ela_image"]) > 50)
        print("    Face crop image generated:", len(data["face_crop_image"]) > 50)
        record_id = data["id"]
        assert record_id is not None

        print(f"[*] Testing /report/{record_id} PDF Generation...")
        res_pdf = client.get(f"/report/{record_id}")
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"
        assert len(res_pdf.content) > 1000
        print(f"    PDF Report generated successfully! Size: {len(res_pdf.content)} bytes")

        print("[*] Testing /stats updated totals...")
        res_stats = client.get("/stats")
        stats_data = res_stats.json()
        assert stats_data["total_scans"] >= 1
        print("    Stats:", stats_data)

        print("[*] Testing /history contains record...")
        res_hist = client.get("/history")
        hist_data = res_hist.json()
        assert len(hist_data["history"]) >= 1
        print(f"    History retrieved {len(hist_data['history'])} records.")

        print("\n==================================================")
        print("   ALL DETECTAI PIPELINE TESTS PASSED 100%!")
        print("==================================================")


if __name__ == "__main__":
    run_full_pipeline_tests()
