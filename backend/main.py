import cv2
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from backend.vision_runtime import VisionRuntime
from backend.site_config import SITES
from backend.ml_layer import CongestionForecaster

app = FastAPI(title="BharatPravah Prototype (Multi-Site)")

runtimes: dict[str, VisionRuntime] = {}
forecaster = CongestionForecaster(model_path="models/forecaster.pkl")


@app.on_event("startup")
def startup_event():
    # Start one runtime per site
    for site_id, cfg in SITES.items():
        rt = VisionRuntime(
            site_id=site_id,
            video_path=cfg["video_path"],
            rois=cfg["rois"],
            resize_w=960,
            resize_h=540,
        )
        rt.start(
            detect_every_n_frames=2,
            metrics_every_sec=1.0,
            recommendation_every_sec=10.0,  # Phase 4 update rate
        )
        runtimes[site_id] = rt


@app.get("/")
def root():
    return {
        "message": "BharatPravah backend running (multi-video + metrics + recommendation + ML)",
        "sites": list(SITES.keys()),
        "ml_model_loaded": forecaster.loaded,
        "ml_model_error": forecaster.load_error,
    }


def get_runtime_or_404(site_id: str):
    rt = runtimes.get(site_id)
    if rt is None:
        return None, JSONResponse(
            {"error": f"Unknown site_id '{site_id}'", "valid": list(runtimes.keys())},
            status_code=404,
        )
    return rt, None


@app.get("/detections")
def detections(site_id: str = Query(...)):
    rt, err = get_runtime_or_404(site_id)
    if err:
        return err
    _, dets = rt.get_latest()
    return JSONResponse({"site_id": site_id, "detections": dets})


@app.get("/metrics")
def metrics(site_id: str = Query(...)):
    rt, err = get_runtime_or_404(site_id)
    if err:
        return err

    payload = rt.get_latest_metrics_payload()
    if payload is None:
        return JSONResponse(
            {"status": "warming_up", "site_id": site_id, "message": "Collecting initial metrics..."}
        )
    return JSONResponse(payload)


@app.get("/recommendation")
def recommendation(site_id: str = Query(...)):
    rt, err = get_runtime_or_404(site_id)
    if err:
        return err

    reco = rt.get_latest_recommendation()
    if reco is None:
        return JSONResponse(
            {"status": "warming_up", "site_id": site_id, "message": "Building first recommendation..."}
        )
    return JSONResponse(reco)


@app.get("/ml/predict")
def ml_predict(site_id: str = Query(...)):
    rt, err = get_runtime_or_404(site_id)
    if err:
        return err

    history = rt.get_metrics_history()
    result = forecaster.predict(history, approaches=("north", "south"))
    result["site_id"] = site_id
    return JSONResponse(result)


def draw_boxes(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def mjpeg_detect_generator(site_id: str):
    rt = runtimes.get(site_id)
    if rt is None:
        return

    while True:
        frame, dets = rt.get_latest()
        if frame is None:
            continue

        out = draw_boxes(frame.copy(), dets)
        ok, jpeg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.get("/stream/detect")
def stream_detect(site_id: str = Query(...)):
    rt, err = get_runtime_or_404(site_id)
    if err:
        return err
    return StreamingResponse(
        mjpeg_detect_generator(site_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/ui", response_class=HTMLResponse)
def ui():
    options_html = "".join(
        [f'<option value="{sid}">{cfg["name"]} ({sid})</option>' for sid, cfg in SITES.items()]
    )

    html = f"""
    <html>
      <head>
        <title>BharatPravah – Smart Traffic Dashboard</title>
        <style>
          body {{ font-family: Arial; margin: 20px; background: #fafafa; }}
          .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }}
          .card {{ padding: 14px; border: 1px solid #ddd; border-radius: 10px; background: white; }}
          img {{ width: 100%; border-radius: 10px; }}
          .small {{ color: #666; font-size: 13px; }}
          .row {{ display:flex; gap:12px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }}
          select {{ padding:8px; border-radius:8px; border:1px solid #ccc; }}
          .stack {{ display:flex; flex-direction:column; gap:12px; }}
          ul {{ margin: 6px 0 0 18px; }}
        </style>
      </head>

      <body>
        <h2>BharatPravah – AI-Powered Traffic Intelligence</h2>

        <div class="row">
          <label><b>Select Intersection:</b></label>
          <select id="siteSelect">
            {options_html}
          </select>
        </div>

        <div class="small" id="currentSiteLine">Currently viewing: —</div>
        <div class="small" id="statusLine" style="margin-bottom:14px;">System status: —</div>

        <div class="grid">
          <div class="card">
            <p><b>Live Traffic Feed (Computer Vision)</b></p>
            <img id="streamImg" src="" />
            <p class="small">Live detections using existing CCTV footage.</p>
          </div>

          <div class="stack">
            <div class="card">
              <p><b>Current Congestion Metrics</b></p>
              <div id="metricsBox">Loading…</div>
            </div>

            <div class="card">
              <p><b>Recommended Signal Timing</b> <span class="small">(updates every 10s)</span></p>
              <div id="recoBox">Loading…</div>
            </div>

            <div class="card">
              <p><b>ML Forecast – Next 30 Seconds</b></p>
              <div id="mlBox">Loading…</div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-top:16px;">
          <p><b>Demo Notes</b></p>
          <ul class="small">
            <li>Prototype uses recorded traffic video for demonstration.</li>
            <li>All analytics are computed in real time using AI.</li>
            <li>No additional hardware required (software-only).</li>
            <li>Designed for Smart City traffic control rooms.</li>
          </ul>
        </div>

        <script>
          const siteSelect = document.getElementById("siteSelect");
          const streamImg = document.getElementById("streamImg");

          function labelScore(v) {{
            if (v < 0.30) return "Low";
            if (v < 0.60) return "Moderate";
            return "High";
          }}

          function setSite(siteId) {{
            streamImg.src = `/stream/detect?site_id=${{siteId}}`;
            const text = siteSelect.options[siteSelect.selectedIndex].text;
            document.getElementById("currentSiteLine").innerText =
              `Currently viewing: ${{text}}`;
          }}

          async function loadStatusOnce() {{
            try {{
              const res = await fetch(`/`);
              const data = await res.json();
              document.getElementById("statusLine").innerText =
                `System status: Running | ML Model Loaded: ${{data.ml_model_loaded ? "Yes" : "No"}}`;
            }} catch {{
              document.getElementById("statusLine").innerText = "System status: Unknown";
            }}
          }}

          async function refreshMetrics() {{
            const siteId = siteSelect.value;
            const res = await fetch(`/metrics?site_id=${{siteId}}`);
            const data = await res.json();
            if (data.status === "warming_up") {{
              document.getElementById("metricsBox").innerText = "Warming up…";
              return;
            }}

            let html = "";
            for (const k in data.metrics) {{
              const m = data.metrics[k];
              html += `
                <div style="margin-bottom:8px;">
                  <b>${{k.toUpperCase()}}</b><br/>
                  Vehicles: ${{m.vehicles}}<br/>
                  Congestion: ${{m.congestion_score.toFixed(2)}} (${{labelScore(m.congestion_score)}})
                </div>
              `;
            }}
            document.getElementById("metricsBox").innerHTML = html;
          }}

          async function refreshRecommendation() {{
            const siteId = siteSelect.value;
            const res = await fetch(`/recommendation?site_id=${{siteId}}`);
            const data = await res.json();
            if (data.status === "warming_up") {{
              document.getElementById("recoBox").innerText = "Warming up…";
              return;
            }}

            let html = `<div>Cycle: <b>${{data.cycle_s}}s</b></div>`;
            for (const k in data.greens) {{
              html += `<div>${{k.toUpperCase()}} Green: <b>${{data.greens[k]}}s</b></div>`;
            }}
            html += `<div class="small">Reason: ${{data.reason}}</div>`;
            document.getElementById("recoBox").innerHTML = html;
          }}

          async function refreshML() {{
            const siteId = siteSelect.value;
            const res = await fetch(`/ml/predict?site_id=${{siteId}}`);
            const data = await res.json();
            if (data.status !== "ok") {{
              document.getElementById("mlBox").innerText = "ML model warming up…";
              return;
            }}

            const f = data.forecast;
            document.getElementById("mlBox").innerHTML = `
              NORTH: <b>${{f.north.toFixed(2)}}</b> (${{labelScore(f.north)}})<br/>
              SOUTH: <b>${{f.south.toFixed(2)}}</b> (${{labelScore(f.south)}})
            `;
          }}

          siteSelect.addEventListener("change", () => {{
            setSite(siteSelect.value);
            refreshMetrics();
            refreshRecommendation();
            refreshML();
          }});

          // Init
          setSite(siteSelect.value);
          loadStatusOnce();
          setInterval(refreshMetrics, 1000);
          setInterval(refreshRecommendation, 2000);
          setInterval(refreshML, 2000);
          refreshMetrics();
          refreshRecommendation();
          refreshML();
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
