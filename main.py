"""
TerraWatch — SIH26001 live landslide risk backend.

Real, continuously-running service (not a demo/replay):
  - Pulls REAL rainfall data every 15 minutes from Open-Meteo (free, no API key,
    ECMWF/GFS-based, works for any India coordinates today).
  - Computes the exact static_score / rain_trigger / risk_score formulas from
    the SIH26001 technical blueprint.
  - Runs the job on Asia/Kolkata (IST) time via APScheduler.
  - Persists every reading + alert to SQLite so history survives restarts.
  - Serves the same REST contract the prototype's "API console" mocked:
      GET  /risk/current
      GET  /risk/history?cell_id=&hours=
      GET  /risk/{cell_id}/explanation
      GET  /exposure/priority
      GET  /alerts
      POST /feedback
      GET  /health

Swap-in points for production, clearly marked below:
  - RAINFALL SOURCE: replace fetch_live_rainfall() with an IMD API call once
    your team has MoES/IMD API credentials — same return shape, nothing else changes.
  - STATIC SUSCEPTIBILITY: CELLS[*]["slope"/"relief"/... ] are placeholder terrain
    values. Replace build_grid() with real GSI/DEM-derived layers when available.
  - MODEL PROBABILITY: model_probability() is a seeded stand-in for a trained
    XGBoost classifier (per the blueprint's evaluation plan). Train one on the
    GSI/ISRO landslide inventory and swap the function body — the rest of the
    pipeline (fusion, alerting, API) does not need to change.
  - SMS: send_sms() only logs to the database right now. Plug in an
    authenticated Twilio/MSG91 client (with DLT-registered sender ID) there.
"""

import math
import sqlite3
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = "terrawatch.db"
db_lock = threading.Lock()

# ============================================================
# Pilot grid — East Khasi Hills corridor, Meghalaya (same pilot
# geography as the SIH26001 blueprint / TerraWatch prototype).
# Terrain values are placeholders — see module docstring.
# ============================================================
CENTER = {"lat": 25.2840, "lon": 91.7360}
GRID, STEP = 6, 0.012


def _seeded(n: int) -> float:
    """Deterministic pseudo-random in [0,1) so terrain layout is reproducible."""
    x = math.sin(n * 12.9898) * 43758.5453
    return x - math.floor(x)


def build_grid():
    cells = []
    cid = 0
    for r in range(GRID):
        for c in range(GRID):
            lat0 = CENTER["lat"] + (r - GRID / 2) * STEP
            lon0 = CENTER["lon"] + (c - GRID / 2) * STEP
            ridge = abs(r - c) / GRID
            slope = min(1, max(0, 0.85 - ridge * 1.3 + (_seeded(cid * 3 + 1) - 0.5) * 0.25))
            relief = min(1, max(0, slope * 0.7 + (_seeded(cid * 3 + 2) - 0.5) * 0.3))
            soil = min(1, max(0, 0.3 + _seeded(cid * 3 + 3) * 0.5))
            landcover = min(1, max(0, 0.25 + _seeded(cid * 5 + 1) * 0.55))
            drainage = min(1, max(0, 0.3 + _seeded(cid * 5 + 2) * 0.6))
            hist = min(1, max(0, max(0, slope - 0.35) * (0.6 + _seeded(cid * 5 + 3) * 0.8)))
            cells.append({
                "id": cid, "r": r, "c": c,
                "lat": lat0 + STEP / 2, "lon": lon0 + STEP / 2,
                "slope": slope, "relief": relief, "soil": soil,
                "landcover": landcover, "drainage": drainage, "hist": hist,
            })
            cid += 1
    return cells


CELLS = build_grid()

ASSETS = [
    {"id": "a1", "name": "NH-206 cut-slope, Sohra approach", "type": "Road (NH)", "lat": 25.2810, "lon": 91.7300, "w": 0.95},
    {"id": "a2", "name": "PHC Sohra", "type": "Health facility", "lat": 25.2840, "lon": 91.7395, "w": 0.90},
    {"id": "a3", "name": "Mawkyrwat link road", "type": "Road (link)", "lat": 25.2900, "lon": 91.7460, "w": 0.65},
    {"id": "a4", "name": "Govt. Secondary School, Sohra", "type": "School", "lat": 25.2790, "lon": 91.7420, "w": 0.80},
    {"id": "a5", "name": "Pynursla settlement cluster", "type": "Settlement", "lat": 25.2930, "lon": 91.7250, "w": 0.70},
    {"id": "a6", "name": "Shillong–Dawki highway bridge", "type": "Bridge", "lat": 25.2760, "lon": 91.7350, "w": 0.85},
]

# ============================================================
# Config — same tunable weights as the blueprint / prototype admin panel
# ============================================================
CONFIG = {
    "static_w": {"slope": 0.30, "relief": 0.20, "soil": 0.15, "landcover": 0.15, "drainage": 0.10, "hist": 0.10},
    "fusion": {"model": 0.60, "static": 0.25, "rain": 0.15},
    "cutoff": {"yellow": 0.25, "orange": 0.50, "red": 0.75},
    "persist_runs": 2,   # consecutive high refreshes required before Orange/Red is confirmed
    "poll_minutes": 15,
}


# ============================================================
# DB
# ============================================================
@contextmanager
def db():
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS risk_current (
            cell_id INTEGER PRIMARY KEY,
            static_score REAL, rain_trigger REAL, model_prob REAL, risk_score REAL,
            level TEXT, consecutive_high INTEGER DEFAULT 0, pending_level TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id INTEGER, risk_score REAL, level TEXT,
            intensity_mm_hr REAL, duration_hr REAL, ts TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id INTEGER, level TEXT, risk_score REAL,
            entered_at TEXT, active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS sms_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            to_number TEXT, body TEXT, sent_at TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id INTEGER, was_useful INTEGER, note TEXT, ts TEXT
        );
        CREATE TABLE IF NOT EXISTS rainfall_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, hourly_json TEXT
        );
        """)


# ============================================================
# REAL live rainfall — Open-Meteo, no API key required.
# Swap this function for an IMD client later; keep the return shape.
# ============================================================
def fetch_live_rainfall():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CENTER["lat"], "longitude": CENTER["lon"],
        "hourly": "precipitation",
        "past_days": 3, "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    times = data["hourly"]["time"]
    precip = data["hourly"]["precipitation"]  # mm per hour, real observed+forecast blend
    with db() as conn:
        conn.execute("INSERT INTO rainfall_cache (fetched_at, hourly_json) VALUES (?, ?)",
                     (datetime.now(IST).isoformat(), str(list(zip(times, precip)))))
    return times, precip


def threshold_intensity(duration_hr: float) -> float:
    """NE Himalaya region-specific I-D threshold: I = 5.8294 * D^-0.4141 (mm/hr)."""
    d = max(duration_hr, 0.05)
    return 5.8294 * (d ** -0.4141)


def compute_rain_trigger(times, precip):
    now_idx = len(precip) - 1
    # find current "storm duration": walk back while hourly rainfall is nonzero-ish
    duration = 0
    i = now_idx
    while i >= 0 and precip[i] > 0.1 and duration < 72:
        duration += 1
        i -= 1
    duration = max(duration, 0.25)
    intensity_1h = precip[now_idx]
    r24 = sum(precip[max(0, now_idx - 23):now_idx + 1])
    r72 = sum(precip[max(0, now_idx - 71):now_idx + 1])
    thr = threshold_intensity(duration)
    ratio = intensity_1h / thr if thr > 0 else 0
    trig = min(1.0, max(0.0,
        1 / (1 + math.exp(-1.6 * (ratio - 1))) * 0.65
        + min(1.0, r24 / 140) * 0.20
        + min(1.0, r72 / 220) * 0.15))
    return {"trig": trig, "I": intensity_1h, "D": duration, "threshold": thr, "r24": r24, "r72": r72}


def static_score(cell):
    w = CONFIG["static_w"]
    return (w["slope"] * cell["slope"] + w["relief"] * cell["relief"] + w["soil"] * cell["soil"]
            + w["landcover"] * cell["landcover"] + w["drainage"] * cell["drainage"] + w["hist"] * cell["hist"])


def model_probability(cell, s, trig):
    """Seeded stand-in for a trained classifier — see module docstring."""
    interaction = s * trig * 1.4
    noise = (_seeded(cell["id"] * 97 + int(time.time() // 900)) - 0.5) * 0.06
    x = 4.2 * (0.55 * trig + 0.45 * s + interaction - 0.62)
    return min(1.0, max(0.0, 1 / (1 + math.exp(-x)) + noise))


def classify(risk):
    if risk >= CONFIG["cutoff"]["red"]:
        return "Red"
    if risk >= CONFIG["cutoff"]["orange"]:
        return "Orange"
    if risk >= CONFIG["cutoff"]["yellow"]:
        return "Yellow"
    return "Green"


def explain(cell, trig, r24):
    factors = [
        ("24h rainfall", min(1, r24 / 140), f"24-hour rainfall reached {r24:.0f} mm, exceeding the local accumulation reference"),
        ("slope", cell["slope"], "the cell sits on a steep escarpment slope"),
        ("historical proximity", cell["hist"], "a previous landslide is within the historical search radius"),
        ("land disturbance", cell["landcover"], "satellite bare-soil/vegetation-loss index is elevated"),
        ("drainage", 1 - cell["drainage"], "the cell is close to a natural drainage line"),
    ]
    factors.sort(key=lambda f: -f[1])
    top = factors[:2]
    return {"chips": [f[0] for f in top], "text": f"Risk rose because {top[0][2]}, and {top[1][2]}."}


def send_sms(to_number, body):
    """Logged only — plug in a real Twilio/MSG91 client here with your DLT-registered sender ID."""
    with db() as conn:
        conn.execute("INSERT INTO sms_log (to_number, body, sent_at) VALUES (?, ?, ?)",
                     (to_number, body, datetime.now(IST).isoformat()))


# ============================================================
# The scheduled job — this is what makes the system "live"
# ============================================================
def refresh_risk_job():
    now = datetime.now(IST).isoformat()
    try:
        times, precip = fetch_live_rainfall()
    except Exception as e:
        print(f"[{now}] rainfall fetch failed: {e}")
        return
    rain = compute_rain_trigger(times, precip)

    with db() as conn:
        for cell in CELLS:
            s = static_score(cell)
            mp = model_probability(cell, s, rain["trig"])
            f = CONFIG["fusion"]
            risk = f["model"] * mp + f["static"] * s + f["rain"] * rain["trig"]
            risk = min(1.0, max(0.0, risk))
            level = classify(risk)

            row = conn.execute("SELECT * FROM risk_current WHERE cell_id=?", (cell["id"],)).fetchone()
            consecutive = row["consecutive_high"] if row else 0
            pending = row["pending_level"] if row else None
            prev_level = row["level"] if row else "Green"

            if level in ("Orange", "Red"):
                consecutive = consecutive + 1 if pending == level else 1
                pending = level
                if consecutive < CONFIG["persist_runs"]:
                    level = prev_level if prev_level in ("Orange", "Red") else "Yellow"
            else:
                consecutive, pending = 0, level

            conn.execute("""
                INSERT INTO risk_current (cell_id, static_score, rain_trigger, model_prob, risk_score,
                    level, consecutive_high, pending_level, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cell_id) DO UPDATE SET
                    static_score=excluded.static_score, rain_trigger=excluded.rain_trigger,
                    model_prob=excluded.model_prob, risk_score=excluded.risk_score,
                    level=excluded.level, consecutive_high=excluded.consecutive_high,
                    pending_level=excluded.pending_level, updated_at=excluded.updated_at
            """, (cell["id"], s, rain["trig"], mp, risk, level, consecutive, pending, now))

            conn.execute("""
                INSERT INTO risk_history (cell_id, risk_score, level, intensity_mm_hr, duration_hr, ts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cell["id"], risk, level, rain["I"], rain["D"], now))

            if level in ("Orange", "Red") and prev_level != level:
                conn.execute("""
                    INSERT INTO alerts (cell_id, level, risk_score, entered_at, active)
                    VALUES (?, ?, ?, ?, 1)
                """, (cell["id"], level, risk, now))
                send_sms(f"+91-98XXX{1000+cell['id']:04d}"[-14:],
                         f"[TerraWatch] Cell R{cell['r']}C{cell['c']} is now {level.upper()}. "
                         f"risk={risk:.2f}. Follow district advisory. ({now} IST)")
            if level == "Green" and prev_level in ("Orange", "Red"):
                conn.execute("UPDATE alerts SET active=0 WHERE cell_id=? AND active=1", (cell["id"],))

    print(f"[{now}] refresh complete — I={rain['I']:.1f}mm/hr D={rain['D']:.1f}hr trig={rain['trig']:.2f}")


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="TerraWatch Live — SIH26001")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

scheduler = BackgroundScheduler(timezone=IST)


@app.on_event("startup")
def startup():
    init_db()
    refresh_risk_job()  # populate immediately so /risk/current isn't empty on first load
    scheduler.add_job(refresh_risk_job, "interval", minutes=CONFIG["poll_minutes"], id="refresh", next_run_time=datetime.now(IST) + timedelta(minutes=CONFIG["poll_minutes"]))
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown(wait=False)


@app.get("/health")
def health():
    return {"status": "ok", "server_time_ist": datetime.now(IST).isoformat(), "poll_minutes": CONFIG["poll_minutes"]}


@app.get("/risk/current")
def risk_current():
    with db() as conn:
        rows = conn.execute("SELECT * FROM risk_current ORDER BY cell_id").fetchall()
    by_id = {r["cell_id"]: dict(r) for r in rows}
    out = []
    for cell in CELLS:
        d = by_id.get(cell["id"], {})
        out.append({
            "cell_id": cell["id"], "r": cell["r"], "c": cell["c"],
            "lat": cell["lat"], "lon": cell["lon"],
            "level": d.get("level", "Green"), "risk_score": round(d.get("risk_score", 0) or 0, 3),
            "static_susceptibility": round(d.get("static_score", 0) or 0, 3),
            "rain_trigger": round(d.get("rain_trigger", 0) or 0, 3),
            "model_probability": round(d.get("model_prob", 0) or 0, 3),
        })
    return {"generated_at_ist": datetime.now(IST).isoformat(), "cells": out}


@app.get("/risk/history")
def risk_history(cell_id: int = 0, hours: int = 48):
    since = (datetime.now(IST) - timedelta(hours=hours)).isoformat()
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM risk_history WHERE cell_id=? AND ts>=? ORDER BY ts", (cell_id, since)
        ).fetchall()
    return {"cell_id": cell_id, "points": [dict(r) for r in rows]}


@app.get("/risk/{cell_id}/explanation")
def risk_explanation(cell_id: int):
    if cell_id < 0 or cell_id >= len(CELLS):
        raise HTTPException(404, "unknown cell_id")
    cell = CELLS[cell_id]
    with db() as conn:
        row = conn.execute("SELECT * FROM risk_current WHERE cell_id=?", (cell_id,)).fetchone()
    if not row:
        raise HTTPException(404, "no data yet — wait for the first refresh cycle")
    ex = explain(cell, row["rain_trigger"], 0)
    return {
        "cell_id": cell_id, "risk_score": round(row["risk_score"], 3), "level": row["level"],
        "top_factors": ex["chips"], "narrative": ex["text"],
        "confidence": {"label_confidence": "medium", "data_freshness": row["updated_at"],
                       "rainfall_quality": "Open-Meteo live", "model_confidence": "medium"},
    }


@app.get("/exposure/priority")
def exposure_priority():
    with db() as conn:
        rows = {r["cell_id"]: dict(r) for r in conn.execute("SELECT * FROM risk_current").fetchall()}

    def nearest_cell(lat, lon):
        return min(CELLS, key=lambda c: (c["lat"] - lat) ** 2 + (c["lon"] - lon) ** 2)

    ranked = []
    for a in ASSETS:
        cell = nearest_cell(a["lat"], a["lon"])
        r = rows.get(cell["id"], {})
        risk = r.get("risk_score", 0) or 0
        ranked.append({"asset": a["name"], "type": a["type"], "level": r.get("level", "Green"),
                        "priority": round(risk * a["w"], 3)})
    ranked.sort(key=lambda x: -x["priority"])
    return {"ranked": ranked}


@app.get("/alerts")
def alerts():
    with db() as conn:
        rows = conn.execute("SELECT * FROM alerts WHERE active=1 ORDER BY entered_at DESC").fetchall()
    return {"active": [dict(r) for r in rows]}


class FeedbackIn(BaseModel):
    cell_id: int
    was_alert_useful: bool
    note: str = ""


@app.post("/feedback")
def feedback(body: FeedbackIn):
    with db() as conn:
        conn.execute("INSERT INTO feedback (cell_id, was_useful, note, ts) VALUES (?, ?, ?, ?)",
                     (body.cell_id, int(body.was_alert_useful), body.note, datetime.now(IST).isoformat()))
    return {"received": True, "status": 202}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
