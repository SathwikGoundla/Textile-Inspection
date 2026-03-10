"""
TextileVision AI - Pure OpenCV, No API needed
"""
import asyncio, base64, json, time, uuid
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from fastapi import Depends
from database.db import get_db, engine
from database import models as db_models

BASE_DIR      = Path(__file__).resolve().parent
FRONTEND_HTML = BASE_DIR.parent / "frontend" / "index.html"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
db_models.Base.metadata.create_all(bind=engine)

class CM:
    def __init__(self): self.ws = []
    async def connect(self, ws): await ws.accept(); self.ws.append(ws)
    def disconnect(self, ws):
        if ws in self.ws: self.ws.remove(ws)
mgr = CM()

# Global scan trigger — POST /api/scan sets this to True
_scan_trigger = False

@app.post("/api/scan")
async def trigger_scan():
    global _scan_trigger
    _scan_trigger = True
    return {"ok": True}

# ════════════════════════════════════════════════════════
#  TEXTILE DETECTION ENGINE
#  Works by analyzing texture, edges, shape, color
# ════════════════════════════════════════════════════════

def analyze_frame(frame: np.ndarray) -> dict:
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # ── DARK CHECK ───────────────────────────────────────
    if np.mean(gray) < 20:
        return mk("NO_OBJECT", False, None, 0, [], "NONE")

    # ── FEATURE EXTRACTION ───────────────────────────────

    # 1. Laplacian variance — fabric=medium texture (200-3000)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap_var = float(np.var(lap))

    # 2. Block uniformity — fabric blocks have similar variance
    bsz = 32
    block_vars = []
    for y in range(0, H-bsz, bsz):
        for x in range(0, W-bsz, bsz):
            block_vars.append(float(np.var(gray[y:y+bsz, x:x+bsz])))
    mean_bv = float(np.mean(block_vars)) if block_vars else 0
    std_bv  = float(np.std(block_vars))  if block_vars else 0
    # Low CoV = uniform = fabric
    cov = std_bv / (mean_bv + 1e-6)

    # 3. Edge map — fabric has many small soft edges
    edges = cv2.Canny(blur, 30, 100)
    edge_density = float(np.sum(edges > 0)) / (H * W)

    # 4. Color uniformity — fabric has consistent hue
    h_ch  = hsv[:,:,0].astype(float)
    s_ch  = hsv[:,:,1].astype(float)
    v_ch  = hsv[:,:,2].astype(float)
    h_std = float(np.std(h_ch))
    s_mean= float(np.mean(s_ch))
    v_std = float(np.std(v_ch))

    # 5. Strong circle detection — only hard circular objects (bottles)
    circ_blur = cv2.GaussianBlur(gray, (11,11), 2)
    circles = cv2.HoughCircles(circ_blur, cv2.HOUGH_GRADIENT, 1.2,
        minDist=min(H,W)//2, param1=100, param2=50,
        minRadius=int(min(H,W)*0.18), maxRadius=int(min(H,W)*0.5))
    strong_circle = False
    if circles is not None:
        for c in circles[0]:
            # Circle is strong only if ALSO metallic/glossy (high v_std)
            if c[2] > min(H,W)*0.2 and v_std > 60 and s_mean < 60:
                strong_circle = True

    # 6. Contour fill — fabric fills frame
    _, th = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
    fill = float(np.sum(th > 0)) / (H * W)

    # ── SCORING ──────────────────────────────────────────
    # TEXTILE EVIDENCE
    tex = 0.0
    # Medium texture (not too smooth, not too rough)
    if 100 < lap_var < 4000:    tex += 0.30
    elif 50 < lap_var < 6000:   tex += 0.15
    # Uniform blocks (consistent weave)
    if cov < 1.0:               tex += 0.25
    elif cov < 1.8:             tex += 0.10
    # Medium edge density (woven = some edges but not chaotic)
    if 0.04 < edge_density < 0.35: tex += 0.20
    # Fills frame (fabric held close)
    if fill > 0.65:             tex += 0.15
    # Consistent hue
    if h_std < 50:              tex += 0.10

    # OBJECT EVIDENCE (deductions)
    obj_score = 0.0
    if strong_circle:           obj_score += 0.6   # very strong bottle signal
    if v_std > 80 and s_mean < 40: obj_score += 0.3  # metallic
    if lap_var < 40:            obj_score += 0.2   # glass-smooth

    # ── DECISION ─────────────────────────────────────────
    if obj_score >= 0.55:
        # Clearly an object
        name = "Metal/Glass Object" if v_std > 80 else "Cylindrical Object"
        conf = round(min(0.95, 0.65 + obj_score * 0.3), 2)
        return mk("REJECTED", False, name, conf, [], "NONE")

    if tex >= 0.45:
        # Textile!
        name  = get_fabric_name(frame, hsv)
        conf  = round(min(0.97, 0.65 + tex * 0.32), 2)
        defs  = detect_defects(frame, gray, hsv, blur)
        wsum  = sum(d["sw"] for d in defs)
        sev   = "NONE" if not defs else ("HIGH" if wsum>=5 else "MEDIUM" if wsum>=3 else "LOW")
        status= "FAIL" if defs else "PASS"
        return mk(status, True, name, conf, defs, sev)

    if tex >= 0.25:
        # Borderline — lean textile for demo
        name = get_fabric_name(frame, hsv)
        conf = round(0.62 + tex * 0.2, 2)
        defs  = detect_defects(frame, gray, hsv, blur)
        wsum  = sum(d["sw"] for d in defs)
        sev   = "NONE" if not defs else "LOW"
        status= "FAIL" if defs else "PASS"
        return mk(status, True, name, conf, defs, sev)

    return mk("REJECTED", False, "Non-textile Object", 0.70, [], "NONE")


def get_fabric_name(frame, hsv) -> str:
    """Accurate color — samples center 40% of frame using median hue"""
    H, W = hsv.shape[:2]
    # Sample center region to avoid background
    cy1,cy2 = int(H*0.30), int(H*0.70)
    cx1,cx2 = int(W*0.30), int(W*0.70)
    roi = hsv[cy1:cy2, cx1:cx2]
    if roi.size == 0: roi = hsv

    h = float(np.median(roi[:,:,0]))
    s = float(np.median(roi[:,:,1]))
    v = float(np.median(roi[:,:,2]))

    # Achromatic first
    if s < 35:
        if v > 210: return "White Fabric"
        if v > 160: return "Off-White / Cream Fabric"
        if v > 110: return "Light Grey Fabric"
        if v > 55:  return "Dark Grey Fabric"
        return "Black Fabric"

    # Chromatic
    if h < 8 or h > 172:        return "Red Fabric"
    if 8   <= h < 18:           return "Maroon / Dark Red Fabric"
    if 18  <= h < 28:           return "Orange Fabric"
    if 28  <= h < 40:           return "Yellow Fabric"
    if 40  <= h < 52:           return "Yellow-Green Fabric"
    if 52  <= h < 85:
        return "Dark Green Fabric" if v < 80 else "Green Fabric"
    if 85  <= h < 105:          return "Teal / Cyan Fabric"
    if 105 <= h < 130:
        return "Blue Fabric" if s > 100 else "Light Blue Fabric"
    if 130 <= h < 148:
        return "Dark Blue / Denim Fabric" if v < 100 else "Blue / Denim Fabric"
    if 148 <= h < 162:          return "Purple / Violet Fabric"
    if 162 <= h < 172:          return "Pink / Magenta Fabric"
    return "Cotton Fabric"


def detect_defects(frame, gray, hsv, blur) -> list:
    H, W = gray.shape
    defects = []

    mean_b = float(np.mean(blur))
    std_b  = float(np.std(blur.astype(float)))

    # HOLE: very dark region compared to surroundings
    hole_th = mean_b - 2.8 * std_b
    if 8 < hole_th < mean_b * 0.5:
        _, dark = cv2.threshold(blur, int(hole_th), 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 700 < area < H*W*0.07:
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx_ = int(M["m10"]/M["m00"])
                cy_ = int(M["m01"]/M["m00"])
                r   = max(int(np.sqrt(area/np.pi)), 8)
                mi  = np.zeros_like(gray); mo = np.zeros_like(gray)
                cv2.circle(mi,(cx_,cy_),r,255,-1)
                cv2.circle(mo,(cx_,cy_),r+18,255,-1)
                ring = cv2.subtract(mo, mi)
                inv = float(np.mean(gray[mi>0]))   if mi.any()   else mean_b
                outv= float(np.mean(gray[ring>0])) if ring.any() else mean_b
                if outv > 0 and inv < outv * 0.62:
                    defects.append({"type":"HOLE","name":"Hole",
                        "desc":"Physical hole in fabric",
                        "conf":round(min(0.96,0.78+(1-inv/max(outv,1))*0.18),2),
                        "x":cx_/W,"y":cy_/H,"sw":3})

    # STAIN: foreign color on uniform fabric
    h_ch  = hsv[:,:,0].astype(float)
    s_ch  = hsv[:,:,1].astype(float)
    h_std = float(np.std(h_ch))
    if h_std < 16:
        h_mean = float(np.mean(h_ch))
        stain  = ((s_ch > 115) & (np.abs(h_ch-h_mean) > 38)).astype(np.uint8)*255
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, k2)
        cnts,_ = cv2.findContours(stain,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 1000 < area < H*W*0.12:
                M = cv2.moments(c)
                if M["m00"]!=0:
                    defects.append({"type":"STAIN","name":"Stain",
                        "desc":"Foreign substance on fabric",
                        "conf":0.80,"x":M["m10"]/M["m00"]/W,
                        "y":M["m01"]/M["m00"]/H,"sw":2})

    return defects[:2]


def mk(status, is_tex, name, conf, defs, sev):
    return {
        "overall_status": status,
        "is_textile": is_tex,
        "textile_type": name if is_tex else None,
        "object_label": name,
        "confidence": conf,
        "defects": [{"type":d["type"],"display_name":d["name"],
                     "description":d["desc"],"confidence":d["conf"],
                     "location":{"x":round(d["x"],3),"y":round(d["y"],3)},
                     "severity_weight":d["sw"]} for d in defs],
        "severity": sev
    }


# ════════════════════════════════════════════════════════
#  DRAW OVERLAY
# ════════════════════════════════════════════════════════
def draw(frame, det, fps):
    out = frame.copy()
    H, W = out.shape[:2]
    status = det["overall_status"]

    C = {"PASS":(30,210,30),"FAIL":(30,30,220),"REJECTED":(20,140,255),"NO_OBJECT":(60,60,60)}.get(status,(80,80,80))

    # Box + brackets
    x1,y1,x2,y2 = int(W*.04),int(H*.05),int(W*.96),int(H*.95)
    bW,bH = x2-x1, y2-y1

    # Soft tint
    ov = out.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),C,-1)
    cv2.addWeighted(ov,0.06,out,0.94,0,out)
    cv2.rectangle(out,(x1,y1),(x2,y2),C,3)

    cs,th=36,5
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(out,(px,py),(px+dx*cs,py),C,th)
        cv2.line(out,(px,py),(px,py+dy*cs),C,th)

    # Label tag
    label = det.get("textile_type") or det.get("object_label","")
    conf  = det.get("confidence",0)
    tag   = f"  {label}  {conf:.0%}  " if label else ""
    if tag:
        (tw,th2),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.60,2)
        ty = max(y1-4,th2+8)
        cv2.rectangle(out,(x1,ty-th2-8),(x1+tw,ty+2),C,-1)
        cv2.putText(out,tag,(x1,ty-2),cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),2)

    # Zoom inset top-right
    crop = frame[y1:y2, x1:x2]
    if crop.size > 100:
        iw,ih = 200,150
        inset = cv2.resize(crop,(iw,ih))
        cv2.rectangle(inset,(0,0),(iw-1,ih-1),C,2)
        cv2.putText(inset,"ZOOM",(5,16),cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1)
        px2=W-iw-8; py2=50
        if py2+ih<H and px2>0:
            out[py2:py2+ih,px2:px2+iw]=inset

    # Defect markers
    for d in det.get("defects",[]):
        if "location" in d:
            cx=x1+int(d["location"]["x"]*bW)
            cy=y1+int(d["location"]["y"]*bH)
            cv2.circle(out,(cx,cy),16,(0,30,220),-1)
            cv2.circle(out,(cx,cy),18,(255,255,255),2)
            cv2.putText(out,d["type"][:5],(cx+20,cy+5),cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,230,0),1)

    # Top bar
    cv2.rectangle(out,(0,0),(W,46),C,-1)
    bar = f"FPS:{fps:.0f}  {status}"
    if label: bar += f"  |  {label}  {conf:.0%}"
    cv2.putText(out,bar,(12,30),cv2.FONT_HERSHEY_SIMPLEX,0.70,(255,255,255),2)

    # Severity badge
    sev = det.get("severity","NONE")
    if sev not in ("NONE",""):
        sc={"LOW":(0,200,200),"MEDIUM":(0,130,255),"HIGH":(20,20,210)}.get(sev,(80,80,80))
        cv2.rectangle(out,(8,H-38),(155,H-8),sc,-1)
        cv2.putText(out,f"SEV: {sev}",(13,H-14),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),2)

    # 10-second countdown bar at bottom
    state = det.get("state","WAITING")
    remaining = det.get("lock_remaining", 0)
    if state == "LOCKED" and remaining > 0:
        bar_w = int(W * (remaining / 10.0))
        cv2.rectangle(out,(0,H-6),(W,H),(30,30,30),-1)
        cv2.rectangle(out,(0,H-6),(bar_w,H),C,-1)
        cv2.putText(out,f"RESULT LOCKED: {remaining:.1f}s — press SCAN for next",
                    (W//2-180,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
    elif state == "WAITING":
        cv2.rectangle(out,(0,H-6),(W,H),(20,20,20),-1)
        cv2.putText(out,"PRESS SCAN BUTTON TO ANALYZE",
                    (W//2-140,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,212,255),1)

    return out


# ════════════════════════════════════════════════════════
#  DEMO MODE (when no camera)
# ════════════════════════════════════════════════════════
_DEMO=[
    {"status":"PASS","name":"Cotton Fabric","tex":0},
    {"status":"PASS","name":"Denim","tex":1},
    {"status":"REJECTED","name":"Bottle","tex":-1},
    {"status":"FAIL","name":"White Cotton","tex":2},
    {"status":"PASS","name":"Green Fabric","tex":3},
]
def demo_frame(count,tex):
    H,W=480,640
    if tex==0:
        f=np.full((H,W,3),(210,200,185),dtype=np.uint8)
        for i in range(0,H,5): cv2.line(f,(0,i),(W,i),(190,178,162),1)
        for j in range(0,W,5): cv2.line(f,(j,0),(j,H),(190,178,162),1)
    elif tex==1:
        f=np.full((H,W,3),(65,105,155),dtype=np.uint8)
        for i in range(-H,W+H,4): cv2.line(f,(i,0),(i+H,H),(75,115,165),1)
    elif tex==2:
        f=np.full((H,W,3),(245,245,240),dtype=np.uint8)
        for i in range(0,H,6): cv2.line(f,(0,i),(W,i),(228,226,220),1)
        for j in range(0,W,6): cv2.line(f,(j,0),(j,H),(228,226,220),1)
        cv2.circle(f,(220,200),30,(15,15,20),-1)
    elif tex==3:
        f=np.full((H,W,3),(50,140,70),dtype=np.uint8)
        for i in range(0,H,5): cv2.line(f,(0,i),(W,i),(44,128,62),1)
        for j in range(0,W,5): cv2.line(f,(j,0),(j,H),(44,128,62),1)
    elif tex==-1:
        f=np.full((H,W,3),(90,90,100),dtype=np.uint8)
        cv2.ellipse(f,(W//2,H//2),(85,160),0,0,360,(165,165,175),-1)
        cv2.ellipse(f,(W//2,H//2),(70,145),0,0,360,(200,200,215),-1)
        cv2.ellipse(f,(W//2-22,H//2),(16,115),0,0,360,(228,228,238),-1)
    else:
        f=np.full((H,W,3),(150,130,110),dtype=np.uint8)
    noise=np.random.randint(-4,4,f.shape,dtype=np.int16)
    f=np.clip(f.astype(np.int16)+noise,0,255).astype(np.uint8)
    sy=int((count*3)%H)
    cv2.line(f,(0,sy),(W,sy),(0,255,150),1)
    return f


# ════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════
@app.get("/")
async def root():
    if FRONTEND_HTML.exists():
        return HTMLResponse(content=FRONTEND_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not found</h2>",404)

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.get("/api/stats")
async def stats(db:Session=Depends(get_db)):
    rows=db.query(db_models.InspectionLog).all(); total=len(rows)
    if total==0: return {"total_inspections":0,"passed":0,"failed":0,"pass_rate":0.0,"common_defects":{},"avg_confidence":0.0}
    passed=sum(1 for r in rows if r.status=="PASS")
    dc,tc={},0.0
    for r in rows:
        tc+=r.confidence
        for d in (json.loads(r.defects) if r.defects else []):
            dc[d.get("type","?")]=dc.get(d.get("type","?"),0)+1
    return {"total_inspections":total,"passed":passed,"failed":total-passed,
            "pass_rate":round(passed/total*100,2),"common_defects":dc,"avg_confidence":round(tc/total,3)}

@app.get("/api/logs")
async def logs(limit:int=50,db:Session=Depends(get_db)):
    rows=db.query(db_models.InspectionLog).order_by(db_models.InspectionLog.timestamp.desc()).limit(limit).all()
    return [{"id":r.inspection_id,"timestamp":r.timestamp.isoformat(),"status":r.status,
             "textile_type":r.textile_type,"confidence":r.confidence,
             "defects":json.loads(r.defects) if r.defects else [],"severity":r.severity} for r in rows]

@app.websocket("/ws/live")
async def ws_live(ws:WebSocket, db:Session=Depends(get_db)):
    await mgr.connect(ws)
    count=0; t0=time.time(); SKIP=3

    # Scan state: wait for SCAN button, then lock result for 10 seconds
    scan_requested = False   # frontend sends {"action":"scan"} to trigger
    locked_result  = None    # current locked result shown for 10s
    lock_until     = 0.0     # timestamp when lock expires

    # Try camera
    camera=None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        try:
            cap=cv2.VideoCapture(0,backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                for _ in range(5): cap.read()
                ret,fr=cap.read()
                if ret and fr is not None and fr.any():
                    camera=cap; break
            cap.release()
        except: pass

    use_demo = camera is None
    print("Camera:", "LIVE" if not use_demo else "DEMO MODE")

    try:
        while True:
            global _scan_trigger
            count+=1
            now = time.time()
            fps=round(count/max(now-t0,0.001),1)
            pt=now

            # Check global scan trigger (set by POST /api/scan)
            if _scan_trigger:
                scan_requested = True
                _scan_trigger = False

            # Get current frame
            if not use_demo:
                ret,frame=camera.read()
                if not ret or frame is None or not frame.any():
                    use_demo=True; frame=demo_frame(count,0)
            else:
                di=(count//180)%len(_DEMO)
                frame=demo_frame(count,_DEMO[di]["tex"])

            # ── SCAN LOGIC ──────────────────────────────────────
            # Case 1: Lock still active → show locked result with live frame
            if now < lock_until and locked_result is not None:
                remaining = round(lock_until - now, 1)
                result = {**locked_result,
                          "inspection_id": locked_result.get("inspection_id",""),
                          "timestamp": datetime.now().isoformat(),
                          "fps": fps,
                          "processing_time_ms": round((time.time()-pt)*1000,1),
                          "lock_remaining": remaining,
                          "state": "LOCKED"}

            # Case 2: Scan button pressed → analyze NOW
            elif scan_requested:
                scan_requested = False
                if use_demo:
                    import random
                    d=_DEMO[(count//180)%len(_DEMO)]
                    jit=(random.random()-0.5)*0.02
                    defs=[]
                    if d["status"]=="FAIL":
                        defs=[{"type":"HOLE","display_name":"Hole","description":"Physical hole",
                               "confidence":0.91,"location":{"x":0.42+jit,"y":0.45+jit},"severity_weight":3}]
                    wsum=sum(df["severity_weight"] for df in defs)
                    sev="NONE" if not defs else "HIGH" if wsum>=5 else "MEDIUM" if wsum>=3 else "LOW"
                    det={"overall_status":d["status"],"is_textile":d["status"] in ("PASS","FAIL"),
                         "textile_type":d["name"] if d["status"]!="REJECTED" else None,
                         "object_label":d["name"],"confidence":round(0.88+jit,2),
                         "defects":defs,"severity":sev}
                else:
                    det=analyze_frame(frame)
                    # Save to DB
                    if det.get("is_textile"):
                        log=db_models.InspectionLog(
                            inspection_id=str(uuid.uuid4())[:8],timestamp=datetime.now(),
                            status=det["overall_status"],
                            textile_type=det.get("textile_type",""),
                            confidence=det.get("confidence",0),
                            defects=json.dumps(det.get("defects",[])),
                            severity=det.get("severity","NONE"))
                        db.add(log); db.commit()

                det["inspection_id"] = str(uuid.uuid4())[:8]
                locked_result = det
                lock_until    = now + 10.0   # lock for 10 seconds
                result = {**det, "timestamp": datetime.now().isoformat(),
                          "fps": fps, "processing_time_ms": round((time.time()-pt)*1000,1),
                          "lock_remaining": 10.0, "state": "LOCKED"}

            # Case 3: Waiting for scan → show live feed, no result panel update
            else:
                result = {"overall_status": "WAITING", "is_textile": False,
                          "textile_type": None, "object_label": "Press SCAN to analyze",
                          "confidence": 0, "defects": [], "severity": "NONE",
                          "inspection_id": "", "timestamp": datetime.now().isoformat(),
                          "fps": fps, "processing_time_ms": 0,
                          "lock_remaining": 0, "state": "WAITING"}

            annotated=draw(frame, result, fps)
            _,buf=cv2.imencode(".jpg",annotated,[cv2.IMWRITE_JPEG_QUALITY,80])
            result["frame"]="data:image/jpeg;base64,"+base64.b64encode(buf).decode()
            await ws.send_json(result)
            await asyncio.sleep(0.020)

    except WebSocketDisconnect: mgr.disconnect(ws)
    except Exception as e: print(f"WS error:{e}"); mgr.disconnect(ws)
    finally:
        if camera: camera.release()
