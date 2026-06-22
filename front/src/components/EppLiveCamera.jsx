import { useCallback, useEffect, useRef, useState } from "react";
import EppZoneEditor from "./EppZoneEditor";
import { getZoneConfig, saveZoneConfig } from "../services/apiDetectionService";
import { openEppSocket } from "../services/eppSocketService";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const IP_CAMERA_URL = "http://192.168.18.6:8081/";
const SAMPLE_INTERVAL_MS = 1000;
const COOLDOWN_SECS = 10;

export default function EppLiveCamera({ active = true, onEppCameraDetection }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ipImgRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const wsRef = useRef(null);
  const handleResultRef = useRef(null);
  const configRef = useRef({});
  const isProcessingRef = useRef(false);
  const isCooldownRef = useRef(false);
  const isActiveRef = useRef(false);
  const isPausedRef = useRef(false);
  const lastFrameUrlRef = useRef(null);
  const lastSavedTsRef = useRef(0);
  const lastAlertTsRef = useRef(0);
  const SAVE_INTERVAL_MS = 8000;
  const ALERT_STICKY_MS = 5000; // maintain alert state for 5s after last detection

  const [cameraSource, setCameraSourceState] = useState("webcam"); // "webcam" | "ip"
  const cameraSourceRef = useRef("webcam");
  const setCameraSource = (src) => { cameraSourceRef.current = src; setCameraSourceState(src); };

  const [isActive, setIsActive] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isEditingZones, setIsEditingZones] = useState(false);
  const [alertVisible, setAlertVisible] = useState(false);
  const [cooldownRemaining, setCooldownRemaining] = useState(0);
  const [lastResult, setLastResult] = useState(null);
  const [annotatedFrame, setAnnotatedFrame] = useState(null);
  const [frozenFrame, setFrozenFrame] = useState(null);
  const [zones, setZones] = useState([]);
  const [defaultZoneEpp, setDefaultZoneEpp] = useState([]);
  const [defaultZoneActive, setDefaultZoneActive] = useState(true);
  const [defaultZoneRequirePerson, setDefaultZoneRequirePerson] = useState(false);
  const [eppClasses, setEppClasses] = useState([]);
  const [error, setError] = useState("");
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [livePersonCount, setLivePersonCount] = useState(0);
  const sessionInfractionsRef = useRef({});
  const [sessionInfractions, setSessionInfractions] = useState({});
  const complianceBucketsRef = useRef([]);
  const [complianceBuckets, setComplianceBuckets] = useState([]);
  const lastInfractionWindowRef = useRef(null);

  const FALLBACK_CLASSES = ["helmet", "vest", "glasses", "hardhat", "gloves", "mask", "boots"];

  // Fetch EPP classes — on camera start OR zone editor open; fallback to common names
  useEffect(() => {
    if (eppClasses.length > 0) return;
    if (!isActive && !isEditingZones) return;
    fetch(`${API_BASE_URL}/api/epp/classes`)
      .then((r) => {
        if (!r.ok) throw new Error("backend error");
        return r.json();
      })
      .then((data) => {
        const compliant = Array.isArray(data.compliant) && data.compliant.length > 0
          ? data.compliant
          : null;
        const all = Array.isArray(data.all) && data.all.length > 0
          ? data.all
          : null;
        const cls = compliant || all;
        setEppClasses(cls && cls.length > 0 ? cls : FALLBACK_CLASSES);
      })
      .catch(() => setEppClasses(FALLBACK_CLASSES));
  }, [isActive, isEditingZones, eppClasses.length]);

  // Load persisted zone config on mount
  useEffect(() => {
    getZoneConfig()
      .then((cfg) => {
        if (cfg.zones?.length > 0) setZones(cfg.zones);
        if (cfg.defaultZoneEpp?.length > 0) setDefaultZoneEpp(cfg.defaultZoneEpp);
        setDefaultZoneActive(cfg.defaultZoneActive ?? true);
        setDefaultZoneRequirePerson(cfg.defaultZoneRequirePerson ?? false);
      })
      .catch(() => {});
  }, []);

  // Accumulate session stats — only on real detection events, not every frame
  useEffect(() => {
    if (!lastResult) return;

    // Person count: always update (cheap, just a number)
    setLivePersonCount(lastResult.personCount ?? lastResult.totalPersons ?? 0);

    // Compliance + infractions: only when model processed a meaningful frame
    const hasActivity =
      (lastResult.detections?.length ?? 0) > 0 ||
      lastResult.alert === true ||
      (lastResult.zoneResults?.length ?? 0) > 0 ||
      lastResult.defaultZoneResult != null;
    if (!hasActivity) return;

    // Compliance buckets — one entry per cooldown window (10s), immutable once written
    const windowMs = COOLDOWN_SECS * 1000;
    const windowTs = Math.floor(Date.now() / windowMs) * windowMs;
    const buckets = complianceBucketsRef.current;
    const last = buckets[buckets.length - 1];
    if (last?.ts !== windowTs) {
      const compliantEpp = (lastResult.detections || []).filter((d) => d.isCompliant).length;
      const totalEpp = (lastResult.detections || []).length;
      const compliantPct = totalEpp > 0
        ? Math.round((compliantEpp / totalEpp) * 100)
        : lastResult.alert ? 0 : 100;
      const newBuckets = [...buckets, { ts: windowTs, compliantPct }];
      if (newBuckets.length > 8) newBuckets.shift();
      complianceBucketsRef.current = newBuckets;
      setComplianceBuckets(newBuckets);
    }

    // Infractions by zone+type — one event per cooldown window (not per frame)
    if (!lastResult.alert) return;
    if (lastInfractionWindowRef.current === windowTs) return;
    lastInfractionWindowRef.current = windowTs;
    const infrNew = { ...sessionInfractionsRef.current };
    const addViolations = (zoneLabel, epp) => {
      if (!epp?.length) return;
      if (!infrNew[zoneLabel]) infrNew[zoneLabel] = {};
      epp.forEach((e) => { infrNew[zoneLabel][e] = (infrNew[zoneLabel][e] || 0) + 1; });
    };
    (lastResult.zoneResults || []).forEach((zr) => {
      if (!zr.compliant) {
        const label = zr.label || zr.zoneId || "Zona";
        addViolations(label, [...(zr.violatingEpp || []), ...(zr.missingEpp || [])]);
      }
    });
    if (lastResult.defaultZoneResult && !lastResult.defaultZoneResult.compliant) {
      addViolations("Por defecto", [
        ...(lastResult.defaultZoneResult.violatingEpp || []),
        ...(lastResult.defaultZoneResult.missingEpp || []),
      ]);
    }
    sessionInfractionsRef.current = infrNew;
    setSessionInfractions({ ...infrNew });
  }, [lastResult]);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    isProcessingRef.current = false;
    isCooldownRef.current = false;
    isActiveRef.current = false;
    isPausedRef.current = false;
    lastSavedTsRef.current = 0;
    lastAlertTsRef.current = 0;
    setIsActive(false);
    setIsPaused(false);
    setIsEditingZones(false);
    setAlertVisible(false);
    setCooldownRemaining(0);
    setLastResult(null);
    setAnnotatedFrame(null);
    setFrozenFrame(null);
    setLivePersonCount(0);
    setSessionInfractions({});
    sessionInfractionsRef.current = {};
    setComplianceBuckets([]);
    complianceBucketsRef.current = [];
    lastInfractionWindowRef.current = null;
  }, []);

  const triggerCooldown = useCallback(() => {
    isCooldownRef.current = true;
    setAlertVisible(true);
    let secs = COOLDOWN_SECS;
    setCooldownRemaining(secs);
    const timer = setInterval(() => {
      secs -= 1;
      setCooldownRemaining(secs);
      if (secs <= 0) {
        clearInterval(timer);
        isCooldownRef.current = false;
        setAlertVisible(false);
        setCooldownRemaining(0);
      }
    }, 1000);
  }, []);

  // Handle one detection payload pushed by the server over the WebSocket.
  const handleResult = useCallback((data) => {
    if (isPausedRef.current) return; // keep the frozen frame while editing zones
    if (data?.error) {
      setError(typeof data.error === "string" ? data.error.slice(0, 200) : "Error del servidor");
      return;
    }
    const isIpMode = cameraSourceRef.current === "ip";
    setError("");
    setLastResult(data);

    if (data.annotated_frame) {
      setAnnotatedFrame(data.annotated_frame);
      if (isIpMode) lastFrameUrlRef.current = data.annotated_frame;
    }

    const now = Date.now();
    if (data.alert) lastAlertTsRef.current = now;
    const stickyAlert = data.alert || (now - lastAlertTsRef.current < ALERT_STICKY_MS);

    if (stickyAlert) {
      if (now - lastSavedTsRef.current >= SAVE_INTERVAL_MS) {
        lastSavedTsRef.current = now;
        onEppCameraDetection?.({
          frameDataUrl: data.annotated_frame || lastFrameUrlRef.current,
          width: isIpMode ? (data.imageSize?.width || 640) : canvasRef.current?.width,
          height: isIpMode ? (data.imageSize?.height || 480) : canvasRef.current?.height,
          result: data,
          timestamp: new Date().toISOString(),
          zonesConfig: zones,
          defaultZoneEpp: defaultZoneEpp,
        });
      }
      if (!isCooldownRef.current) {
        triggerCooldown();
      }
    }
  }, [zones, defaultZoneEpp, triggerCooldown, onEppCameraDetection]);

  // Build the config message sent to the server (once on open, then on changes).
  const buildConfigMsg = useCallback(() => ({
    type: "config",
    mode: cameraSourceRef.current === "ip" ? "ip" : "webcam",
    camera_url: cameraSourceRef.current === "ip" ? IP_CAMERA_URL : undefined,
    zones: zones.map((z) => ({
      id: z.id,
      label: z.label,
      bbox: z.bbox,
      required_epp: z.requiredEpp,
      active: z.active !== false,
      require_person: z.requirePerson === true,
    })),
    defaultZoneEpp,
    defaultZoneActive,
    defaultZoneRequirePerson,
  }), [zones, defaultZoneEpp, defaultZoneActive, defaultZoneRequirePerson]);

  // Webcam only: grab a frame from the canvas and push the JPEG bytes over the socket.
  const sendFrame = useCallback(async () => {
    if (isPausedRef.current) return;
    if (cameraSourceRef.current === "ip") return; // IP frames are server-driven
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    const vidW = video.videoWidth || 640;
    const vidH = video.videoHeight || 480;
    const scale = vidW < 640 ? 640 / vidW : 1;
    canvas.width = Math.round(vidW * scale);
    canvas.height = Math.round(vidH * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    lastFrameUrlRef.current = canvas.toDataURL("image/jpeg", 0.92);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) return;
    const buffer = await blob.arrayBuffer();
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(buffer);
  }, []);

  // Keep refs fresh so the long-lived socket callbacks always use current handlers.
  useEffect(() => { handleResultRef.current = handleResult; }, [handleResult]);

  // Push updated config to the server whenever zones/defaults change mid-session.
  useEffect(() => {
    const cfg = buildConfigMsg();
    configRef.current = cfg;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify(cfg)); } catch { /* ignore */ }
    }
  }, [buildConfigMsg]);

  const startCamera = useCallback(async () => {
    setError("");
    if (cameraSourceRef.current === "ip") {
      setIsActive(true);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsActive(true);
      // Frame capture interval is started by the WebSocket effect on ws.onopen.
    } catch (err) {
      setError("No se pudo acceder a la cámara: " + (err?.message || "permiso denegado"));
    }
  }, []);

  const updateZone = (id, updates) => {
    setZones((prev) => prev.map((z) => (z.id === id ? { ...z, ...updates } : z)));
  };

  const toggleZoneEpp = (id, cls) => {
    setZones((prev) =>
      prev.map((z) => {
        if (z.id !== id) return z;
        const epp = z.requiredEpp.includes(cls)
          ? z.requiredEpp.filter((e) => e !== cls)
          : [...z.requiredEpp, cls];
        return { ...z, requiredEpp: epp };
      })
    );
  };

  const closeZonePanel = () => {
    setSelectedZoneId(null);
    saveZoneConfig({ zones, defaultZoneEpp, defaultZoneActive, defaultZoneRequirePerson }).catch(() => {});
  };

  const handlePause = () => {
    isPausedRef.current = true;
    setIsPaused(true);
    setFrozenFrame(lastFrameUrlRef.current);
    setIsEditingZones(true);
    // Stop the server-driven IP push while editing zones (webcam stops via the guard).
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: "stop" })); } catch { /* ignore */ }
    }
  };

  const handleResume = () => {
    isPausedRef.current = false;
    setIsPaused(false);
    setIsEditingZones(false);
    isCooldownRef.current = false;
    // Re-send config to restart the IP push loop with the latest zones.
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify(buildConfigMsg())); } catch { /* ignore */ }
    }
    saveZoneConfig({
      zones,
      defaultZoneEpp,
      defaultZoneActive,
      defaultZoneRequirePerson,
    }).catch(() => {});
  };

  // IP camera: swap src imperatively so browser doesn't flash on each new frame
  useEffect(() => {
    if (cameraSource !== "ip" || !annotatedFrame || !ipImgRef.current) return;
    const tmp = new Image();
    tmp.onload = () => { if (ipImgRef.current) ipImgRef.current.src = annotatedFrame; };
    tmp.src = annotatedFrame;
  }, [annotatedFrame, cameraSource]);

  useEffect(() => {
    if (!active && isActive) stopCamera();
  }, [active, isActive, stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Open the persistent WebSocket while the camera is active.
  // Webcam: push frames on an interval. IP: server pushes frames to us.
  useEffect(() => {
    if (!isActive) return;
    isActiveRef.current = true;

    const ws = openEppSocket({
      onMessage: (data) => handleResultRef.current?.(data),
      onError: () => setError("Error de conexión con el servidor (WebSocket)"),
    });
    wsRef.current = ws;

    let interval = null;
    ws.onopen = () => {
      try { ws.send(JSON.stringify(configRef.current)); } catch { /* ignore */ }
      if (cameraSourceRef.current !== "ip") {
        interval = setInterval(sendFrame, SAMPLE_INTERVAL_MS);
        intervalRef.current = interval;
      }
    };

    return () => {
      if (interval) clearInterval(interval);
      if (intervalRef.current === interval) intervalRef.current = null;
      try { ws.close(); } catch { /* ignore */ }
      if (wsRef.current === ws) wsRef.current = null;
    };
  }, [isActive, cameraSource, sendFrame]);

  const detections = lastResult?.detections || [];
  const zoneResults = lastResult?.zoneResults || [];
  const defaultZoneResult = lastResult?.defaultZoneResult ?? null;
  const hasAlert = lastResult?.alert;
  const hasDetections = detections.length > 0;

  let statusBadge = null;
  if (lastResult && !isPaused) {
    if (hasAlert) {
      statusBadge = { text: "EPP incompleto", cls: "border-warn-500/40 bg-warn-500/20 text-warn-100" };
    } else if (hasDetections) {
      statusBadge = { text: "EPP correcto", cls: "border-ok-500/40 bg-ok-500/20 text-ok-100" };
    } else {
      statusBadge = { text: "Sin detecciones", cls: "border-[#2A2A2E] bg-steel-950/80 text-[#C0C7D4]" };
    }
  }

  return (
    <section id="epp-camera-section" className="relative flex flex-col h-full overflow-hidden rounded-[2rem] border border-[#2A2A2E] bg-[#161618] p-5 shadow-glow">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent-500/60 to-transparent" />
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-[#FB923C]/75">Cámara</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Detección EPP en tiempo real</h2>
          <p className="mt-2 text-sm leading-6 text-[#C0C7D4]">
            El modelo detecta EPP directamente. Define zonas con requisitos distintos.
          </p>
          {isActive && !isPaused && (
            <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-ok-500/20 bg-ok-500/10 px-3 py-1">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ok-400" />
              <span className="text-xs text-ok-300">
                {livePersonCount > 0
                  ? `Monitoreando ${livePersonCount} persona${livePersonCount !== 1 ? "s" : ""} en tiempo real`
                  : "Monitoreando en tiempo real"}
              </span>
            </div>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          {!isActive ? (
            <div className="flex flex-col items-end gap-2">
              {/* Camera source selector */}
              <div className="flex items-center gap-1 rounded-2xl border border-[#2E2E33] bg-[#1C1C1F] p-1">
                <button
                  type="button"
                  onClick={() => setCameraSource("webcam")}
                  className={`rounded-xl px-4 py-2 text-xs font-semibold transition ${
                    cameraSource === "webcam"
                      ? "bg-[#F97316]/30 text-white ring-1 ring-[#F97316]/40"
                      : "text-steel-400 hover:text-white"
                  }`}
                >
                  Cámara web
                </button>
                <button
                  type="button"
                  onClick={() => setCameraSource("ip")}
                  className={`rounded-xl px-4 py-2 text-xs font-semibold transition ${
                    cameraSource === "ip"
                      ? "bg-[#F97316]/30 text-white ring-1 ring-[#F97316]/40"
                      : "text-steel-400 hover:text-white"
                  }`}
                >
                  Cámara IP
                </button>
              </div>
              {cameraSource === "ip" && (
                <span className="text-[10px] text-steel-500">{IP_CAMERA_URL}</span>
              )}
              <div className="flex flex-row gap-2">
                <button
                  type="button"
                  onClick={startCamera}
                  className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
                >
                  {cameraSource === "ip" ? "Conectar cámara IP" : "Activar cámara EPP"}
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditingZones((v) => !v)}
                  className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
                >
                  {isEditingZones ? "Cerrar zonas" : "Configurar zonas"}
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-end gap-2">
              <span className="text-[10px] text-steel-500">
                {cameraSource === "ip" ? `Cámara IP · ${IP_CAMERA_URL}` : "Cámara web"}
              </span>
              <div className="flex flex-row gap-2">
                {!isPaused ? (
                  <button
                    type="button"
                    onClick={handlePause}
                    className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
                  >
                    Pausar · Editar zonas
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleResume}
                    className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
                  >
                    Reanudar detecciones
                  </button>
                )}
                <button
                  type="button"
                  onClick={stopCamera}
                  className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
                >
                  Detener cámara
                </button>
              </div>
            </div>
          )}

          {(zones.length > 0 || defaultZoneEpp.length > 0) && isActive && (
            <span className="text-xs text-ok-300">
              {zones.length > 0 && `${zones.length} zona${zones.length > 1 ? "s" : ""}`}
              {zones.length > 0 && defaultZoneEpp.length > 0 && " · "}
              {defaultZoneEpp.length > 0 && "zona por defecto activa"}
            </span>
          )}
          {zones.length === 0 && defaultZoneEpp.length === 0 && isActive && (
            <span className="text-xs text-steel-500">
              Sin zonas configuradas · alerta cuando falta casco o chaleco
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-2xl border border-warn-500/30 bg-warn-500/10 px-4 py-3 text-sm text-warn-200">
          {error}
        </div>
      )}

      {isPaused && (
        <div className="mt-4 rounded-2xl border border-accent-400/30 bg-accent-500/10 px-4 py-3 text-sm text-[#FDBA74]">
          Detecciones en pausa, la cámara sigue activa - dibujar zonas y reanudar cuando se terminen los ajustes
        </div>
      )}

      {alertVisible && !isPaused && (
        <div className="mt-4 animate-pulse rounded-2xl border border-warn-500/50 bg-warn-500/20 px-4 py-4 text-center">
          <p className="text-lg font-bold text-warn-100">EPP incompleto detectado</p>
          <p className="mt-1 text-sm text-warn-300">Próxima detección en {cooldownRemaining}s</p>
        </div>
      )}

      {/* Zone results badges */}
      {(zoneResults.length > 0 || defaultZoneResult) && !isPaused && (
        <div className="mt-4 flex flex-wrap gap-2">
          {zoneResults.map((zr) => {
            const inactive = zr.active === false;
            const violations = zr.violatingEpp || [];
            const missing = zr.missingEpp || [];
            const nonCompliantParts = [
              violations.length > 0 ? `sin: ${violations.join(", ")}` : null,
              missing.length > 0 ? `falta: ${missing.join(", ")}` : null,
            ].filter(Boolean);
            return (
              <button
                key={zr.zoneId}
                type="button"
                onClick={() => setSelectedZoneId(zr.zoneId)}
                className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition hover:brightness-125 cursor-pointer ${
                  inactive
                    ? "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500"
                    : !zr.hasRequired
                      ? "border-[#2A2A2E] bg-[#1C1C1F] text-steel-400"
                      : zr.compliant
                        ? "border-ok-500/30 bg-ok-500/15 text-ok-200"
                        : "border-warn-500/30 bg-warn-500/15 text-warn-200"
                }`}
              >
                {zr.label || zr.zoneId}
                {inactive && <span className="ml-1 opacity-60">· inactiva</span>}
                {!inactive && zr.hasRequired && (
                  <span className="ml-1">
                    {zr.compliant ? "" : `${nonCompliantParts.join(" · ")}`}
                  </span>
                )}
              </button>
            );
          })}
          {/* Default zone badge — always shown when named zones exist so config is accessible */}
          {(() => {
            const violations = defaultZoneResult?.violatingEpp || [];
            const missing = defaultZoneResult?.missingEpp || [];
            const nonCompliantParts = [
              violations.length > 0 ? `sin: ${violations.join(", ")}` : null,
              missing.length > 0 ? `falta: ${missing.join(", ")}` : null,
            ].filter(Boolean);
            const inactive = defaultZoneResult?.active === false;
            const hasResult = defaultZoneResult !== null;
            return (
              <button
                type="button"
                onClick={() => setSelectedZoneId("default")}
                className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition hover:brightness-125 cursor-pointer ${
                  inactive
                    ? "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500"
                    : !hasResult || !defaultZoneResult.hasRequired
                      ? "border-[#2A2A2E] bg-[#1C1C1F] text-steel-400"
                      : defaultZoneResult.compliant
                        ? "border-ok-500/30 bg-ok-500/15 text-ok-200"
                        : "border-warn-500/30 bg-warn-500/15 text-warn-200"
                }`}
              >
                Por defecto
                {inactive && <span className="ml-1 opacity-60">· inactiva</span>}
                {!inactive && hasResult && defaultZoneResult.hasRequired && (
                  <span className="ml-1">
                    {defaultZoneResult.compliant ? "" : nonCompliantParts.join(" · ")}
                  </span>
                )}
              </button>
            );
          })()}
        </div>
      )}

      {/* Camera + side panel */}
      <div className={`mt-5 grid gap-4 ${isEditingZones ? "" : "xl:grid-cols-[minmax(0,1fr)_240px]"}`}>
        {/* Video */}
        <div className="relative overflow-hidden rounded-[1.75rem] border border-[#2A2A2E] bg-steel-950/80">
          <video
            ref={videoRef}
            className={`block h-auto w-full ${isActive && cameraSource === "webcam" ? "" : "hidden"}`}
            style={{ maxHeight: "420px", objectFit: "cover" }}
            playsInline
            muted
          />
          {isActive && cameraSource === "ip" && (
            <>
              <img
                ref={ipImgRef}
                alt="Cámara IP · detección en vivo"
                className={`block h-auto w-full ${annotatedFrame ? "" : "hidden"}`}
                style={{ maxHeight: "420px", objectFit: "cover" }}
              />
              {!annotatedFrame && (
                <div className="flex min-h-[360px] items-center justify-center">
                  <span className="animate-pulse text-sm text-steel-400">Conectando a cámara IP…</span>
                </div>
              )}
            </>
          )}
          <canvas ref={canvasRef} className="hidden" />

          {statusBadge && isActive && (
            <div className="absolute left-4 top-4">
              <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${statusBadge.cls}`}>
                {statusBadge.text}
              </span>
            </div>
          )}

          {isPaused && isActive && (
            <div className="absolute left-4 top-4">
              <span className="rounded-full border border-accent-400/40 bg-steel-950/90 px-3 py-1 text-xs font-semibold text-[#FDBA74]">
                En pausa
              </span>
            </div>
          )}

          {!isActive && (
            <div className="flex min-h-[360px] flex-col items-center justify-center px-6 text-center">
              <div className="flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-[#F97316]/12 text-4xl text-[#FDBA74]">
                ◎
              </div>
              <h3 className="mt-5 text-xl font-semibold text-white">
                {cameraSource === "ip" ? "Cámara IP inactiva" : "Cámara EPP inactiva"}
              </h3>
              <p className="mt-2 max-w-lg text-sm leading-6 text-steel-400">
                {cameraSource === "ip"
                  ? `Click en "Conectar cámara IP" para iniciar stream desde ${IP_CAMERA_URL}`
                  : `Click en "Activar cámara EPP" para iniciar la detección en tiempo real.`}
              </p>
            </div>
          )}
        </div>

        {/* Side panel: last annotated frame + detections (only when not editing zones) */}
        {!isEditingZones && (
          <div className="flex flex-col gap-4">
            {annotatedFrame && cameraSource !== "ip" && (
              <div className="rounded-[1.75rem] border border-[#2A2A2E] bg-steel-900/70 overflow-hidden">
                <p className="px-4 pt-3 text-xs uppercase tracking-[0.25em] text-steel-400">Último frame</p>
                <img src={annotatedFrame} alt="Frame anotado" className="mt-2 w-full object-contain" />
              </div>
            )}

            {detections.length > 0 && (
              <div className="rounded-[1.75rem] border border-[#2A2A2E] bg-steel-900/70 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-steel-400">Detecciones</p>
                <div className="mt-2 space-y-2">
                  {detections.map((det) => (
                    <div key={det.id} className="flex items-center justify-between text-xs">
                      <span className={`rounded-full border px-2 py-0.5 ${
                        det.isCompliant
                          ? "border-ok-500/20 bg-ok-500/10 text-ok-200"
                          : "border-warn-500/20 bg-warn-500/10 text-warn-200"
                      }`}>
                        {det.label}
                      </span>
                      <span className="text-steel-400">{(det.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!annotatedFrame && !detections.length && isActive && !isPaused && (
              <div className="rounded-[1.75rem] border border-[#2A2A2E] bg-steel-900/70 p-4 text-sm text-steel-400">
                Esperando detecciones EPP…
              </div>
            )}
          </div>
        )}
      </div>

      {/* Session stats: compliance history + infractions by zone */}
      {!isEditingZones && (
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {/* Compliance per-minute bars */}
          <div className="rounded-[1.75rem] border border-[#2A2A2E] bg-[#0F0F11]/60 p-4">
            <p className="text-[10px] uppercase tracking-[0.25em] text-steel-400">Cumplimiento EPP · últimas detecciones</p>
            {complianceBuckets.length > 0 ? (
              <div className="mt-3 space-y-2.5">
                {complianceBuckets.map((bucket, i) => {
                  const nonCompliantPct = 100 - (bucket.compliantPct ?? 100);
                  const isOk = nonCompliantPct === 0;
                  return (
                    <div key={i} className="flex items-center gap-3 text-xs">
                      <span className="w-14 shrink-0 text-right tabular-nums text-steel-500">
                        {new Date(bucket.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      </span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#2A2A2E]">
                        <div
                          className={`h-full rounded-full ${isOk ? "bg-ok-500" : "bg-warn-500"}`}
                          style={{ width: isOk ? "100%" : `${Math.max(nonCompliantPct, 8)}%` }}
                        />
                      </div>
                      <span className={`w-16 shrink-0 text-right font-semibold tabular-nums ${isOk ? "text-ok-300" : "text-warn-300"}`}>
                        {!isOk && nonCompliantPct < 100 ? `${nonCompliantPct}%` : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="mt-3 text-xs text-steel-500">Acumulando datos…</p>
            )}
          </div>

          {/* Infractions by zone + EPP type */}
          <div className="rounded-[1.75rem] border border-[#2A2A2E] bg-[#0F0F11]/60 p-4">
            <p className="text-[10px] uppercase tracking-[0.25em] text-steel-400">Infracciones · sesión actual</p>
            {Object.keys(sessionInfractions).length > 0 ? (
              <div className="mt-3 space-y-4">
                {Object.entries(sessionInfractions).map(([zoneLabel, eppCounts]) => (
                  <div key={zoneLabel}>
                    <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-steel-500">{zoneLabel}</p>
                    <div className="space-y-1">
                      {Object.entries(eppCounts)
                        .sort(([, a], [, b]) => b - a)
                        .map(([epp, count]) => (
                          <div key={epp} className="flex items-center justify-between text-xs">
                            <span className="capitalize text-[#C0C7D4]">Sin {epp}</span>
                            <span className="font-semibold tabular-nums text-warn-300">{count}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-xs text-ok-300">Sin infracciones registradas</p>
            )}
          </div>
        </div>
      )}

      {/* Zone editor — shown when paused or camera off */}
      {isEditingZones && (
        <div className="mt-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-white">Editor de zonas</p>
              <p className="text-xs text-steel-400">
                {isActive
                  ? "Dibuja rectángulos sobre el frame y asigna EPP requerido por zona"
                  : "Edita zonas existentes · activa la cámara para dibujar nuevas"}
              </p>
            </div>
            {isActive ? (
              <button
                type="button"
                onClick={handleResume}
                className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
              >
                Reanudar con estas zonas
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  setIsEditingZones(false);
                  saveZoneConfig({ zones, defaultZoneEpp, defaultZoneActive, defaultZoneRequirePerson }).catch(() => {});
                }}
                className="inline-flex items-center justify-center rounded-2xl bg-[#F97316]/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#F97316]/30 ring-1 ring-[#F97316]/40"
              >
                Guardar y cerrar
              </button>
            )}
          </div>

          <EppZoneEditor
            frameUrl={frozenFrame}
            zones={zones}
            onZonesChange={setZones}
            eppClasses={eppClasses}
            defaultZoneEpp={defaultZoneEpp}
            onDefaultZoneEppChange={setDefaultZoneEpp}
            defaultZoneActive={defaultZoneActive}
            onDefaultZoneActiveChange={setDefaultZoneActive}
            defaultZoneRequirePerson={defaultZoneRequirePerson}
            onDefaultZoneRequirePersonChange={setDefaultZoneRequirePerson}
          />
        </div>
      )}

      {/* Zone config panel — no pause needed */}
      {selectedZoneId !== null && (() => {
        const isDefault = selectedZoneId === "default";
        const zone = isDefault ? null : zones.find((z) => z.id === selectedZoneId);
        if (!isDefault && !zone) return null;

        const ZONE_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"];
        const zoneIdx = zones.findIndex((z) => z.id === selectedZoneId);
        const color = !isDefault && zoneIdx >= 0 ? ZONE_COLORS[zoneIdx % ZONE_COLORS.length] : "#94a3b8";

        const currentEpp = isDefault ? defaultZoneEpp : zone.requiredEpp;
        const currentActive = isDefault ? defaultZoneActive !== false : zone.active !== false;
        const currentRequirePerson = isDefault ? defaultZoneRequirePerson : zone.requirePerson;

        return (
          <div
            className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 bg-black/60"
            onClick={closeZonePanel}
          >
            <div
              className="w-full max-w-sm rounded-[2rem] border border-[#2E2E33] bg-steel-900 p-5 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: color }} />
                  {isDefault ? (
                    <div>
                      <p className="text-sm font-semibold text-white">Zona por defecto</p>
                      <p className="text-[10px] text-steel-500">Resto de pantalla sin zona definida</p>
                    </div>
                  ) : (
                    <input
                      type="text"
                      value={zone.label}
                      onChange={(e) => updateZone(zone.id, { label: e.target.value })}
                      className="bg-transparent text-sm font-semibold text-white outline-none border-b border-[#2E2E33] focus:border-[#F97316]/50 transition w-40"
                      placeholder="Nombre de zona"
                    />
                  )}
                </div>
                <button
                  type="button"
                  onClick={closeZonePanel}
                  className="rounded-xl border border-[#2A2A2E] bg-[#1C1C1F] px-2.5 py-1 text-xs text-steel-400 hover:text-white transition shrink-0"
                >
                  ✕ Cerrar
                </button>
              </div>

              <div className="flex flex-wrap gap-2 mb-4">
                <button
                  type="button"
                  onClick={() =>
                    isDefault
                      ? setDefaultZoneActive(!currentActive)
                      : updateZone(zone.id, { active: !currentActive })
                  }
                  className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition ${
                    currentActive
                      ? "border-ok-500/30 bg-ok-500/10 text-ok-300 hover:bg-ok-500/20"
                      : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500 hover:border-[#444448] hover:text-white"
                  }`}
                >
                  {currentActive ? "Activa" : "Inactiva"}
                </button>
                <button
                  type="button"
                  onClick={() =>
                    isDefault
                      ? setDefaultZoneRequirePerson(!currentRequirePerson)
                      : updateZone(zone.id, { requirePerson: !currentRequirePerson })
                  }
                  className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition ${
                    currentRequirePerson
                      ? "border-ok-500/30 bg-ok-500/10 text-ok-300 hover:bg-ok-500/20"
                      : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500 hover:border-[#444448] hover:text-white"
                  }`}
                >
                  {currentRequirePerson ? "Persona: Sí" : "Persona: No"}
                </button>
              </div>

              <div>
                <p className="text-[10px] uppercase tracking-[0.25em] text-steel-500 mb-1">EPP requerido</p>
                {eppClasses.length === 0 ? (
                  <p className="mt-2 text-xs text-steel-500 italic">Cargando clases…</p>
                ) : (
                  <>
                    <div className="flex flex-wrap gap-1.5 mt-1.5">
                      {eppClasses.map((cls) => {
                        const isSelected = currentEpp.includes(cls);
                        return (
                          <button
                            key={cls}
                            type="button"
                            onClick={() => {
                              if (isDefault) {
                                setDefaultZoneEpp((prev) =>
                                  prev.includes(cls) ? prev.filter((e) => e !== cls) : [...prev, cls]
                                );
                              } else {
                                toggleZoneEpp(zone.id, cls);
                              }
                            }}
                            className={`rounded-xl border px-2.5 py-1 text-xs font-medium transition ${
                              isSelected
                                ? "border-ok-500/50 bg-ok-500/20 text-ok-200"
                                : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-400 hover:border-[#444448] hover:text-steel-200"
                            }`}
                          >
                            {cls}
                          </button>
                        );
                      })}
                    </div>
                    {currentEpp.length === 0 && (
                      <p className="mt-1.5 text-[10px] text-steel-500">Sin EPP requerido → solo informativa</p>
                    )}
                    {currentEpp.length > 0 && (
                      <p className="mt-1.5 text-[10px] text-ok-400">Requiere: {currentEpp.join(", ")}</p>
                    )}
                  </>
                )}
              </div>

              <p className="mt-4 text-[10px] text-steel-600 text-center">
                Detección sigue activa · cambios se guardan al cerrar
              </p>
            </div>
          </div>
        );
      })()}

      <div className="mt-3 flex flex-wrap gap-3 text-xs text-steel-400">
        {!isActive ? (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-steel-500" />
            Estado cámara · Inactiva
          </span>
        ) : !isPaused ? (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-ok-400" />
            Estado cámara · Activa
          </span>
        ) : (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-accent-400" />
            Estado cámara · En pausa
          </span>
        )}
        {isActive && cooldownRemaining > 0 && (
          <span className="text-warn-300">· cooldown {cooldownRemaining}s</span>
        )}
        {isActive && zones.length > 0 && (
          <span className="text-ok-400">· {zones.length} zona{zones.length > 1 ? "s" : ""}</span>
        )}
        {isActive && defaultZoneEpp.length > 0 && (
          <span className="text-ok-400">· por defecto: {defaultZoneEpp.join(", ")}</span>
        )}
      </div>
    </section>
  );
}
