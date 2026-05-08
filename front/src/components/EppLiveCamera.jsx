import { useCallback, useEffect, useRef, useState } from "react";
import EppZoneEditor from "./EppZoneEditor";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const SAMPLE_INTERVAL_MS = 1000;
const COOLDOWN_SECS = 10;

export default function EppLiveCamera({ active = true, onEppCameraDetection }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const isProcessingRef = useRef(false);
  const isCooldownRef = useRef(false);
  const lastFrameUrlRef = useRef(null);

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

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    isProcessingRef.current = false;
    isCooldownRef.current = false;
    setIsActive(false);
    setIsPaused(false);
    setIsEditingZones(false);
    setAlertVisible(false);
    setCooldownRemaining(0);
    setLastResult(null);
    setAnnotatedFrame(null);
    setFrozenFrame(null);
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

  const captureAndSend = useCallback(async () => {
    if (isProcessingRef.current || isPaused) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    // Keep last frame for zone editor reference
    lastFrameUrlRef.current = canvas.toDataURL("image/jpeg", 0.8);

    isProcessingRef.current = true;
    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      if (zones.length > 0) {
        formData.append(
          "zones",
          JSON.stringify(
            zones.map((z) => ({
              id: z.id,
              label: z.label,
              bbox: z.bbox,
              required_epp: z.requiredEpp,
              active: z.active !== false,
              require_person: z.requirePerson === true,
            }))
          )
        );
      }

      if (defaultZoneEpp.length > 0) {
        formData.append("default_zone_epp", JSON.stringify(defaultZoneEpp));
      }

      formData.append("default_zone_active", defaultZoneActive ? "true" : "false");
      formData.append("default_zone_require_person", defaultZoneRequirePerson ? "true" : "false");

      const resp = await fetch(`${API_BASE_URL}/api/epp/detect-frame`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) return;

      const data = await resp.json();
      setLastResult(data);

      if (data.annotated_frame) {
        setAnnotatedFrame(data.annotated_frame);
      }

      if (data.alert && !isCooldownRef.current) {
        triggerCooldown();
        onEppCameraDetection?.({
          frameDataUrl: data.annotated_frame || lastFrameUrlRef.current,
          width: canvas.width,
          height: canvas.height,
          result: data,
          timestamp: new Date().toISOString(),
        });
      }
    } catch {
      // silence transient network errors
    } finally {
      isProcessingRef.current = false;
    }
  }, [isPaused, zones, defaultZoneActive, defaultZoneRequirePerson, triggerCooldown, onEppCameraDetection]);

  const startCamera = useCallback(async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsActive(true);
      intervalRef.current = setInterval(captureAndSend, SAMPLE_INTERVAL_MS);
    } catch (err) {
      setError("No se pudo acceder a la cámara: " + (err?.message || "permiso denegado"));
    }
  }, [captureAndSend]);

  const handlePause = () => {
    setIsPaused(true);
    setFrozenFrame(lastFrameUrlRef.current);
    setIsEditingZones(true);
  };

  const handleResume = () => {
    setIsPaused(false);
    setIsEditingZones(false);
    isCooldownRef.current = false;
  };

  useEffect(() => {
    if (!active && isActive) stopCamera();
  }, [active, isActive, stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Restart interval when isPaused changes so captureAndSend picks up new state
  useEffect(() => {
    if (!isActive) return;
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(captureAndSend, SAMPLE_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isActive, captureAndSend]);

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
      statusBadge = { text: "Sin detecciones", cls: "border-white/8 bg-steel-950/80 text-steel-300" };
    }
  }

  return (
    <section id="epp-camera-section" className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Cámara</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Detección EPP en tiempo real</h2>
          <p className="mt-2 text-sm leading-6 text-steel-300">
            El modelo detecta EPP directamente. Define zonas con requisitos distintos.
          </p>
        </div>

        <div className="flex flex-col items-end gap-2">
          {!isActive ? (
            <button
              type="button"
              onClick={startCamera}
              className="inline-flex items-center justify-center rounded-2xl bg-sky-100/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-100/30 ring-1 ring-white/20"
            >
              Activar cámara EPP
            </button>
          ) : (
            <div className="flex flex-wrap items-center gap-2 justify-end">
              {!isPaused ? (
                <button
                  type="button"
                  onClick={handlePause}
                  className="inline-flex items-center justify-center rounded-2xl bg-sky-100/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-100/30 ring-1 ring-white/20"
                >
                  Pausar · Editar zonas
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleResume}
                  className="inline-flex items-center justify-center rounded-2xl bg-sky-100/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-100/30 ring-1 ring-white/20"
                >
                  Reanudar detecciones
                </button>
              )}
              <button
                type="button"
                onClick={stopCamera}
                className="inline-flex items-center justify-center rounded-2xl bg-sky-100/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-100/30 ring-1 ring-white/20"
              >
                Detener cámara
              </button>
            </div>
          )}

          {(zones.length > 0 || defaultZoneEpp.length > 0) && isActive && (
            <span className="text-xs text-ok-300">
              {zones.length > 0 && `${zones.length} zona${zones.length > 1 ? "s" : ""}`}
              {zones.length > 0 && defaultZoneEpp.length > 0 && " · "}
              {defaultZoneEpp.length > 0 && "zona por defecto activa"}
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
        <div className="mt-4 rounded-2xl border border-accent-400/30 bg-accent-500/10 px-4 py-3 text-sm text-accent-200">
          Detecciones en pausa · la cámara sigue activa · dibuja zonas y reanuda cuando estés listo
        </div>
      )}

      {alertVisible && !isPaused && (
        <div className="mt-4 animate-pulse rounded-2xl border border-warn-500/50 bg-warn-500/20 px-4 py-4 text-center">
          <p className="text-lg font-bold text-warn-100">⚠ EPP incompleto detectado</p>
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
              <span
                key={zr.zoneId}
                className={`rounded-xl border px-3 py-1.5 text-xs font-medium ${
                  inactive
                    ? "border-white/8 bg-white/5 text-steel-500"
                    : !zr.hasRequired
                      ? "border-white/8 bg-white/5 text-steel-400"
                      : zr.compliant
                        ? "border-ok-500/30 bg-ok-500/15 text-ok-200"
                        : "border-warn-500/30 bg-warn-500/15 text-warn-200"
                }`}
              >
                {zr.label || zr.zoneId}
                {inactive && <span className="ml-1 opacity-60">· inactiva</span>}
                {!inactive && zr.hasRequired && (
                  <span className="ml-1">
                    {zr.compliant ? "✓" : `✗ ${nonCompliantParts.join(" · ")}`}
                  </span>
                )}
              </span>
            );
          })}
          {defaultZoneResult && (() => {
            const violations = defaultZoneResult.violatingEpp || [];
            const missing = defaultZoneResult.missingEpp || [];
            const nonCompliantParts = [
              violations.length > 0 ? `sin: ${violations.join(", ")}` : null,
              missing.length > 0 ? `falta: ${missing.join(", ")}` : null,
            ].filter(Boolean);
            const inactive = defaultZoneResult.active === false;
            return (
              <span
                className={`rounded-xl border px-3 py-1.5 text-xs font-medium ${
                  inactive
                    ? "border-white/8 bg-white/5 text-steel-500"
                    : defaultZoneResult.compliant
                      ? "border-ok-500/30 bg-ok-500/15 text-ok-200"
                      : "border-warn-500/30 bg-warn-500/15 text-warn-200"
                }`}
              >
                Por defecto
                {inactive && <span className="ml-1 opacity-60">· inactiva</span>}
                {!inactive && (
                  <span className="ml-1">
                    {defaultZoneResult.compliant ? "✓" : `✗ ${nonCompliantParts.join(" · ")}`}
                  </span>
                )}
              </span>
            );
          })()}
        </div>
      )}

      {/* Camera + side panel */}
      <div className={`mt-5 grid gap-4 ${isEditingZones ? "" : "xl:grid-cols-[minmax(0,1fr)_240px]"}`}>
        {/* Video */}
        <div className="relative overflow-hidden rounded-[1.75rem] border border-white/8 bg-steel-950/80">
          <video
            ref={videoRef}
            className={`block h-auto w-full ${isActive ? "" : "hidden"}`}
            style={{ maxHeight: "420px", objectFit: "cover" }}
            playsInline
            muted
          />
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
              <span className="rounded-full border border-accent-400/40 bg-steel-950/90 px-3 py-1 text-xs font-semibold text-accent-200">
                ⏸ En pausa
              </span>
            </div>
          )}

          {!isActive && (
            <div className="flex min-h-[360px] flex-col items-center justify-center px-6 text-center">
              <div className="flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-accent-500/12 text-4xl text-accent-200">
                ◎
              </div>
              <h3 className="mt-5 text-xl font-semibold text-white">Cámara EPP inactiva</h3>
              <p className="mt-2 max-w-lg text-sm leading-6 text-steel-400">
                Click en "Activar cámara EPP" para iniciar la detección en tiempo real.
              </p>
            </div>
          )}
        </div>

        {/* Side panel: last annotated frame + detections (only when not editing zones) */}
        {!isEditingZones && (
          <div className="flex flex-col gap-4">
            {annotatedFrame && (
              <div className="rounded-[1.75rem] border border-white/8 bg-steel-900/70 overflow-hidden">
                <p className="px-4 pt-3 text-xs uppercase tracking-[0.25em] text-steel-400">Último frame</p>
                <img src={annotatedFrame} alt="Frame anotado" className="mt-2 w-full object-contain" />
              </div>
            )}

            {detections.length > 0 && (
              <div className="rounded-[1.75rem] border border-white/8 bg-steel-900/70 p-4">
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
              <div className="rounded-[1.75rem] border border-white/8 bg-steel-900/70 p-4 text-sm text-steel-400">
                Esperando detecciones EPP…
              </div>
            )}
          </div>
        )}
      </div>

      {/* Zone editor — shown when paused */}
      {isEditingZones && (
        <div className="mt-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-white">Editor de zonas</p>
              <p className="text-xs text-steel-400">Dibuja rectángulos sobre el frame y asigna EPP requerido por zona</p>
            </div>
            <button
              type="button"
              onClick={handleResume}
              className="inline-flex items-center justify-center rounded-2xl bg-sky-100/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-100/30 ring-1 ring-white/20"
            >
              Reanudar con estas zonas
            </button>
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

      {isActive && (
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-steel-400">
          {!isPaused ? (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-ok-400" />
              Activo · detección continua
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full bg-accent-400" />
              En pausa · cámara activa
            </span>
          )}
          {cooldownRemaining > 0 && (
            <span className="text-warn-300">· cooldown {cooldownRemaining}s</span>
          )}
          {zones.length > 0 && (
            <span className="text-ok-400">· {zones.length} zona{zones.length > 1 ? "s" : ""}</span>
          )}
          {defaultZoneEpp.length > 0 && (
            <span className="text-ok-400">· por defecto: {defaultZoneEpp.join(", ")}</span>
          )}
        </div>
      )}
    </section>
  );
}
