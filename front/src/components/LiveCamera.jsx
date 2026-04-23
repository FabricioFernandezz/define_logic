import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const SAMPLE_INTERVAL_MS = 1000;
const COOLDOWN_SECS = 10;

export default function LiveCamera({
  onCameraDetection,
  active = true,
  keepAlive = false,
  onKeepAliveChange,
  onIsActiveChange,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const isProcessingRef = useRef(false);
  const isCooldownRef = useRef(false);

  const [isActive, setIsActive] = useState(false);
  const [alertVisible, setAlertVisible] = useState(false);
  const [cooldownRemaining, setCooldownRemaining] = useState(0);
  const [lastResult, setLastResult] = useState(null);
  const [error, setError] = useState("");

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
    setAlertVisible(false);
    setCooldownRemaining(0);
    setLastResult(null);
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
    if (isProcessingRef.current || isCooldownRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    isProcessingRef.current = true;
    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const resp = await fetch(`${API_BASE_URL}/api/detect-frame`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) return;

      const data = await resp.json();
      setLastResult(data);

      if (data.person_detected && data.alert && data.annotated_frame) {
        triggerCooldown();
        onCameraDetection?.({
          frameDataUrl: data.annotated_frame,
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
  }, [triggerCooldown, onCameraDetection]);

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

  // Stop camera when navigating away from live view (unless keepAlive)
  useEffect(() => {
    if (!active && !keepAlive && isActive) {
      stopCamera();
    }
  }, [active, keepAlive, isActive, stopCamera]);

  // Notify parent of running state
  useEffect(() => {
    onIsActiveChange?.(isActive);
  }, [isActive, onIsActiveChange]);

  // Cleanup on unmount
  useEffect(() => () => stopCamera(), [stopCamera]);

  const statusBadge = lastResult
    ? lastResult.alert
      ? { text: "Sin casco detectado", cls: "border-warn-500/40 bg-warn-500/20 text-warn-100" }
      : lastResult.person_detected
        ? { text: "Con casco", cls: "border-ok-500/40 bg-ok-500/20 text-ok-100" }
        : { text: "Sin personas", cls: "border-white/8 bg-steel-950/80 text-steel-300" }
    : null;

  return (
    <section id="camera-section" className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Cámara en vivo</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Detección en tiempo real</h2>
          <p className="mt-2 text-sm leading-6 text-steel-300">
            Activar la cámara para detectar cascos automáticamente cada segundo.
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          {isActive ? (
            <button
              type="button"
              onClick={stopCamera}
              className="inline-flex items-center justify-center rounded-2xl border border-warn-500/30 bg-warn-500/10 px-5 py-3 text-sm font-semibold text-warn-200 transition hover:bg-warn-500/20"
            >
              Detener cámara
            </button>
          ) : (
            <button
              type="button"
              onClick={startCamera}
              className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-accent-500 to-ok-500 px-5 py-3 text-sm font-semibold text-steel-950 transition hover:brightness-110"
            >
              Activar cámara
            </button>
          )}

          {/* Background persistence toggle — only when camera is active */}
          {isActive && (
            <button
              type="button"
              onClick={() => onKeepAliveChange?.(!keepAlive)}
              className={`inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-medium transition ${
                keepAlive
                  ? "border-ok-500/40 bg-ok-500/15 text-ok-200 hover:bg-ok-500/25"
                  : "border-white/8 bg-white/5 text-steel-400 hover:border-white/15 hover:text-steel-200"
              }`}
            >
              <span
                className={`inline-block h-1.5 w-1.5 rounded-full ${
                  keepAlive ? "animate-pulse bg-ok-400" : "bg-steel-600"
                }`}
              />
              {keepAlive ? "Activa en segundo plano" : "Pausar al cambiar vista"}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-2xl border border-warn-500/30 bg-warn-500/10 px-4 py-3 text-sm text-warn-200">
          {error}
        </div>
      )}

      {alertVisible && (
        <div className="mt-4 animate-pulse rounded-2xl border border-warn-500/50 bg-warn-500/20 px-4 py-4 text-center">
          <p className="text-lg font-bold text-warn-100">⚠ Persona sin casco detectada</p>
          <p className="mt-1 text-sm text-warn-300">Próxima detección en {cooldownRemaining}s</p>
        </div>
      )}

      <div className="relative mt-5 overflow-hidden rounded-[1.75rem] border border-white/8 bg-steel-950/80">
        <video
          ref={videoRef}
          className={`block h-auto w-full ${isActive ? "" : "hidden"}`}
          style={{ maxHeight: "480px", objectFit: "cover" }}
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

        {!isActive && (
          <div className="flex min-h-[360px] flex-col items-center justify-center px-6 text-center">
            <div className="flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-accent-500/12 text-4xl text-accent-200">
              ◎
            </div>
            <h3 className="mt-5 text-xl font-semibold text-white">Cámara inactiva</h3>
            <p className="mt-2 max-w-lg text-sm leading-6 text-steel-400">
              Haz click en "Activar cámara" para iniciar la detección en tiempo real.
            </p>
          </div>
        )}
      </div>

      {isActive && (
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-steel-400">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-ok-400" />
            Activo · muestreo cada 1s
          </span>
          {cooldownRemaining > 0 && (
            <span className="text-warn-300">· cooldown {cooldownRemaining}s</span>
          )}
          {keepAlive && (
            <span className="text-ok-400">· persiste al navegar</span>
          )}
        </div>
      )}
    </section>
  );
}
