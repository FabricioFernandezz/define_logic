import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const formatFileSize = (size) => {
  if (!size) return "0 KB";
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
};

const COMPLIANT_STYLE = "border-ok-500/20 bg-ok-500/10 text-ok-200";
const NON_COMPLIANT_STYLE = "border-warn-500/20 bg-warn-500/10 text-warn-200";

export default function EppImageDetector({ onEppDetection }) {
  const [isDragging, setIsDragging] = useState(false);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");

  const loadImage = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const previewUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImage({
        file,
        name: file.name,
        previewUrl,
        naturalWidth: img.naturalWidth,
        naturalHeight: img.naturalHeight,
      });
      setResult(null);
      setError("");
    };
    img.src = previewUrl;
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) loadImage(file);
    e.target.value = "";
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    loadImage(e.dataTransfer.files?.[0]);
  };

  const handleProcess = async () => {
    if (!image?.file || isProcessing) return;
    setIsProcessing(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", image.file);
      const resp = await fetch(`${API_BASE_URL}/api/epp/detect-image`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Error procesando imagen EPP");
      }
      const data = await resp.json();
      setResult(data);
      onEppDetection?.({ image, result: data });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error inesperado");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    setResult(null);
    setError("");
  };

  const displaySrc = result?.annotatedImage || image?.previewUrl;
  const detections = result?.detections || [];
  const summary = result?.summary;

  return (
    <div className="flex flex-col gap-6">
      {/* Upload section */}
      <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">EPP · Carga</p>
            <h2 className="mt-1 text-2xl font-semibold text-white">Detección EPP en imagen</h2>
            <p className="mt-2 text-sm leading-6 text-steel-300">
              Modelo YOLO directo para detectar EPP. Procesa en una sola pasada.
            </p>
          </div>
          <label className="inline-flex cursor-pointer items-center justify-center rounded-2xl bg-gradient-to-r from-accent-500 to-ok-500 px-5 py-3 text-sm font-semibold text-steel-950 transition hover:brightness-110">
            Seleccionar archivo
            <input type="file" accept="image/*" className="hidden" onChange={handleChange} />
          </label>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
            onDrop={handleDrop}
            className={`rounded-[1.75rem] border border-dashed bg-steel-950/70 p-4 transition ${
              isDragging ? "border-accent-400/60 bg-accent-500/5" : "border-white/10"
            }`}
          >
            {image ? (
              <div className="space-y-4">
                <div className="overflow-hidden rounded-[1.5rem] border border-white/8 bg-steel-900/90">
                  <img src={image.previewUrl} alt={image.name} className="h-64 w-full object-cover sm:h-72" />
                </div>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-medium text-white">{image.name}</p>
                    <p className="text-xs text-steel-400">{image.naturalWidth} × {image.naturalHeight} px</p>
                  </div>
                  <div className="text-xs text-steel-400">Tamaño: {formatFileSize(image.file?.size)}</div>
                </div>
              </div>
            ) : (
              <div className="flex h-full min-h-[280px] flex-col items-center justify-center rounded-[1.5rem] bg-gradient-to-br from-white/5 to-transparent px-6 text-center">
                <p className="mt-4 text-lg font-medium text-white">
                  {isDragging ? "Soltar imagen aquí" : "Seleccionar imagen"}
                </p>
                <p className="mt-2 max-w-md text-sm leading-6 text-steel-400">
                  Arrastra o usa el botón para cargar
                </p>
              </div>
            )}
          </div>

          <div className="flex flex-col justify-between gap-4 rounded-[1.75rem] border border-white/8 bg-steel-900/70 p-4">
            <div>
              <p className="text-sm font-medium text-white">Modelo EPP (YOLO ONNX)</p>
              <ul className="mt-3 space-y-3 text-sm text-steel-300">
                <li className="flex gap-3">
                  <span className="mt-1 h-2.5 w-2.5 rounded-full bg-accent-400" />
                  Detección directa sin etapa de clasificación.
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2.5 w-2.5 rounded-full bg-ok-400" />
                  Verde = EPP presente · Rojo = EPP ausente.
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2.5 w-2.5 rounded-full bg-warn-400" />
                  Una sola pasada de inferencia ONNX.
                </li>
              </ul>
            </div>
            <button
              type="button"
              onClick={handleClear}
              disabled={!image}
              className="inline-flex items-center justify-center rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-sm font-medium text-white transition hover:border-white/15 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Limpiar imagen
            </button>
          </div>
        </div>
      </section>

      {/* Viewer / results section */}
      <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">EPP · Resultado</p>
            <h2 className="mt-1 text-2xl font-semibold text-white">Visor de resultados EPP</h2>
            <p className="mt-2 text-sm leading-6 text-steel-300">
              El backend detecta EPP con yolo26_epp y devuelve imagen anotada.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={handleProcess}
              disabled={!image?.file || isProcessing}
              className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-accent-500 to-ok-500 px-5 py-3 text-sm font-semibold text-steel-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {isProcessing ? "Procesando..." : "Detectar EPP"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-2xl border border-warn-500/30 bg-warn-500/10 px-4 py-3 text-sm text-warn-200">
            {error}
          </div>
        )}

        <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
          <div className="relative overflow-hidden rounded-[1.75rem] border border-white/8 bg-steel-950/80">
            {displaySrc ? (
              <div className="relative">
                <img
                  src={displaySrc}
                  alt={image?.name}
                  className="block h-auto w-full object-contain"
                />
                {detections.length > 0 && !result?.annotatedImage && (
                  <div className="absolute inset-0">
                    {detections.map((det) => {
                      const [x1, y1, x2, y2] = det.bbox_pixels;
                      const left = (x1 / image.naturalWidth) * 100;
                      const top = (y1 / image.naturalHeight) * 100;
                      const width = ((x2 - x1) / image.naturalWidth) * 100;
                      const height = ((y2 - y1) / image.naturalHeight) * 100;
                      return (
                        <div
                          key={det.id}
                          className={`absolute rounded-3xl border-2 ${det.isCompliant ? "border-ok-400" : "border-warn-400"}`}
                          style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                        >
                          <div className={`absolute -top-4 left-2 rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.25em] ${det.isCompliant ? COMPLIANT_STYLE : NON_COMPLIANT_STYLE}`}>
                            {det.label}
                          </div>
                          <div className="absolute bottom-2 left-2 rounded-xl bg-steel-950/90 px-2.5 py-1 text-xs text-white shadow-lg">
                            {det.confidence.toFixed(2)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
                <div className="absolute left-4 top-4 flex flex-wrap gap-2">
                  <span className="rounded-full border border-white/8 bg-steel-950/80 px-3 py-1 text-xs text-steel-200">{image?.name}</span>
                  {result && (
                    <span className="rounded-full border border-accent-400/30 bg-accent-500/20 px-3 py-1 text-xs text-accent-100">
                      yolo26_epp · {result.processingTimeMs}ms
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex min-h-[520px] flex-col items-center justify-center px-6 text-center">
                <div className="flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-accent-500/12 text-4xl text-accent-200">
                  ⦿
                </div>
                <h3 className="mt-5 text-xl font-semibold text-white">Visor EPP</h3>
                <p className="mt-4 text-xs uppercase tracking-[0.3em] text-steel-500">SOLO IMÁGENES</p>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-4 rounded-[1.75rem] border border-white/8 bg-steel-900/70 p-4">
            <div className="rounded-2xl border border-white/8 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.25em] text-steel-400">Resumen</p>
              <div className="mt-3 space-y-3 text-sm">
                <div className="flex items-center justify-between rounded-2xl bg-ok-500/10 px-4 py-3 text-ok-200">
                  <span>EPP presente</span>
                  <span className="font-semibold">{summary?.compliantCount ?? 0}</span>
                </div>
                <div className="flex items-center justify-between rounded-2xl bg-warn-500/10 px-4 py-3 text-warn-200">
                  <span>EPP ausente</span>
                  <span className="font-semibold">{summary?.nonCompliantCount ?? 0}</span>
                </div>
              </div>
            </div>

            {detections.length > 0 && (
              <div className="rounded-2xl border border-white/8 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-steel-400">Detecciones</p>
                <div className="mt-2 space-y-2">
                  {detections.map((det) => (
                    <div key={det.id} className="flex items-center justify-between text-xs">
                      <span className={`rounded-full border px-2 py-0.5 ${det.isCompliant ? COMPLIANT_STYLE : NON_COMPLIANT_STYLE}`}>
                        {det.label}
                      </span>
                      <span className="text-steel-400">{(det.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-2xl border border-white/8 bg-gradient-to-br from-accent-500/10 to-ok-500/10 p-4 text-sm text-steel-200">
              <p className="text-xs uppercase tracking-[0.25em] text-accent-200">Estado</p>
              <p className="mt-2 leading-6">
                {isProcessing
                  ? "Inferencia ONNX en progreso."
                  : result
                    ? `${summary?.result} · confianza ${(summary?.confidence * 100).toFixed(0)}%`
                    : image
                      ? "Presiona Detectar EPP."
                      : "Carga una imagen primero."}
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
