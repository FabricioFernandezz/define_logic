const RESULT_STYLES = {
  con: { badge: "border-ok-500/30 bg-ok-500/15 text-ok-200", dot: "bg-ok-400" },
  sin: { badge: "border-warn-500/30 bg-warn-500/15 text-warn-200", dot: "bg-warn-400" },
  mixto: { badge: "border-accent-500/30 bg-accent-500/15 text-accent-200", dot: "bg-accent-400" },
};

const resultKey = (result = "") => {
  if (result.startsWith("con")) return "con";
  if (result.startsWith("sin")) return "sin";
  return "mixto";
};

export default function DetectionReview({ entry, formatTimestamp, onSave, onDelete, onBack }) {
  if (!entry) return null;

  const rKey = resultKey(entry.result);
  const style = RESULT_STYLES[rKey] ?? RESULT_STYLES.mixto;
  const isCameraFrame = entry.modelName === "live-cam";
  const noHelmetPersons = entry.detections.filter((d) => !d.helmetDetected);
  const helmetPersons = entry.detections.filter((d) => d.helmetDetected);

  return (
    <div className="animate-fadeUp flex flex-col gap-6">
      {/* Header bar */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-2 rounded-2xl border border-white/8 bg-white/5 px-4 py-2 text-sm font-medium text-steel-300 transition hover:border-accent-400/30 hover:bg-white/10 hover:text-white"
        >
          ← Volver
        </button>
        <span className="text-sm text-steel-500">
          {isCameraFrame ? "Frame de cámara en vivo" : "Imagen procesada"}
        </span>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_380px]">
        {/* Left — image + summary */}
        <div className="flex flex-col gap-4">
          <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
            <div className="flex flex-col gap-1">
              <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Detalle</p>
              <h2 className="text-2xl font-semibold text-white truncate">{entry.name}</h2>
              <p className="text-sm text-steel-400">{formatTimestamp(entry.timestamp)}</p>
            </div>

            {/* Annotated image */}
            <div className="mt-4 overflow-hidden rounded-[1.5rem] border border-white/8 bg-steel-950">
              <img
                src={entry.annotatedPreviewUrl || entry.previewUrl}
                alt="Frame detectado"
                className="block h-auto w-full object-contain"
                style={{ maxHeight: "480px" }}
              />
            </div>

            {/* Quick stats row */}
            <div className="mt-4 grid grid-cols-3 gap-3 text-center">
              <div className="rounded-2xl border border-white/8 bg-steel-900/70 px-3 py-3">
                <p className="text-xs uppercase tracking-[0.2em] text-steel-400">Personas</p>
                <p className="mt-1 text-xl font-semibold text-white">{entry.detections.length}</p>
              </div>
              <div className="rounded-2xl border border-ok-500/20 bg-ok-500/10 px-3 py-3">
                <p className="text-xs uppercase tracking-[0.2em] text-ok-300">Con casco</p>
                <p className="mt-1 text-xl font-semibold text-ok-300">{helmetPersons.length}</p>
              </div>
              <div className="rounded-2xl border border-warn-500/20 bg-warn-500/10 px-3 py-3">
                <p className="text-xs uppercase tracking-[0.2em] text-warn-300">Sin casco</p>
                <p className="mt-1 text-xl font-semibold text-warn-300">{noHelmetPersons.length}</p>
              </div>
            </div>
          </section>

          {/* AI decision banner */}
          <div
            className={`rounded-[1.75rem] border px-5 py-4 ${
              noHelmetPersons.length > 0
                ? "border-warn-500/30 bg-warn-500/10"
                : "border-ok-500/30 bg-ok-500/10"
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{noHelmetPersons.length > 0 ? "⚠" : "✓"}</span>
              <div>
                <p
                  className={`text-base font-semibold ${
                    noHelmetPersons.length > 0 ? "text-warn-100" : "text-ok-100"
                  }`}
                >
                  {noHelmetPersons.length > 0
                    ? `${noHelmetPersons.length} persona${noHelmetPersons.length > 1 ? "s" : ""} sin casco detectada${noHelmetPersons.length > 1 ? "s" : ""}`
                    : "Todas las personas llevan casco"}
                </p>
                <p className="text-xs text-steel-400 mt-0.5">
                  Confianza media ViT: {(entry.confidence * 100).toFixed(1)}% ·{" "}
                  {entry.processingTimeMs > 0 ? `${entry.processingTimeMs}ms` : "Tiempo real"}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right — per-person cards + action panel */}
        <div className="flex flex-col gap-4">
          {/* Per-person detection cards */}
          <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Por persona</p>
            <h3 className="mt-1 text-lg font-semibold text-white">Resultados individuales</h3>

            <div className="mt-4 space-y-3">
              {entry.detections.length === 0 && (
                <p className="text-sm text-steel-400">Sin personas detectadas en este frame.</p>
              )}
              {entry.detections.map((det, idx) => {
                const isHelmet = Boolean(det.helmetDetected);
                return (
                  <div
                    key={det.id ?? idx}
                    className={`rounded-2xl border p-4 ${
                      isHelmet
                        ? "border-ok-500/20 bg-ok-500/8"
                        : "border-warn-500/25 bg-warn-500/10"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex h-7 w-7 items-center justify-center rounded-xl text-xs font-bold ${
                            isHelmet
                              ? "bg-ok-500/20 text-ok-200"
                              : "bg-warn-500/20 text-warn-200"
                          }`}
                        >
                          {det.personIndex ?? idx + 1}
                        </span>
                        <span className="text-sm font-medium text-white">
                          Persona {det.personIndex ?? idx + 1}
                        </span>
                      </div>
                      <span
                        className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] ${
                          isHelmet
                            ? "border-ok-500/30 bg-ok-500/10 text-ok-200"
                            : "border-warn-500/30 bg-warn-500/10 text-warn-200"
                        }`}
                      >
                        {det.label}
                      </span>
                    </div>

                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-steel-400">
                      <div>
                        <span className="text-steel-500">Confianza ViT</span>
                        <p className="mt-0.5 font-semibold text-steel-200">
                          {(det.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      {det.bbox && (
                        <div>
                          <span className="text-steel-500">Bbox (px)</span>
                          <p className="mt-0.5 font-semibold text-steel-200">
                            ({det.bbox.x}, {det.bbox.y}) → ({det.bbox.x + det.bbox.width}, {det.bbox.y + det.bbox.height})
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Confidence bar */}
                    <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-steel-800">
                      <div
                        className={`h-full rounded-full transition-all ${
                          isHelmet ? "bg-ok-400" : "bg-warn-400"
                        }`}
                        style={{ width: `${(det.confidence * 100).toFixed(0)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Validation / action panel */}
          <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Validación</p>
            <h3 className="mt-1 text-lg font-semibold text-white">Confirmar decisión de IA</h3>
            <p className="mt-2 text-sm leading-6 text-steel-400">
              Revisar la detección y confirmar si es correcta para registrarla o eliminarla por
              falsa detección.
            </p>

            <div className="mt-5 flex flex-col gap-3">
              {/* Save */}
              <button
                type="button"
                onClick={() => onSave(entry)}
                className="flex w-full items-center justify-center gap-2.5 rounded-2xl bg-gradient-to-r from-ok-600 to-ok-500 px-5 py-3.5 text-sm font-semibold text-white transition hover:brightness-110 active:scale-[0.98]"
              >
                <span className="text-base">✓</span>
                Guardar detección
              </button>

              {/* Delete */}
              <button
                type="button"
                onClick={() => onDelete(entry.id)}
                className="flex w-full items-center justify-center gap-2.5 rounded-2xl border border-warn-500/30 bg-warn-500/10 px-5 py-3.5 text-sm font-semibold text-warn-200 transition hover:bg-warn-500/20 active:scale-[0.98]"
              >
                <span className="text-base">✕</span>
                Eliminar (falsa detección)
              </button>
            </div>

            <p className="mt-4 text-center text-xs text-steel-500">
              Guardar: registra infracción · Eliminar: descarta error del modelo
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
