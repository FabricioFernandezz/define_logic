const getSummaryFromDetections = (detections) => {
  const helmetCount = detections.filter((item) => item.helmetDetected).length;
  const noHelmetCount = detections.length - helmetCount;

  return { helmetCount, noHelmetCount };
};

const statusStyles = {
  helmet: "border-ok-500/20 bg-ok-500/10 text-ok-200",
  noHelmet: "border-warn-500/20 bg-warn-500/10 text-warn-200",
};

export default function DetectionViewer({
  image,
  detections,
  isProcessing,
  processError,
  onProcess,
  onNavigateHistory,
}) {
  const summary = getSummaryFromDetections(detections);
  const displayImageSrc = image?.processedPreviewUrl || image?.previewUrl;
  const canProcess = Boolean(image?.file) && !isProcessing;

  return (
    <section id="viewer-section" className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Procesamiento</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Visor de resultados</h2>
          <p className="mt-2 text-sm leading-6 text-steel-300">
            El visor consume el backend de inferencia y dibuja los resultados sobre la imagen estática seleccionada.
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onProcess}
            disabled={!canProcess}
            className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-accent-500 to-ok-500 px-5 py-3 text-sm font-semibold text-steel-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
          >
            {isProcessing ? "Procesando..." : "Procesar imagen"}
          </button>
          <button
            type="button"
            onClick={onNavigateHistory}
            className="inline-flex items-center justify-center rounded-2xl border border-white/8 bg-white/5 px-5 py-3 text-sm font-medium text-white transition hover:border-white/15 hover:bg-white/10"
          >
            Ver actividad reciente
          </button>
        </div>
      </div>

      {image && !image.file && (
        <div className="mt-3 rounded-2xl border border-accent-500/20 bg-accent-500/10 px-4 py-3 text-xs text-accent-100">
          Esta imagen proviene de actividad reciente y no conserva el archivo original. Carga el archivo nuevamente para reprocesar con IA.
        </div>
      )}

      {processError && (
        <div className="mt-4 rounded-2xl border border-warn-500/30 bg-warn-500/10 px-4 py-3 text-sm text-warn-200">
          {processError}
        </div>
      )}

      <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
        <div className="relative overflow-hidden rounded-[1.75rem] border border-white/8 bg-steel-950/80">
          {image ? (
            <div className="relative">
              <img
                src={displayImageSrc}
                alt={image.name}
                className="block h-auto w-full object-contain"
              />

              {detections.length > 0 && !image?.processedPreviewUrl && (
                <div className="absolute inset-0">
                  {detections.map((detection) => {
                    const left = (detection.bbox.x / image.naturalWidth) * 100;
                    const top = (detection.bbox.y / image.naturalHeight) * 100;
                    const width = (detection.bbox.width / image.naturalWidth) * 100;
                    const height = (detection.bbox.height / image.naturalHeight) * 100;
                    const isHelmet = detection.helmetDetected;

                    return (
                      <div
                        key={detection.id}
                        className={`absolute rounded-3xl border-2 ${
                          isHelmet ? "border-ok-400" : "border-warn-400"
                        }`}
                        style={{
                          left: `${left}%`,
                          top: `${top}%`,
                          width: `${width}%`,
                          height: `${height}%`,
                        }}
                      >
                        <div
                          className={`absolute -top-4 left-2 rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.25em] ${
                            isHelmet ? statusStyles.helmet : statusStyles.noHelmet
                          }`}
                        >
                          {detection.label}
                        </div>
                        <div className="absolute bottom-2 left-2 rounded-xl bg-steel-950/90 px-2.5 py-1 text-xs text-white shadow-lg">
                          {detection.confidence.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="absolute left-4 top-4 flex flex-wrap gap-2">
                <span className="rounded-full border border-white/8 bg-steel-950/80 px-3 py-1 text-xs text-steel-200">
                  {image.name}
                </span>
                {image.result && (
                  <span className="rounded-full border border-white/8 bg-white/10 px-3 py-1 text-xs text-white">
                    {image.result}
                  </span>
                )}
                {image.modelName && (
                  <span className="rounded-full border border-accent-400/30 bg-accent-500/20 px-3 py-1 text-xs text-accent-100">
                    {image.modelName}
                  </span>
                )}
              </div>
            </div>
          ) : (
            <div className="flex min-h-[520px] flex-col items-center justify-center px-6 text-center">
              <div className="flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-accent-500/12 text-4xl text-accent-200">
                ⦿
              </div>
              <h3 className="mt-5 text-xl font-semibold text-white">Visor principal</h3>
              <p className="mt-2 max-w-lg text-sm leading-6 text-steel-400">
                Carga una imagen estática para verla aquí con sus boxes de casco/sin casco. La arquitectura ya está conectada a una API Python real.
              </p>
              <p className="mt-4 text-xs uppercase tracking-[0.3em] text-steel-500">
                SOLO IMÁGENES ESTÁTICAS
              </p>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-4 rounded-[1.75rem] border border-white/8 bg-steel-900/70 p-4">
          <div className="rounded-2xl border border-white/8 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.25em] text-steel-400">Resumen</p>
            <div className="mt-3 space-y-3 text-sm">
              <div className="flex items-center justify-between rounded-2xl bg-ok-500/10 px-4 py-3 text-ok-200">
                <span>Casco detectado</span>
                <span className="font-semibold">{summary.helmetCount}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-warn-500/10 px-4 py-3 text-warn-200">
                <span>Sin casco</span>
                <span className="font-semibold">{summary.noHelmetCount}</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-white/8 bg-white/5 p-4 text-sm text-steel-300">
            <p className="font-medium text-white">Notas de diseño</p>
            <p className="mt-2 leading-6">
              El visor usa imagen estática, el backend procesa YOLO + ViT y el frontend pinta las detecciones encima del preview.
            </p>
            {(image?.modelName || image?.processingTimeMs) && (
              <div className="mt-3 space-y-1 text-xs text-steel-300">
                {image?.modelName && <p>Modelo: {image.modelName}</p>}
                {Number.isFinite(image?.processingTimeMs) && image.processingTimeMs > 0 && (
                  <p>Tiempo: {image.processingTimeMs} ms</p>
                )}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/8 bg-gradient-to-br from-accent-500/10 to-ok-500/10 p-4 text-sm text-steel-200">
            <p className="text-xs uppercase tracking-[0.25em] text-accent-200">Estado</p>
            <p className="mt-2 leading-6">
              {isProcessing
                ? "El backend está procesando la imagen seleccionada."
                : image
                  ? "Listo para procesar una nueva imagen."
                  : "Espera a que el usuario cargue una imagen."}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}