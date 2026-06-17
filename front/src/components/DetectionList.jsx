const resultStyles = {
  con:        "border-ok-500/30 bg-ok-500/15 text-ok-300",
  sin:        "border-warn-500/30 bg-warn-500/15 text-warn-300",
  mixto:      "border-accent-500/30 bg-accent-500/15 text-accent-300",
  cumple:     "border-ok-500/30 bg-ok-500/15 text-ok-300",
  "no cumple":"border-warn-500/30 bg-warn-500/15 text-warn-300",
};

export default function DetectionList({ items, onSelectItem, formatTimestamp, fullWidth = false }) {
  return (
    <section className={`flex flex-col rounded-2xl border border-steel-200 bg-steel-700 p-4 shadow-glow ${fullWidth ? "" : "h-full"}`}>
      <div>
        <p className="text-[10px] uppercase tracking-[0.3em] text-accent-500/80">Actividad reciente</p>
        <h2 className="mt-1 text-xl font-semibold text-white">Últimas detecciones</h2>
      </div>

      <div className={`mt-3 overflow-y-auto pr-1 scrollbar-thin ${fullWidth ? "" : "flex-1 min-h-0"}`}>
        {items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-steel-200 bg-steel-800 px-4 py-5 text-center text-sm text-steel-400">
            No hay detecciones aún.
          </div>
        ) : (
          <div className="space-y-2">
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectItem(item)}
                className="flex w-full gap-2.5 rounded-xl border border-steel-200 bg-steel-800 p-2.5 text-left transition hover:border-accent-500/40 hover:bg-steel-300"
              >
                <div className="h-12 w-12 shrink-0 overflow-hidden rounded-lg border border-steel-200 bg-steel-900">
                  <img src={item.previewUrl} alt={item.name} className="h-full w-full object-cover" />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex items-start justify-between gap-2">
                    <p className="truncate text-sm font-medium text-white">{item.name}</p>
                    <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.2em] ${
                      resultStyles[item.result] ?? resultStyles.con
                    }`}>
                      {item.result}
                    </span>
                  </div>

                  <p className="mt-1 text-xs text-steel-400">
                    {formatTimestamp(item.timestamp)}
                  </p>
                  {item.alertingZones?.length > 0 && (
                    <p className="mt-0.5 text-xs text-warn-300">
                      {item.alertingZones.map((zr) => zr.label || zr.zoneId).join(" · ")}
                    </p>
                  )}
                  {!item.alertingZones?.length && item.defaultZoneResult && !item.defaultZoneResult.compliant && (
                    <p className="mt-0.5 text-xs text-warn-300">Zona por defecto</p>
                  )}
                  <p className="mt-0.5 text-xs text-steel-400">
                    {item.detections.length > 0
                      ? `${item.detections.length} det. EPP`
                      : item.personCount > 0
                        ? `${item.personCount} persona${item.personCount !== 1 ? "s" : ""} detectada${item.personCount !== 1 ? "s" : ""}`
                        : "sin detecciones"}
                    {item.confidence > 0 ? ` · conf. ${item.confidence.toFixed(2)}` : ""}
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
