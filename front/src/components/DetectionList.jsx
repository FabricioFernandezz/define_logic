const resultStyles = {
  con: "border-ok-500/20 bg-ok-500/10 text-ok-200",
  sin: "border-warn-500/20 bg-warn-500/10 text-warn-200",
  mixto: "border-accent-500/20 bg-accent-500/10 text-accent-200",
};

export default function DetectionList({ items, onSelectItem, formatTimestamp }) {
  return (
    <section className="flex h-full flex-col rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Actividad reciente</p>
        <h2 className="mt-1 text-2xl font-semibold text-white">Últimas detecciones</h2>
        
      </div>

      <div className="mt-5 space-y-3 overflow-y-auto pr-1 scrollbar-thin">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => onSelectItem(item)}
            className="flex w-full gap-3 rounded-[1.5rem] border border-white/8 bg-steel-950/70 p-3 text-left transition hover:border-accent-400/30 hover:bg-steel-900"
          >
            <div className="h-16 w-16 shrink-0 overflow-hidden rounded-2xl border border-white/8 bg-steel-900">
              <img src={item.previewUrl} alt={item.name} className="h-full w-full object-cover" />
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-2">
                <p className="truncate text-sm font-medium text-white">{item.name}</p>
                <span
                  className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.25em] ${
                    resultStyles[item.result] ?? resultStyles.con
                  }`}
                >
                  {item.result}
                </span>
              </div>

              <p className="mt-2 text-xs leading-5 text-steel-400">
                {formatTimestamp(item.timestamp)}
              </p>
              <p className="mt-1 text-xs leading-5 text-steel-300">
                {item.detections.length} personas detectadas · confianza media {item.confidence.toFixed(2)}
              </p>
            </div>
          </button>
        ))}
      </div>

      {items.length === 0 && (
        <div className="mt-4 rounded-3xl border border-dashed border-white/10 bg-steel-950/70 p-6 text-center text-sm text-steel-400">
          No hay detecciones aún.
        </div>
      )}
    </section>
  );
}