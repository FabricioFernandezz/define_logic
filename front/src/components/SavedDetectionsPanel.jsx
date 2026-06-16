export default function SavedDetectionsPanel({
  items,
  selectedId,
  onSelect,
  loading = false,
  error = null,
  onRetry,
}) {
  const selected = items.find((item) => item.id === selectedId) || items[0] || null;

  return (
    <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
      <section className="rounded-2xl border border-steel-200 bg-steel-700 p-4 shadow-glow">
        <p className="text-[10px] uppercase tracking-[0.3em] text-accent-500/80">Base de datos</p>
        <h2 className="mt-1 text-xl font-semibold text-white">Imágenes guardadas</h2>

        <div className="mt-4 space-y-2 overflow-y-auto pr-1 scrollbar-thin max-h-[62vh]">
          {loading && (
            <div className="rounded-xl border border-dashed border-steel-200 bg-steel-800 p-5 text-center text-sm text-steel-400">
              Cargando imágenes...
            </div>
          )}

          {!loading && error && (
            <div className="rounded-xl border border-dashed border-warn-500/30 bg-warn-500/10 p-5 text-center">
              <p className="text-sm text-warn-300">{error}</p>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="mt-3 text-xs font-semibold uppercase tracking-[0.2em] text-accent-500 transition hover:text-accent-400"
                >
                  Reintentar
                </button>
              )}
            </div>
          )}

          {!loading && !error && items.map((item) => {
            const isActive = selected?.id === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelect(item.id)}
                className={`flex w-full gap-2.5 rounded-xl border p-2.5 text-left transition ${
                  isActive
                    ? "border-accent-500/40 bg-accent-500/10"
                    : "border-steel-200 bg-steel-800 hover:border-accent-500/30 hover:bg-steel-300"
                }`}
              >
                <div className="h-12 w-12 shrink-0 overflow-hidden rounded-lg border border-steel-200 bg-steel-900">
                  <img src={item.imagen} alt={item.nombre} className="h-full w-full object-cover" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-white">{item.nombre}</p>
                  <p className="mt-0.5 text-xs text-steel-400">ID #{item.id}</p>
                  <p className="mt-0.5 truncate text-xs text-steel-400">
                    {item.descripcion || "Sin descripción"}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        {!loading && !error && items.length === 0 && (
          <div className="mt-3 rounded-xl border border-dashed border-steel-200 bg-steel-800 p-5 text-center text-sm text-steel-400">
            No hay imágenes guardadas en base de datos.
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-steel-200 bg-steel-700 p-4 shadow-glow">
        {!selected ? (
          <div className="rounded-xl border border-dashed border-steel-200 bg-steel-800 p-10 text-center text-sm text-steel-400">
            Selecciona imagen guardada para ver detalle.
          </div>
        ) : (
          <>
            <p className="text-[10px] uppercase tracking-[0.3em] text-accent-500/80">Detalle guardado</p>
            <h3 className="mt-1 text-xl font-semibold text-white truncate">{selected.nombre}</h3>
            <p className="mt-0.5 text-xs text-steel-400">Registro #{selected.id}</p>

            <div className="mt-4 overflow-hidden rounded-xl border border-steel-200 bg-steel-900">
              <img
                src={selected.imagen}
                alt={selected.nombre}
                className="block max-h-[60vh] w-full object-contain"
              />
            </div>

            <div className="mt-3 rounded-xl border border-steel-200 bg-steel-800 p-4">
              <p className="text-[10px] uppercase tracking-[0.2em] text-steel-400">Descripción</p>
              <p className="mt-1.5 text-sm leading-6 text-gray-300">
                {selected.descripcion || "Sin descripción"}
              </p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
