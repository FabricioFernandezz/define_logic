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
    <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
      <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Base de datos</p>
        <h2 className="mt-1 text-2xl font-semibold text-white">Imágenes guardadas</h2>

        <div className="mt-5 space-y-3 overflow-y-auto pr-1 scrollbar-thin max-h-[62vh]">
          {loading && (
            <div className="mt-4 rounded-3xl border border-dashed border-white/10 bg-steel-950/70 p-6 text-center text-sm text-steel-400">
              Cargando imágenes...
            </div>
          )}

          {!loading && error && (
            <div className="mt-4 rounded-3xl border border-dashed border-warn-500/30 bg-warn-500/10 p-6 text-center">
              <p className="text-sm text-warn-300">{error}</p>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="mt-3 text-xs font-semibold uppercase tracking-[0.2em] text-accent-300 transition hover:text-accent-200"
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
                className={`flex w-full gap-3 rounded-[1.4rem] border p-3 text-left transition ${
                  isActive
                    ? "border-accent-400/30 bg-accent-500/10"
                    : "border-white/8 bg-steel-950/70 hover:border-accent-400/20 hover:bg-steel-900"
                }`}
              >
                <div className="h-14 w-14 shrink-0 overflow-hidden rounded-2xl border border-white/8 bg-steel-900">
                  <img src={item.imagen} alt={item.nombre} className="h-full w-full object-cover" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-white">{item.nombre}</p>
                  <p className="mt-1 text-xs text-steel-400">ID #{item.id}</p>
                  <p className="mt-1 truncate text-xs text-steel-500">
                    {item.descripcion || "Sin descripción"}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        {!loading && !error && items.length === 0 && (
          <div className="mt-4 rounded-3xl border border-dashed border-white/10 bg-steel-950/70 p-6 text-center text-sm text-steel-400">
            No hay imágenes guardadas en base de datos.
          </div>
        )}
      </section>

      <section className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
        {!selected ? (
          <div className="rounded-3xl border border-dashed border-white/10 bg-steel-950/70 p-10 text-center text-sm text-steel-400">
            Selecciona imagen guardada para ver detalle.
          </div>
        ) : (
          <>
            <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Detalle guardado</p>
            <h3 className="mt-1 text-2xl font-semibold text-white truncate">{selected.nombre}</h3>
            <p className="mt-1 text-xs text-steel-400">Registro #{selected.id}</p>

            <div className="mt-4 overflow-hidden rounded-[1.4rem] border border-white/8 bg-steel-950">
              <img
                src={selected.imagen}
                alt={selected.nombre}
                className="block max-h-[66vh] w-full object-contain"
              />
            </div>

            <div className="mt-4 rounded-2xl border border-white/8 bg-steel-950/70 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-steel-400">Descripción</p>
              <p className="mt-2 text-sm leading-6 text-steel-200">
                {selected.descripcion || "Sin descripción"}
              </p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
