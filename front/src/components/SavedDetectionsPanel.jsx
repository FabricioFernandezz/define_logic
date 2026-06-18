import { useState } from "react";

export default function SavedDetectionsPanel({
  items,
  selectedId,
  onSelect,
  loading = false,
  error = null,
  onRetry,
  onDelete,
}) {
  const selected = items.find((item) => item.id === selectedId) || items[0] || null;
  const [confirmId, setConfirmId] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    if (!selected || !onDelete) return;
    setDeleting(true);
    try {
      await onDelete(selected.id);
      setConfirmId(null);
    } finally {
      setDeleting(false);
    }
  };

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
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[10px] uppercase tracking-[0.3em] text-accent-500/80">Detalle guardado</p>
                <h3 className="mt-1 text-xl font-semibold text-white truncate">{selected.nombre}</h3>
                <p className="mt-0.5 text-xs text-steel-400">Registro #{selected.id}</p>
              </div>

              {onDelete && (
                <div className="shrink-0">
                  {confirmId === selected.id ? (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-steel-400">¿Eliminar?</span>
                      <button
                        type="button"
                        onClick={handleDelete}
                        disabled={deleting}
                        className="rounded-lg bg-warn-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-warn-400 disabled:opacity-60"
                      >
                        {deleting ? "…" : "Sí, eliminar"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmId(null)}
                        disabled={deleting}
                        className="rounded-lg border border-steel-200 px-3 py-1.5 text-xs font-medium text-steel-300 transition hover:text-white"
                      >
                        Cancelar
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setConfirmId(selected.id)}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-warn-500/30 bg-warn-500/10 px-3 py-1.5 text-xs font-semibold text-warn-300 transition hover:bg-warn-500/20"
                    >
                      <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
                        <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" />
                      </svg>
                      Eliminar
                    </button>
                  )}
                </div>
              )}
            </div>

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
