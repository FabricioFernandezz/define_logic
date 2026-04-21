export default function StatsPanel({ history }) {
  return (
    <section id="stats-section" className="rounded-[2rem] border border-dashed border-white/10 bg-white/[0.03] p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Estadísticas</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Espacio reservado para métricas</h2>
          <p className="mt-2 text-sm leading-6 text-steel-400">
            Aquí podrás añadir gráficos, tendencias, alertas y KPIs cuando conectes el dashboard con datos reales.
          </p>
        </div>

        <div className="rounded-2xl border border-white/8 bg-steel-950/70 px-4 py-3 text-sm text-steel-300">
          Procesadas: <span className="font-semibold text-white">{history.length}</span>
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        <div className="min-h-40 rounded-[1.5rem] border border-white/8 bg-steel-900/50 p-4">
          <div className="h-full rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-5 text-sm text-steel-400">
            Bloque para gráficos de tendencia.
          </div>
        </div>
        <div className="min-h-40 rounded-[1.5rem] border border-white/8 bg-steel-900/50 p-4">
          <div className="h-full rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-5 text-sm text-steel-400">
            Bloque para métricas por turno o por cámara.
          </div>
        </div>
        <div className="min-h-40 rounded-[1.5rem] border border-white/8 bg-steel-900/50 p-4">
          <div className="h-full rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-5 text-sm text-steel-400">
            Bloque para alertas y distribución de resultados.
          </div>
        </div>
      </div>
    </section>
  );
}