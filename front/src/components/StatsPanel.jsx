export default function StatsPanel({ history }) {
  const totalDetections = history.reduce((sum, item) => sum + item.detections.length, 0);
  const helmets = history.reduce(
    (sum, item) => sum + item.detections.filter((d) => d.helmetDetected).length,
    0,
  );
  const noHelmets = totalDetections - helmets;
  const compliance = totalDetections > 0 ? Math.round((helmets / totalDetections) * 100) : 0;

  const complianceColor =
    compliance >= 80
      ? { bar: "bg-ok-400", text: "text-ok-300", border: "border-ok-500/20", bg: "bg-ok-500/10" }
      : compliance >= 50
        ? { bar: "bg-accent-400", text: "text-accent-300", border: "border-accent-500/20", bg: "bg-accent-500/10" }
        : { bar: "bg-warn-400", text: "text-warn-300", border: "border-warn-500/20", bg: "bg-warn-500/10" };

  return (
    <section id="stats-section" className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Estadísticas</p>
        <h2 className="mt-1 text-2xl font-semibold text-white">Métricas de cumplimiento</h2>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-[1.5rem] border border-white/8 bg-steel-900/70 p-4 text-center">
          <p className="text-xs uppercase tracking-[0.25em] text-steel-400">Total</p>
          <p className="mt-2 text-3xl font-semibold text-white">{totalDetections}</p>
          <p className="mt-1 text-xs text-steel-400">detecciones</p>
        </div>

        <div className="rounded-[1.5rem] border border-ok-500/20 bg-ok-500/10 p-4 text-center">
          <p className="text-xs uppercase tracking-[0.25em] text-ok-300">Con casco</p>
          <p className="mt-2 text-3xl font-semibold text-ok-300">{helmets}</p>
          <p className="mt-1 text-xs text-ok-400">personas</p>
        </div>

        <div className="rounded-[1.5rem] border border-warn-500/20 bg-warn-500/10 p-4 text-center">
          <p className="text-xs uppercase tracking-[0.25em] text-warn-300">Sin casco</p>
          <p className="mt-2 text-3xl font-semibold text-warn-300">{noHelmets}</p>
          <p className="mt-1 text-xs text-warn-400">personas</p>
        </div>

        <div className={`rounded-[1.5rem] border p-4 text-center ${complianceColor.border} ${complianceColor.bg}`}>
          <p className={`text-xs uppercase tracking-[0.25em] ${complianceColor.text}`}>Cumplimiento</p>
          <p className={`mt-2 text-3xl font-semibold ${complianceColor.text}`}>{compliance}%</p>
          <p className={`mt-1 text-xs ${complianceColor.text} opacity-75`}>uso de casco</p>
        </div>
      </div>

      {totalDetections > 0 ? (
        <div className="mt-4 rounded-[1.5rem] border border-white/8 bg-steel-900/50 p-4">
          <div className="mb-2 flex justify-between text-xs text-steel-400">
            <span>Cumplimiento EPP</span>
            <span>
              {helmets} / {totalDetections} personas
            </span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-steel-800">
            <div
              className={`h-full rounded-full transition-all duration-500 ${complianceColor.bar}`}
              style={{ width: `${compliance}%` }}
            />
          </div>
        </div>
      ) : (
        <p className="mt-5 text-center text-sm text-steel-500">
        </p>
      )}
    </section>
  );
}
