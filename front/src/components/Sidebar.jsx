const menuItems = [
  { id: "upload-section", label: "Cargar imágenes", hint: "Select" },
  { id: "history-panel", label: "Historial de detecciones", hint: "Log" },
];

export default function Sidebar({ collapsed, onToggle, onNavigate }) {
  return (
    <aside
      className={`sticky top-0 flex h-screen flex-col border-r border-white/6 bg-steel-950/90 backdrop-blur-xl transition-all duration-300 ${
        collapsed ? "w-20" : "w-72"
      }`}
    >
      <div className="flex items-center justify-between gap-3 border-b border-white/6 px-4 py-5">
        <button
          type="button"
          className="flex items-center gap-3 text-left"
          onClick={() => onNavigate("upload-section")}
        >
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-400 to-ok-400 text-lg font-black text-steel-950 shadow-lg shadow-accent-500/20">
            H
          </div>
          {!collapsed && (
            <div>
              <p className="text-sm font-semibold text-white">Helmet Vision</p>
              <p className="text-xs text-steel-400">Image Dashboard</p>
            </div>
          )}
        </button>

        <button
          type="button"
          onClick={onToggle}
          className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/8 bg-white/5 text-steel-200 transition hover:border-accent-400/40 hover:bg-white/10"
          aria-label={collapsed ? "Expandir sidebar" : "Colapsar sidebar"}
        >
          {collapsed ? "→" : "←"}
        </button>
      </div>

      <div className="flex-1 px-3 py-4">
        <div className="mb-4 rounded-3xl border border-accent-500/15 bg-accent-500/8 p-4 text-sm text-steel-200">
          {!collapsed ? (
            <>
              <p className="font-medium text-white">Sistema estático</p>
              <p className="mt-1 text-sm leading-6 text-steel-300">
                Diseñado para trabajar con imágenes cargadas manualmente, sin cámara en vivo.
              </p>
            </>
          ) : (
            <p className="text-center text-xs leading-5 text-steel-300">Static</p>
          )}
        </div>

        <nav className="space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.id)}
              className="group flex w-full items-center gap-3 rounded-2xl border border-transparent px-3 py-3 text-left transition hover:border-white/8 hover:bg-white/5"
            >
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-steel-900 text-xs font-semibold text-accent-200 ring-1 ring-white/6 transition group-hover:bg-accent-500/12 group-hover:text-white">
                {item.hint}
              </span>
              {!collapsed && (
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-white">{item.label}</p>
                  <p className="text-xs text-steel-400">Navegación rápida</p>
                </div>
              )}
            </button>
          ))}
        </nav>
      </div>

      <div className="border-t border-white/6 p-4">
        <div className="rounded-3xl border border-white/8 bg-white/5 p-4">
          {!collapsed ? (
            <>
              <p className="text-sm font-medium text-white">Arquitectura futura</p>
              <p className="mt-1 text-xs leading-5 text-steel-400">
                La base queda lista para conectar un backend de inferencia o streaming en el futuro.
              </p>
            </>
          ) : (
            <p className="text-center text-xs text-steel-400">Future-ready</p>
          )}
        </div>
      </div>
    </aside>
  );
}