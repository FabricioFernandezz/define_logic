const menuItems = [
  { id: "upload-section", label: "Cargar imágenes", hint: "IMG" },
  { id: "history-panel", label: "Historial de detecciones", hint: "LOG" },
];

export default function Sidebar({ collapsed, onToggle, onNavigate }) {
  return (
    <aside
      style={{ width: collapsed ? "70px" : "240px", transition: "width 300ms ease" }}
      className="flex h-full shrink-0 flex-col border-r border-white/6 bg-steel-950/90 backdrop-blur-xl overflow-hidden"
    >
      <div className="flex items-center justify-between gap-2 border-b border-white/6 px-3 py-5">
        <button
          type="button"
          className="flex items-center gap-2 text-left min-w-0"
          onClick={() => onNavigate("upload-section")}
        >
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-400 to-ok-400 text-sm font-black text-steel-950 shadow-lg shadow-accent-500/20">
            DL
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-white">DefineLogic</p>
              <p className="truncate text-xs text-steel-400">Prevencion de accidentes laborales</p>
            </div>
          )}
        </button>

        <button
          type="button"
          onClick={onToggle}
          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/8 bg-white/5 text-steel-200 transition hover:border-accent-400/40 hover:bg-white/10"
          aria-label={collapsed ? "Expandir sidebar" : "Colapsar sidebar"}
        >
          {collapsed ? "→" : "←"}
        </button>
      </div>

      <div className="flex-1 px-2 py-4">
        <nav className="space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.id)}
              className="group flex w-full items-center gap-3 rounded-2xl border border-transparent px-2 py-3 text-left transition hover:border-white/8 hover:bg-white/5"
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-steel-900 text-[10px] font-semibold text-accent-200 ring-1 ring-white/6 transition group-hover:bg-accent-500/12 group-hover:text-white">
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
    </aside>
  );
}
