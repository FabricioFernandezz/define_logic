const menuItems = [
  { id: "epp-image", label: "Carga de imagen", hint: "I", sub: "EPP en imagen" },
  { id: "epp-live", label: "Cámara en vivo", hint: "TR", sub: "EPP en tiempo real" },
  { id: "epp-history", label: "Actividad registrada", hint: "A", sub: "Detecciones EPP registradas" },
  { id: "saved", label: "Imágenes guardadas", hint: "DB", sub: "Registros en base de datos" },
];

export default function Sidebar({ collapsed, onToggle, activeView, onNavigate }) {
  return (
    <aside
      style={{ width: collapsed ? "70px" : "240px", transition: "width 300ms ease" }}
      className="flex h-full shrink-0 flex-col border-r border-white/6 bg-steel-950/90 backdrop-blur-xl overflow-hidden"
    >
      <div className="border-b border-white/6 px-3 py-4">
        {collapsed ? (
          <div className="flex justify-center">
            <button
              type="button"
              onClick={onToggle}
              className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/8 bg-white/5 text-steel-200 transition hover:border-accent-400/40 hover:bg-white/10"
              aria-label="Expandir sidebar"
            >
              →
            </button>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              className="flex items-center gap-2 text-left min-w-0"
              onClick={() => onNavigate("epp-image")}
            >
              <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-300 to-blue-500 text-sm font-black text-white shadow-lg shadow-blue-500/40">
              DL
              </div>
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-white">DefineLogic</p>
                <p className="truncate text-xs text-steel-400">Prevención de accidentes</p>
              </div>
            </button>

            <button
              type="button"
              onClick={onToggle}
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/8 bg-white/5 text-steel-200 transition hover:border-accent-400/40 hover:bg-white/10"
              aria-label="Colapsar sidebar"
            >
              ←
            </button>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-4">
        <nav className="space-y-2">
          {menuItems.map((item) => {
            const isActive = activeView === item.id || (activeView === "epp-review" && item.id === "epp-history");
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={`group flex w-full items-center gap-3 rounded-2xl border px-2 py-3 text-left transition ${
                  isActive
                    ? "border-accent-400/30 bg-accent-500/10"
                    : "border-transparent hover:border-white/8 hover:bg-white/5"
                }`}
              >
                <span
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-[10px] font-semibold ring-1 transition ${
                    isActive
                      ? "bg-accent-500/20 text-accent-200 ring-accent-400/30"
                      : "bg-steel-900 text-ok-300 ring-white/6 group-hover:bg-ok-500/12 group-hover:text-white"
                  }`}
                >
                  {item.hint}
                </span>
                {!collapsed && (
                  <div className="min-w-0">
                    <p className={`truncate text-sm font-medium ${isActive ? "text-accent-200" : "text-white"}`}>
                      {item.label}
                    </p>
                    <p className="text-xs text-steel-400">{item.sub}</p>
                  </div>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
