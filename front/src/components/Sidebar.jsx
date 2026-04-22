const menuItems = [
  { id: "image", label: "Cargar imágenes", hint: "IMG", sub: "Modo imagen estática" },
  { id: "live", label: "Cámara en vivo", hint: "CAM", sub: "Detección en tiempo real" },
  { id: "history", label: "Historial", hint: "LOG", sub: "Detecciones registradas" },
];

export default function Sidebar({ collapsed, onToggle, activeView, onNavigate, cameraBackground = false }) {
  return (
    <aside
      style={{ width: collapsed ? "70px" : "240px", transition: "width 300ms ease" }}
      className="flex h-full shrink-0 flex-col border-r border-white/6 bg-steel-950/90 backdrop-blur-xl overflow-hidden"
    >
      <div className="border-b border-white/6 px-3 py-4">
        {collapsed ? (
          /* Collapsed: only toggle button, centered */
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
          /* Expanded: logo + toggle side by side */
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              className="flex items-center gap-2 text-left min-w-0"
              onClick={() => onNavigate("image")}
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-400 to-ok-400 text-sm font-black text-steel-950 shadow-lg shadow-accent-500/20">
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

      <div className="flex-1 px-2 py-4">
        <nav className="space-y-2">
          {menuItems.map((item) => {
            const isActive = activeView === item.id || (activeView === "review" && item.id === "history");
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
                  className={`relative flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-[10px] font-semibold ring-1 transition ${
                    isActive
                      ? "bg-accent-500/20 text-accent-200 ring-accent-400/30"
                      : "bg-steel-900 text-accent-200 ring-white/6 group-hover:bg-accent-500/12 group-hover:text-white"
                  }`}
                >
                  {item.hint}
                  {item.id === "live" && cameraBackground && (
                    <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 animate-pulse rounded-full bg-ok-400 ring-2 ring-steel-950" />
                  )}
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
