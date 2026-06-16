function IconImage() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-[18px] w-[18px]">
      <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" />
    </svg>
  );
}

function IconVideo() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-[18px] w-[18px]">
      <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z" />
    </svg>
  );
}

function IconClock() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-[18px] w-[18px]">
      <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67V7z" />
    </svg>
  );
}

function IconDatabase() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-[18px] w-[18px]">
      <path d="M12 3C7.58 3 4 4.79 4 7v10c0 2.21 3.59 4 8 4s8-1.79 8-4V7c0-2.21-3.58-4-8-4zm6 14c0 .64-2.13 2-6 2s-6-1.36-6-2v-2.23C7.61 15.58 9.72 16 12 16s4.39-.42 6-1.23V17zm0-4.5c0 .64-2.13 2-6 2s-6-1.36-6-2v-2.23C7.61 11.08 9.72 11.5 12 11.5s4.39-.42 6-1.23V12.5zM12 9.5C8.13 9.5 6 8.14 6 7.5S8.13 5.5 12 5.5s6 1.36 6 2S15.87 9.5 12 9.5z" />
    </svg>
  );
}

function IconChevronLeft() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
      <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
    </svg>
  );
}

const menuItems = [
  { id: "epp-image",   label: "Carga de imagen",    sub: "EPP en imagen",          icon: <IconImage /> },
  { id: "epp-live",    label: "Cámara en vivo",      sub: "EPP en tiempo real",     icon: <IconVideo /> },
  { id: "epp-history", label: "Actividad reciente",  sub: "Últimas detecciones",    icon: <IconClock /> },
  { id: "saved",       label: "Imágenes guardadas",  sub: "Registros en base datos", icon: <IconDatabase /> },
];

export default function Sidebar({ collapsed, onToggle, activeView, onNavigate }) {
  return (
    <aside
      style={{ width: collapsed ? "64px" : "228px", transition: "width 280ms cubic-bezier(0.4, 0, 0.2, 1)" }}
      className="flex h-full shrink-0 flex-col bg-steel-700 overflow-hidden"
    >
      {/* Logo */}
      <div className="border-b border-white/10 px-3 py-[14px]">
        {collapsed ? (
          <div className="flex justify-center">
            <button
              type="button"
              onClick={onToggle}
              title="Expandir"
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-accent-500 text-[11px] font-black text-white shadow-md shadow-accent-600/40 transition hover:bg-accent-400"
              aria-label="Expandir sidebar"
            >
              DL
            </button>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              className="flex min-w-0 items-center gap-2.5 text-left"
              onClick={() => onNavigate("epp-image")}
            >
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-500 text-[11px] font-black text-white shadow-md shadow-accent-600/40">
                DL
              </div>
              <div className="min-w-0">
                <p className="truncate text-[13px] font-semibold leading-tight text-white">
                  DefineLogic
                </p>
                <p className="truncate text-[10px] leading-tight text-white/50">
                  Prevención de accidentes
                </p>
              </div>
            </button>
            <button
              type="button"
              onClick={onToggle}
              className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-white/40 transition hover:bg-[#222226] hover:text-white"
              aria-label="Colapsar sidebar"
            >
              <IconChevronLeft />
            </button>
          </div>
        )}
      </div>

      {/* Nav */}
      <div className="flex-1 overflow-y-auto px-2 py-3">
        <nav className="space-y-0.5">
          {menuItems.map((item) => {
            const isActive =
              activeView === item.id ||
              (activeView === "epp-review" && item.id === "epp-history");
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={`group relative flex w-full items-center gap-3 rounded-xl px-2.5 py-2.5 text-left transition-all duration-150 ${
                  collapsed ? "justify-center" : ""
                } ${
                  isActive
                    ? "bg-[#252529] text-white"
                    : "text-white/55 hover:bg-[#1E1E22] hover:text-white"
                }`}
              >
                {isActive && (
                  <span className="absolute inset-y-2.5 left-0 w-0.5 rounded-r-full bg-accent-400" />
                )}

                <span className={`shrink-0 transition-colors ${
                  isActive ? "text-accent-300" : "text-white/40 group-hover:text-white/70"
                }`}>
                  {item.icon}
                </span>

                {!collapsed && (
                  <div className="min-w-0 flex-1">
                    <p className={`truncate text-[13px] font-medium leading-tight ${
                      isActive ? "text-white" : "text-white/70 group-hover:text-white"
                    }`}>
                      {item.label}
                    </p>
                    <p className="mt-0.5 truncate text-[10px] leading-tight text-white/40">
                      {item.sub}
                    </p>
                  </div>
                )}

                {collapsed && (
                  <span className="pointer-events-none absolute left-full z-50 ml-3 whitespace-nowrap rounded-lg border border-steel-200 bg-steel-700 px-2.5 py-1.5 text-xs font-medium text-white opacity-0 shadow-glow transition-opacity group-hover:opacity-100">
                    {item.label}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer */}
      <div className="border-t border-white/10 px-3 py-3">
        {collapsed ? (
          <div className="flex justify-center">
            <span className="h-1.5 w-1.5 rounded-full bg-ok-400" />
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-ok-400" />
            <p className="text-[10px] text-white/40">Sistema activo</p>
          </div>
        )}
      </div>
    </aside>
  );
}
