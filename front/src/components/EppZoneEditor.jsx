import { useCallback, useEffect, useRef, useState } from "react";

const ZONE_COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#f59e0b",
  "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
];

// Display-only aliases. Stored values stay as the model class names (English).
const EPP_LABELS = {
  harness: "Arnés",
  helmet: "Casco",
  vest: "Chaleco",
  gloves: "Guantes",
  goggles: "Gafas",
  boots: "Botas",
};
const eppLabel = (cls) => EPP_LABELS[cls?.toLowerCase()] ?? cls;

const createId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `zone-${Date.now()}-${Math.random().toString(16).slice(2)}`;

function EppSelector({ selected, eppClasses, onToggle, emptyHint }) {
  if (eppClasses.length === 0) {
    return <p className="mt-2 text-xs text-steel-500 italic">Cargando clases del modelo…</p>;
  }
  return (
    <>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {eppClasses.map((cls) => {
          const active = selected.includes(cls);
          return (
            <button
              key={cls}
              type="button"
              onClick={() => onToggle(cls)}
              className={`rounded-xl border px-2.5 py-1 text-xs font-medium transition ${
                active
                  ? "border-ok-500/50 bg-ok-500/20 text-ok-200"
                  : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-400 hover:border-white/20 hover:text-white"
              }`}
            >
              {eppLabel(cls)}
            </button>
          );
        })}
      </div>
      {selected.length === 0 && (
        <p className="mt-1.5 text-[10px] text-steel-500">{emptyHint}</p>
      )}
      {selected.length > 0 && (
        <p className="mt-1.5 text-[10px] text-ok-400">Requiere: {selected.map(eppLabel).join(", ")}</p>
      )}
    </>
  );
}

export default function EppZoneEditor({
  frameUrl,
  zones,
  onZonesChange,
  eppClasses,
  defaultZoneEpp,
  onDefaultZoneEppChange,
  defaultZoneActive,
  onDefaultZoneActiveChange,
  defaultZoneRequirePerson,
  onDefaultZoneRequirePersonChange,
}) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [drawCurrent, setDrawCurrent] = useState(null);

  const getColor = (idx) => ZONE_COLORS[idx % ZONE_COLORS.length];

  // Returns the pixel rect of the actual image content within the canvas,
  // accounting for objectFit:contain letterboxing caused by maxHeight capping.
  const getImgRect = (W, H) => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) return { x: 0, y: 0, w: W, h: H };
    const nat = img.naturalWidth / img.naturalHeight;
    const con = W / H;
    if (nat > con) {
      const h = W / nat;
      return { x: 0, y: (H - h) / 2, w: W, h };
    }
    const w = H * nat;
    return { x: (W - w) / 2, y: 0, w, h: H };
  };

  // Normalize mouse event to [0,1] relative to the actual image content rect.
  const getNorm = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    const ir = getImgRect(rect.width, rect.height);
    return {
      x: Math.max(0, Math.min(1, (relX - ir.x) / ir.w)),
      y: Math.max(0, Math.min(1, (relY - ir.y) / ir.h)),
    };
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height } = canvas.getBoundingClientRect();
    if (!width || !height) return;
    if (canvas.width !== Math.round(width) || canvas.height !== Math.round(height)) {
      canvas.width = Math.round(width);
      canvas.height = Math.round(height);
    }
    const W = canvas.width;
    const H = canvas.height;
    const ir = getImgRect(W, H);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    zones.forEach((zone, i) => {
      const color = getColor(i);
      const x = ir.x + zone.bbox.x * ir.w;
      const y = ir.y + zone.bbox.y * ir.h;
      const w = zone.bbox.w * ir.w;
      const h = zone.bbox.h * ir.h;
      ctx.fillStyle = color + "30";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const badge = zone.label || `Zona ${i + 1}`;
      ctx.font = "bold 12px sans-serif";
      const tw = ctx.measureText(badge).width;
      ctx.fillStyle = color + "dd";
      ctx.fillRect(x + 4, y + 4, tw + 10, 22);
      ctx.fillStyle = "#fff";
      ctx.fillText(badge, x + 9, y + 20);
    });

    if (drawing && drawStart && drawCurrent) {
      const x = ir.x + Math.min(drawStart.x, drawCurrent.x) * ir.w;
      const y = ir.y + Math.min(drawStart.y, drawCurrent.y) * ir.h;
      const w = Math.abs(drawCurrent.x - drawStart.x) * ir.w;
      const h = Math.abs(drawCurrent.y - drawStart.y) * ir.h;
      ctx.fillStyle = "rgba(255,255,255,0.10)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }, [zones, drawing, drawStart, drawCurrent]);

  useEffect(() => { redraw(); }, [redraw]);

  useEffect(() => {
    const obs = new ResizeObserver(redraw);
    if (canvasRef.current) obs.observe(canvasRef.current);
    return () => obs.disconnect();
  }, [redraw]);

  const onDown = useCallback((e) => {
    e.preventDefault();
    setDrawing(true);
    setDrawStart(getNorm(e));
    setDrawCurrent(getNorm(e));
  }, [getNorm]);

  const onMove = useCallback((e) => {
    if (!drawing) return;
    setDrawCurrent(getNorm(e));
  }, [drawing, getNorm]);

  const onUp = useCallback((e) => {
    if (!drawing || !drawStart) return;
    setDrawing(false);
    const end = getNorm(e);
    const x = Math.min(drawStart.x, end.x);
    const y = Math.min(drawStart.y, end.y);
    const w = Math.abs(end.x - drawStart.x);
    const h = Math.abs(end.y - drawStart.y);
    setDrawStart(null);
    setDrawCurrent(null);
    if (w < 0.02 || h < 0.02) return;
    onZonesChange([
      ...zones,
      { id: createId(), label: `Zona ${zones.length + 1}`, bbox: { x, y, w, h }, requiredEpp: [], active: true, requirePerson: false },
    ]);
  }, [drawing, drawStart, zones, onZonesChange, getNorm]);

  const updateZone = useCallback((id, updates) => {
    onZonesChange(zones.map((z) => (z.id === id ? { ...z, ...updates } : z)));
  }, [zones, onZonesChange]);

  const deleteZone = useCallback((id) => {
    onZonesChange(zones.filter((z) => z.id !== id));
  }, [zones, onZonesChange]);

  const toggleZoneEpp = useCallback((zoneId, cls) => {
    const zone = zones.find((z) => z.id === zoneId);
    if (!zone) return;
    const next = zone.requiredEpp.includes(cls)
      ? zone.requiredEpp.filter((e) => e !== cls)
      : [...zone.requiredEpp, cls];
    updateZone(zoneId, { requiredEpp: next });
  }, [zones, updateZone]);

  const toggleDefaultEpp = useCallback((cls) => {
    const next = defaultZoneEpp.includes(cls)
      ? defaultZoneEpp.filter((e) => e !== cls)
      : [...defaultZoneEpp, cls];
    onDefaultZoneEppChange(next);
  }, [defaultZoneEpp, onDefaultZoneEppChange]);

  return (
    <div className="flex flex-col gap-4">
      {/* Canvas */}
      <div className="relative overflow-hidden rounded-[1.75rem] border border-accent-400/30 bg-steel-950/80">
        {frameUrl ? (
          <img
            ref={imgRef}
            src={frameUrl}
            alt="Frame congelado"
            className="block w-full h-auto select-none"
            style={{ maxHeight: "420px", objectFit: "contain" }}
            draggable={false}
            onLoad={redraw}
          />
        ) : (
          <div className="flex min-h-[260px] items-center justify-center text-steel-500 text-sm">
            Activa la cámara primero
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-crosshair select-none"
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={() => {
            if (drawing) { setDrawing(false); setDrawStart(null); setDrawCurrent(null); }
          }}
          style={{ touchAction: "none" }}
        />
        <div className="absolute left-4 top-4 flex items-center gap-2">
          <span className="rounded-full border border-accent-400/40 bg-steel-950/90 px-3 py-1 text-xs text-accent-200">
            Arrastra para dibujar zona
          </span>
          {zones.length > 0 && (
            <button
              type="button"
              onClick={() => onZonesChange([])}
              className="rounded-full border border-warn-500/30 bg-steel-950/90 px-3 py-1 text-xs text-warn-300 transition hover:bg-warn-500/10"
            >
              Limpiar zonas
            </button>
          )}
        </div>
      </div>

      {/* Zone cards */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {zones.length === 0 && (
          <div className="col-span-full rounded-[1.75rem] border border-dashed border-[#2E2E33] bg-steel-950/50 py-5 text-center text-sm text-steel-500">
            Sin zonas. Dibuja en la imagen para crear una.
          </div>
        )}

        {zones.map((zone, i) => {
          const color = getColor(i);
          const isActive = zone.active !== false;
          return (
            <div
              key={zone.id}
              className={`rounded-[1.5rem] border bg-steel-900/70 p-4 transition ${!isActive ? "opacity-50" : ""}`}
              style={{ borderColor: color + (isActive ? "55" : "28") }}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: isActive ? color : "#6b7280" }} />
                  <input
                    type="text"
                    value={zone.label}
                    onChange={(e) => updateZone(zone.id, { label: e.target.value })}
                    className="w-full min-w-0 bg-transparent text-sm font-medium text-white outline-none border-b border-[#2E2E33] focus:border-white/30 transition"
                    placeholder="Nombre de zona"
                  />
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    onClick={() => updateZone(zone.id, { active: !isActive })}
                    className={`rounded-xl border px-2 py-1 text-xs transition ${
                      isActive
                        ? "border-ok-500/30 bg-ok-500/10 text-ok-300 hover:bg-ok-500/20"
                        : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500 hover:border-white/20 hover:text-white"
                    }`}
                  >
                    {isActive ? "Activa" : "Inactiva"}
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteZone(zone.id)}
                    className="rounded-xl border border-[#2A2A2E] bg-[#1C1C1F] px-2 py-1 text-xs text-steel-400 transition hover:border-warn-500/30 hover:bg-warn-500/10 hover:text-warn-200"
                  >
                    ✕
                  </button>
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => updateZone(zone.id, { requirePerson: !zone.requirePerson })}
                  className={`rounded-xl border px-2 py-1 text-xs transition ${
                    zone.requirePerson
                      ? "border-ok-500/30 bg-ok-500/10 text-ok-300 hover:bg-ok-500/20"
                      : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500 hover:border-white/20 hover:text-white"
                  }`}
                >
                  {zone.requirePerson ? "Persona: Sí" : "Persona: No"}
                </button>
                <p className="text-[10px] text-steel-600">
                  {zone.requirePerson ? "Solo chequea EPP si hay persona en zona" : "Chequea EPP sin importar personas"}
                </p>
              </div>
              <div className="mt-2">
                <p className="text-[10px] uppercase tracking-[0.25em] text-steel-500">EPP requerido en zona</p>
                <EppSelector
                  selected={zone.requiredEpp}
                  eppClasses={eppClasses}
                  onToggle={(cls) => toggleZoneEpp(zone.id, cls)}
                  emptyHint="Sin EPP requerido → solo informativa"
                />
              </div>
            </div>
          );
        })}

        {/* Default zone — always present, not deletable */}
        <div className={`rounded-[1.5rem] border border-white/12 bg-steel-900/40 p-4 sm:col-span-2 xl:col-span-3 transition ${defaultZoneActive === false ? "opacity-50" : ""}`}>
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className={`h-3 w-3 shrink-0 rounded-full ${defaultZoneActive !== false ? "bg-steel-300" : "bg-steel-600"}`} />
              <p className="text-sm font-medium text-white">Zona por defecto</p>
              <span className="ml-1 rounded-full border border-[#2A2A2E] bg-[#1C1C1F] px-2 py-0.5 text-[10px] text-steel-400">
                Resto de la pantalla sin zona definida
              </span>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              {onDefaultZoneRequirePersonChange && (
                <button
                  type="button"
                  onClick={() => onDefaultZoneRequirePersonChange(!defaultZoneRequirePerson)}
                  className={`rounded-xl border px-2 py-1 text-xs transition ${
                    defaultZoneRequirePerson
                      ? "border-ok-500/30 bg-ok-500/10 text-ok-300 hover:bg-ok-500/20"
                      : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500 hover:border-white/20 hover:text-white"
                  }`}
                >
                  {defaultZoneRequirePerson ? "Persona: Sí" : "Persona: No"}
                </button>
              )}
              {onDefaultZoneActiveChange && (
                <button
                  type="button"
                  onClick={() => onDefaultZoneActiveChange(!(defaultZoneActive !== false))}
                  className={`rounded-xl border px-2 py-1 text-xs transition ${
                    defaultZoneActive !== false
                      ? "border-ok-500/30 bg-ok-500/10 text-ok-300 hover:bg-ok-500/20"
                      : "border-[#2A2A2E] bg-[#1C1C1F] text-steel-500 hover:border-white/20 hover:text-white"
                  }`}
                >
                  {defaultZoneActive !== false ? "Activa" : "Inactiva"}
                </button>
              )}
            </div>
          </div>
          <div className="mt-3">
            <p className="text-[10px] uppercase tracking-[0.25em] text-steel-500">
              EPP requerido fuera de zonas dibujadas
            </p>
            <EppSelector
              selected={defaultZoneEpp}
              eppClasses={eppClasses}
              onToggle={toggleDefaultEpp}
              emptyHint="Sin requisitos → no se alerta en área sin zonas"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
