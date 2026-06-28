import { useState } from "react";
import { generateAiDescription } from "../services/apiDetectionService";

const EPP_TYPE_MAP = [
  { key: "helmet", label: "Casco", keywords: ["helmet", "hardhat"] },
  { key: "vest", label: "Chaleco", keywords: ["vest"] },
  { key: "gloves", label: "Guantes", keywords: ["gloves"] },
  { key: "boots", label: "Botas", keywords: ["boots"] },
  { key: "glasses", label: "Lentes", keywords: ["glasses", "goggles"] },
  { key: "mask", label: "Máscara", keywords: ["mask"] },
];

function groupByEppType(detections, personCount, modelClasses, requiredEppFilter = null) {
  const activeTypes = modelClasses.length > 0
    ? EPP_TYPE_MAP.filter((type) =>
        type.keywords.some((kw) =>
          modelClasses.some((cls) => cls.toLowerCase().includes(kw))
        )
      )
    : EPP_TYPE_MAP;

  // When zone config is present, only show EPP types that zone requires
  const visibleTypes = requiredEppFilter
    ? activeTypes.filter((type) =>
        requiredEppFilter.some((req) =>
          type.keywords.some((kw) =>
            kw.includes(req.toLowerCase()) || req.toLowerCase().includes(kw)
          )
        )
      )
    : activeTypes;

  return visibleTypes.map((type) => {
    const matches = detections.filter((d) =>
      type.keywords.some((kw) => d.label.toLowerCase().includes(kw))
    );
    const compliant = matches.filter((d) => d.helmetDetected).length;
    const missing = personCount > 0
      ? Math.max(0, personCount - compliant)
      : matches.filter((d) => !d.helmetDetected).length;
    const total = personCount > 0 ? personCount : matches.length;
    return { ...type, total, compliant, missing };
  }).filter(Boolean);
}

const RESULT_STYLES = {
  con: { badge: "border-ok-500/30 bg-ok-500/15 text-ok-200", dot: "bg-ok-400" },
  sin: { badge: "border-warn-500/30 bg-warn-500/15 text-warn-200", dot: "bg-warn-400" },
  mixto: { badge: "border-accent-500/30 bg-accent-500/15 text-accent-200", dot: "bg-accent-400" },
};

const resultKey = (result = "") => {
  if (result.startsWith("con")) return "con";
  if (result.startsWith("sin")) return "sin";
  return "mixto";
};

export default function DetectionReview({
  entry,
  currentIndex,
  totalCount,
  formatTimestamp,
  onPrev,
  onNext,
  onSave,
  onDelete,
  onBack,
  eppModelClasses = [],
}) {
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saveDescription, setSaveDescription] = useState("");
  const [saveError, setSaveError] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  if (!entry) return null;

  const rKey = resultKey(entry.result);
  const style = RESULT_STYLES[rKey] ?? RESULT_STYLES.mixto;
  const isCameraFrame = entry.modelName === "live-cam";
  const isEppEntry = entry.modelName?.startsWith("epp_models") ?? false;
  const noHelmetPersons = entry.detections.filter((d) => !d.helmetDetected);
  const helmetPersons = entry.detections.filter((d) => d.helmetDetected);
  const personCount = entry.personCount ?? 0;

  // Build required EPP filter from zone config — show only EPP types the zones require
  const requiredEppForAlert = (() => {
    const zonesConfig = entry.zonesConfig || [];
    const defaultZoneEpp = entry.defaultZoneEpp || [];

    const required = new Set();
    zonesConfig.forEach((z) => {
      if (z.active !== false && z.requiredEpp?.length > 0) {
        z.requiredEpp.forEach((e) => required.add(e));
      }
    });
    defaultZoneEpp.forEach((e) => required.add(e));
    return required.size > 0 ? [...required] : null;
  })();

  const eppGroups = isEppEntry ? groupByEppType(entry.detections, personCount, eppModelClasses, requiredEppForAlert) : [];
  const nonCompliantGroups = eppGroups.filter((g) => g.missing > 0);

  const openSaveModal = () => {
    const defaultName = entry.result === "sin casco" ? "Operario sin casco" : entry.name;
    setSaveName(defaultName || "");
    setSaveDescription("");
    setSaveError("");
    setIsSaveModalOpen(true);
  };

  const closeSaveModal = () => {
    if (isSaving) return;
    setIsSaveModalOpen(false);
    setSaveDescription("");
    setSaveError("");
  };

  const handleGenerateDescription = async () => {
    setIsGenerating(true);
    try {
      const imageUrl = entry.annotatedPreviewUrl || entry.previewUrl;
      const desc = await generateAiDescription({
        imageDataUrl: imageUrl,
        detections: entry.detections || [],
        personCount: entry.personCount ?? 0,
        result: entry.result || "",
        alertingZones: entry.alertingZones || null,
        defaultZoneResult: entry.defaultZoneResult || null,
      });
      setSaveDescription(desc);
    } catch (err) {
      setSaveError(err.message || "No se pudo generar descripción con IA");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleConfirmSave = async () => {
    const nombre = saveName.trim();
    if (!nombre) {
      setSaveError("Nombre obligatorio para guardar.");
      return;
    }

    setSaveError("");
    setIsSaving(true);
    try {
      await onSave(entry, { nombre, descripcion: saveDescription });
      setIsSaveModalOpen(false);
      setSaveDescription("");
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "No se pudo guardar detección");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="animate-fadeUp flex flex-col gap-6">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="inline-flex items-center gap-2 rounded-2xl border border-[#2A2A2E] bg-[#1C1C1F] px-4 py-2 text-sm font-medium text-[#C0C7D4] transition hover:border-accent-400/30 hover:bg-[#222226] hover:text-white"
          >
            ← Volver
          </button>
          <span className="text-sm text-steel-500">
            {isCameraFrame ? "Frame de cámara en vivo" : "Imagen procesada"}
          </span>
        </div>

        {/* Carousel navigation */}
        {totalCount > 1 && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onPrev}
              disabled={currentIndex === 0}
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-[#2A2A2E] bg-[#1C1C1F] text-[#C0C7D4] transition hover:border-accent-400/30 hover:bg-[#222226] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
              aria-label="Detección anterior"
            >
              ‹
            </button>
            <span className="min-w-[4rem] text-center text-xs text-steel-400">
              {currentIndex + 1} / {totalCount}
            </span>
            <button
              type="button"
              onClick={onNext}
              disabled={currentIndex === totalCount - 1}
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-[#2A2A2E] bg-[#1C1C1F] text-[#C0C7D4] transition hover:border-accent-400/30 hover:bg-[#222226] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
              aria-label="Siguiente detección"
            >
              ›
            </button>
          </div>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_380px]">
        {/* Left — image + summary */}
        <div className="flex flex-col gap-4">
          <section className="rounded-[2rem] border border-[#2A2A2E] bg-[#161618] p-5 shadow-glow">
            <div className="flex flex-col gap-1">
              <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Detalle</p>
              <h2 className="text-2xl font-semibold text-white truncate">{entry.name}</h2>
              <p className="text-sm text-steel-400">{formatTimestamp(entry.timestamp)}</p>
              {entry.alertingZones?.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {entry.alertingZones.map((zr) => (
                    <span key={zr.zoneId} className="rounded-full border border-warn-500/30 bg-warn-500/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] text-warn-200">
                      {zr.label || zr.zoneId}
                    </span>
                  ))}
                </div>
              )}
              {(!entry.alertingZones?.length) && entry.defaultZoneResult && !entry.defaultZoneResult.compliant && (
                <span className="mt-1 self-start rounded-full border border-warn-500/30 bg-warn-500/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] text-warn-200">
                  Zona por defecto
                </span>
              )}
            </div>

            {/* Annotated image */}
            <div className="mt-4 overflow-hidden rounded-[1.5rem] border border-[#2A2A2E] bg-steel-950">
              <img
                src={entry.annotatedPreviewUrl || entry.previewUrl}
                alt="Frame detectado"
                className="block h-auto w-full object-contain"
                style={{ maxHeight: "480px" }}
              />
            </div>

            {/* Quick stats row */}
            {isEppEntry ? null : (
              <div className="mt-4 grid grid-cols-3 gap-3 text-center">
                <div className="rounded-2xl border border-[#2A2A2E] bg-steel-900/70 px-3 py-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-steel-400">Personas</p>
                  <p className="mt-1 text-xl font-semibold text-white">{entry.detections.length}</p>
                </div>
                <div className="rounded-2xl border border-ok-500/20 bg-ok-500/10 px-3 py-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-ok-300">Con casco</p>
                  <p className="mt-1 text-xl font-semibold text-ok-300">{helmetPersons.length}</p>
                </div>
                <div className="rounded-2xl border border-warn-500/20 bg-warn-500/10 px-3 py-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-warn-300">Sin casco</p>
                  <p className="mt-1 text-xl font-semibold text-warn-300">{noHelmetPersons.length}</p>
                </div>
              </div>
            )}

            {/* EPP horizontal breakdown — only for EPP v2 entries */}
            {isEppEntry && (
              <div className="mt-4 flex overflow-hidden rounded-2xl border border-[#2A2A2E]">
                {/* Summary column */}
                <div className="flex w-36 shrink-0 flex-col justify-center gap-1 border-r border-[#2A2A2E] bg-steel-900/60 px-4 py-4">
                  <p className="text-[9px] uppercase tracking-[0.25em] text-steel-500">Detecciones</p>
                  <p className="text-3xl font-bold text-white">{personCount}</p>
                  <p className="mt-2 text-[9px] uppercase leading-4 tracking-[0.2em] text-steel-500">
                    Estado<br />detallado EPP
                  </p>
                </div>
                {/* Scrollable EPP type cards */}
                <div className="flex flex-1 overflow-x-auto">
                  {eppGroups.length === 0 ? (
                    <div className="flex items-center px-5 text-sm text-steel-400">Sin EPP detectado</div>
                  ) : (
                    eppGroups.map((group) => (
                      <div
                        key={group.key}
                        className={`flex shrink-0 min-w-[128px] flex-col justify-between border-r border-[#2A2A2E] px-4 py-4 last:border-r-0 ${
                          group.missing > 0 ? "bg-warn-500/5" : "bg-ok-500/5"
                        }`}
                      >
                        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#C0C7D4]">
                          {group.label}
                        </p>
                        <div className="mt-3 space-y-1.5">
                          <div className="flex items-baseline justify-between gap-2">
                            <span className="text-[10px] text-ok-400">Completos:</span>
                            <span className="text-sm font-bold text-ok-300">{group.compliant}</span>
                          </div>
                          <div className="flex items-baseline justify-between gap-2">
                            <span className="text-[10px] text-warn-400">Faltantes:</span>
                            <span className={`text-sm font-bold ${group.missing > 0 ? "text-warn-200" : "text-steel-500"}`}>
                              {group.missing}
                            </span>
                          </div>
                        </div>
                        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-steel-800">
                          <div
                            className="h-full rounded-full bg-ok-400"
                            style={{ width: `${group.total > 0 ? (group.compliant / group.total) * 100 : 0}%` }}
                          />
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </section>

          {/* AI decision banner */}
          {isEppEntry ? (
            <div
              className={`rounded-[1.75rem] border px-5 py-4 ${
                entry.detections.length === 0
                  ? "border-[#2A2A2E] bg-[#1C1C1F]"
                  : nonCompliantGroups.length > 0
                    ? "border-warn-500/30 bg-warn-500/10"
                    : "border-ok-500/30 bg-ok-500/10"
              }`}
            >
              <div className="flex items-start gap-3">
                <span className="mt-0.5 text-2xl">
                  {entry.detections.length === 0 ? "—" : nonCompliantGroups.length > 0 ? "⚠" : "✓"}
                </span>
                <div className="min-w-0 flex-1">
                  <p
                    className={`text-base font-semibold ${
                      entry.detections.length === 0
                        ? "text-[#C0C7D4]"
                        : nonCompliantGroups.length > 0
                          ? "text-warn-100"
                          : "text-ok-100"
                    }`}
                  >
                    {entry.detections.length === 0
                      ? "Sin EPP detectado en este frame"
                      : nonCompliantGroups.length > 0
                        ? `EPP incompleto: ${nonCompliantGroups.map((g) => `${g.missing} ${g.label.toLowerCase()}`).join(", ")}`
                        : "Todos los EPP presentes y completos"}
                  </p>
                  <p className="mt-0.5 text-xs text-steel-400">
                    {personCount} persona{personCount !== 1 ? "s" : ""} detectada{personCount !== 1 ? "s" : ""} ·{" "}
                    {entry.processingTimeMs > 0 ? `${entry.processingTimeMs}ms` : "Tiempo real"}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div
              className={`rounded-[1.75rem] border px-5 py-4 ${
                noHelmetPersons.length > 0
                  ? "border-warn-500/30 bg-warn-500/10"
                  : entry.detections.length === 0
                    ? "border-[#2A2A2E] bg-[#1C1C1F]"
                    : "border-ok-500/30 bg-ok-500/10"
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="text-2xl">
                  {noHelmetPersons.length > 0 ? "⚠" : entry.detections.length === 0 ? "—" : "✓"}
                </span>
                <div>
                  <p
                    className={`text-base font-semibold ${
                      noHelmetPersons.length > 0
                        ? "text-warn-100"
                        : entry.detections.length === 0
                          ? "text-[#C0C7D4]"
                          : "text-ok-100"
                    }`}
                  >
                    {noHelmetPersons.length > 0
                      ? `${noHelmetPersons.length} persona${noHelmetPersons.length > 1 ? "s" : ""} sin casco detectada${noHelmetPersons.length > 1 ? "s" : ""}`
                      : entry.detections.length === 0
                        ? "Sin personas detectadas en este frame"
                        : "Todas las personas llevan casco"}
                  </p>
                  <p className="text-xs text-steel-400 mt-0.5">
                    {entry.processingTimeMs > 0 ? `${entry.processingTimeMs}ms` : "Tiempo real"}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right — per-person cards (non-EPP only) + action panel */}
        <div className="flex flex-col gap-4">
          {/* Per-person cards — only for original helmet model */}
          {!isEppEntry && (
            <section className="rounded-[2rem] border border-[#2A2A2E] bg-[#161618] p-5 shadow-glow">
              <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Por persona</p>
              <h3 className="mt-1 text-lg font-semibold text-white">Resultados individuales</h3>
              <div className="mt-4 space-y-3">
                {entry.detections.length === 0 && (
                  <p className="text-sm text-steel-400">Sin personas detectadas en este frame.</p>
                )}
                {entry.detections.map((det, idx) => {
                  const isHelmet = Boolean(det.helmetDetected);
                  return (
                    <div
                      key={det.id ?? idx}
                      className={`rounded-2xl border p-4 ${
                        isHelmet ? "border-ok-500/20 bg-ok-500/8" : "border-warn-500/25 bg-warn-500/10"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span
                            className={`inline-flex h-7 w-7 items-center justify-center rounded-xl text-xs font-bold ${
                              isHelmet ? "bg-ok-500/20 text-ok-200" : "bg-warn-500/20 text-warn-200"
                            }`}
                          >
                            {det.personIndex ?? idx + 1}
                          </span>
                          <span className="text-sm font-medium text-white">
                            Persona {det.personIndex ?? idx + 1}
                          </span>
                        </div>
                        <span
                          className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] ${
                            isHelmet
                              ? "border-ok-500/30 bg-ok-500/10 text-ok-200"
                              : "border-warn-500/30 bg-warn-500/10 text-warn-200"
                          }`}
                        >
                          {det.label}
                        </span>
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-steel-400">
                        <div>
                          <span className="text-steel-500">Confianza ViT</span>
                          <p className="mt-0.5 font-semibold text-[#D1D5DB]">
                            {(det.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                        {det.bbox && (
                          <div>
                            <span className="text-steel-500">Bbox (px)</span>
                            <p className="mt-0.5 font-semibold text-[#D1D5DB]">
                              ({det.bbox.x}, {det.bbox.y}) → ({det.bbox.x + det.bbox.width}, {det.bbox.y + det.bbox.height})
                            </p>
                          </div>
                        )}
                      </div>
                      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-steel-800">
                        <div
                          className={`h-full rounded-full transition-all ${isHelmet ? "bg-ok-400" : "bg-warn-400"}`}
                          style={{ width: `${(det.confidence * 100).toFixed(0)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Validacion / action panel */}
          <section className="rounded-[2rem] border border-[#2A2A2E] bg-[#161618] p-5 shadow-glow">
            <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Validación</p>
            <h3 className="mt-1 text-lg font-semibold text-white">Confirmar decisión</h3>
            <p className="mt-2 text-sm leading-6 text-steel-400">
              Revisar la detección y confirmar o eliminar según la necesidad.
            </p>

            <div className="mt-5 flex flex-col gap-3">
              {/* Guardar */}
              <button
                type="button"
                onClick={openSaveModal}
                className="flex w-full items-center justify-center gap-2.5 rounded-2xl bg-gradient-to-r from-ok-600 to-ok-500 px-5 py-3.5 text-sm font-semibold text-white transition hover:brightness-110 active:scale-[0.98]"
              >
                Guardar detección
              </button>

              {/* Eliminar */}
              <button
                type="button"
                onClick={() => onDelete(entry.id)}
                className="flex w-full items-center justify-center gap-2.5 rounded-2xl border border-warn-500/30 bg-warn-500/10 px-5 py-3.5 text-sm font-semibold text-warn-200 transition hover:bg-warn-500/20 active:scale-[0.98]"
              >
                Eliminar
              </button>
            </div>

            <p className="mt-4 text-center text-xs text-steel-500">
              Guardar: registra infracción · Eliminar: imagenes sin valor o por error del modelo
            </p>
          </section>
        </div>
      </div>

      {isSaveModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-steel-950/70 px-4">
          <div className="w-full max-w-md rounded-[1.5rem] border border-[#2E2E33] bg-steel-900/95 p-5 shadow-glow">
            <p className="text-xs uppercase tracking-[0.28em] text-accent-300/80">Guardar detección</p>
            <h4 className="mt-2 text-lg font-semibold text-white">Nombre de registro</h4>
            <p className="mt-1 text-sm text-steel-400">Escribe etiqueta para imagen guardada.</p>

            <div className="mt-4">
              <label htmlFor="save-name" className="text-xs uppercase tracking-[0.2em] text-steel-400">
                Nombre
              </label>
              <input
                id="save-name"
                type="text"
                value={saveName}
                onChange={(event) => setSaveName(event.target.value)}
                disabled={isSaving}
                className="mt-2 w-full rounded-xl border border-[#2E2E33] bg-steel-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-accent-400/70"
                placeholder="Ej: Operario sin casco"
              />
            </div>

            <div className="mt-4">
              <div className="flex items-center justify-between gap-2">
                <label htmlFor="save-description" className="text-xs uppercase tracking-[0.2em] text-steel-400">
                  Descripción (opcional)
                </label>
                <button
                  type="button"
                  onClick={handleGenerateDescription}
                  disabled={isSaving || isGenerating}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-accent-400/30 bg-accent-500/10 px-2.5 py-1 text-xs font-medium text-accent-200 transition hover:bg-accent-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isGenerating ? (
                    <>
                      <span className="inline-block h-3 w-3 animate-spin rounded-full border border-accent-400/40 border-t-accent-300" />
                      Generando…
                    </>
                  ) : (
                    "✦ Generar con IA"
                  )}
                </button>
              </div>
              <textarea
                id="save-description"
                value={saveDescription}
                onChange={(event) => setSaveDescription(event.target.value)}
                disabled={isSaving || isGenerating}
                className="mt-2 min-h-[96px] w-full resize-y rounded-xl border border-[#2E2E33] bg-steel-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-accent-400/70"
                placeholder="Ej: Operario ingresó sin casco al área de carga"
              />
              {saveError && <p className="mt-2 text-xs text-warn-300">{saveError}</p>}
            </div>

            <div className="mt-5 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={closeSaveModal}
                disabled={isSaving}
                className="rounded-xl border border-[#2E2E33] bg-[#1C1C1F] px-4 py-2 text-sm font-semibold text-[#D1D5DB] transition hover:bg-[#222226] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={handleConfirmSave}
                disabled={isSaving}
                className="rounded-xl bg-gradient-to-r from-ok-600 to-ok-500 px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isSaving ? "Guardando..." : "Guardar"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
