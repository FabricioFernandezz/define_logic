import { Fragment, useCallback, useEffect, useState } from "react";
import DetectionList from "./components/DetectionList";
import DetectionReview from "./components/DetectionReview";
import SavedDetectionsPanel from "./components/SavedDetectionsPanel";
import EppImageDetector from "./components/EppImageDetector";
import EppLiveCamera from "./components/EppLiveCamera";
import AuthScreen from "./components/AuthScreen";
import InvitePanel from "./components/InvitePanel";
import { useAuth } from "./context/AuthContext";
import {
  getSavedDetectionsFromBackend,
  saveDetectionToBackend,
  deleteSavedDetectionFromBackend,
} from "./services/apiDetectionService";

const createId = () => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const formatTimestamp = (value) => {
  const date = new Date(value);
  return new Intl.DateTimeFormat("es-ES", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
};

const buildEppImageEntry = (image, result) => ({
  id: createId(),
  name: image.name,
  file: image.file,
  timestamp: new Date().toISOString(),
  previewUrl: image.previewUrl,
  annotatedPreviewUrl: result.annotatedImage || image.previewUrl,
  naturalWidth: image.naturalWidth,
  naturalHeight: image.naturalHeight,
  result: result.summary.result,
  confidence: result.summary.confidence,
  personCount: result.personCount ?? 0,
  detections: (result.detections || []).map((d, i) => ({
    id: `epp-img-${i}-${d.id}`,
    bbox: {
      x: d.bbox_pixels[0],
      y: d.bbox_pixels[1],
      width: d.bbox_pixels[2] - d.bbox_pixels[0],
      height: d.bbox_pixels[3] - d.bbox_pixels[1],
    },
    helmetDetected: d.isCompliant,
    label: d.label,
    confidence: Number(d.confidence),
    personIndex: i + 1,
  })),
  processingTimeMs: result.processingTimeMs,
  modelName: result.modelName,
});

const buildEppCameraEntry = (frameData) => {
  const dets = frameData.result.detections || [];
  const eppConfidence =
    dets.length > 0 ? dets.reduce((s, d) => s + d.confidence, 0) / dets.length : 0;
  const personConfidence = frameData.result.personConfidence ?? 0;
  const confidence = dets.length > 0 ? eppConfidence : personConfidence;
  const zoneResults = frameData.result.zoneResults || [];
  const alertingZones = zoneResults.filter((zr) => zr.active !== false && !zr.compliant);
  return {
    id: createId(),
    name: `EPP ${new Date(frameData.timestamp).toLocaleTimeString("es-ES")}`,
    file: null,
    timestamp: frameData.timestamp,
    previewUrl: frameData.frameDataUrl,
    annotatedPreviewUrl: frameData.frameDataUrl,
    naturalWidth: frameData.width,
    naturalHeight: frameData.height,
    result: "no cumple",
    confidence: Number(confidence.toFixed(2)),
    personCount: frameData.result.personCount ?? 0,
    detections: dets.map((d, i) => ({
      id: `epp-cam-${i}`,
      bbox: {
        x: d.bbox_pixels[0],
        y: d.bbox_pixels[1],
        width: d.bbox_pixels[2] - d.bbox_pixels[0],
        height: d.bbox_pixels[3] - d.bbox_pixels[1],
      },
      helmetDetected: d.isCompliant,
      label: d.label,
      confidence: Number(d.confidence),
      personIndex: i + 1,
    })),
    processingTimeMs: 0,
    modelName: "epp_models",
    zoneResults,
    alertingZones,
    defaultZoneResult: frameData.result.defaultZoneResult ?? null,
    zonesConfig: frameData.zonesConfig || [],
    defaultZoneEpp: frameData.defaultZoneEpp || [],
  };
};

const VIEW_META = {
  saved: {
    label: "Base de datos",
    title: "Imágenes guardadas",
    subtitle: "Consulta registros persistidos en base de datos. Abre cualquier imagen en un click.",
  },
  "epp-image": {
    label: "Análisis de imagen",
    title: "Detección en imagen",
    subtitle: "Sube una imagen para analizar el cumplimiento de EPP.",
  },
  "epp-live": {
    label: "Tiempo real",
    title: "Detección en vivo",
    subtitle: "Detección continua de EPP desde la cámara en tiempo real.",
  },
  "epp-history": {
    label: "Historial de sesión",
    title: "Actividad reciente",
    subtitle: "Todas las detecciones registradas en esta sesión.",
  },
  invite: {
    label: "Equipo",
    title: "Encargados de la industria",
    subtitle: "Invita por email a los encargados que podrán acceder a esta industria.",
  },
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const NAV_ITEMS = [
  { id: "epp-live",    label: "Cámara en vivo" },
  { id: "epp-image",   label: "Imágenes" },
  { id: "epp-history", label: "Actividad" },
  { id: "saved",       label: "Base de datos" },
];

function TopNav({ activeView, onNavigate, user, onLogout }) {
  const [time, setTime] = useState(() => {
    const d = new Date();
    return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  });

  useEffect(() => {
    const tick = setInterval(() => {
      const d = new Date();
      setTime(`${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`);
    }, 10000);
    return () => clearInterval(tick);
  }, []);

  const navItems =
    user?.rol === "owner"
      ? [...NAV_ITEMS, { id: "invite", label: "Equipo" }]
      : NAV_ITEMS;

  return (
    <header className="flex h-12 shrink-0 items-center justify-between px-4" style={{ background: '#111113', borderBottom: '1px solid #2A2A2E' }}>
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl text-[11px] font-black text-white shadow" style={{ background: '#F97316' }}>
          DL
        </div>
        <span className="text-[13px] font-semibold text-white">DefineLogic</span>
      </div>

      <nav className="flex items-center">
        {navItems.map((item, i) => {
          const isActive =
            activeView === item.id ||
            (activeView === "epp-review" && item.id === "epp-history");
          return (
            <Fragment key={item.id}>
              {i > 0 && (
                <span className="select-none px-1 text-[10px] text-steel-600">|</span>
              )}
              <button
                type="button"
                onClick={() => onNavigate(item.id)}
                style={isActive ? { background: '#F97316' } : {}}
                className={`rounded-full px-3 py-1 text-sm font-medium transition ${
                  isActive
                    ? "text-white"
                    : "text-[#9CA3AF] hover:text-white"
                }`}
              >
                {isActive && item.id === "epp-live" && (
                  <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-white/80" />
                )}
                {item.label}
              </button>
            </Fragment>
          );
        })}
        <span className="ml-5 font-mono text-xs tabular-nums text-steel-500">{time}</span>
      </nav>

      <div className="flex items-center gap-3">
        {user && (
          <div className="hidden text-right sm:block">
            <p className="text-xs font-semibold leading-tight text-white">{user.nombre}</p>
            <p className="text-[10px] leading-tight text-steel-500">
              {user.rol === "owner" ? "Encargado dueño" : "Encargado"}
            </p>
          </div>
        )}
        <button
          type="button"
          onClick={onLogout}
          className="rounded-lg border border-steel-200 px-3 py-1 text-xs font-medium text-steel-300 transition hover:border-accent-500 hover:text-white"
        >
          Salir
        </button>
      </div>
    </header>
  );
}

function ViewHeader({ label, title, subtitle, badge, stats }) {
  return (
    <header className="relative overflow-hidden rounded-2xl border border-steel-200 bg-steel-700 px-6 py-5 shadow-glow">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent-500/60 to-transparent" />

      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          <p className="font-mono text-[10px] font-medium uppercase tracking-[0.45em] text-accent-500/80">
            DefineLogic · {label}
          </p>
          <h1 className="mt-2 font-display text-3xl font-bold tracking-tight text-white sm:text-[2.25rem] sm:leading-tight">
            {title}
          </h1>
          <p className="mt-1.5 max-w-lg text-sm leading-relaxed text-steel-400">{subtitle}</p>
        </div>

        {(stats?.length > 0 || badge) && (
          <div className="flex shrink-0 flex-wrap items-start gap-2 sm:pt-0.5">
            {badge && (
              <div className="inline-flex items-center gap-1.5 rounded-lg border border-accent-500/30 bg-accent-500/10 px-3 py-1.5">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-500" />
                <span className="font-mono text-xs font-medium text-accent-400">{badge}</span>
              </div>
            )}
            {stats?.map((s) => (
              <div
                key={s.label}
                className={`min-w-[58px] rounded-xl border px-3 py-2 text-center ${
                  s.type === "ok"
                    ? "border-ok-500/25 bg-ok-500/10"
                    : s.type === "warn"
                    ? "border-warn-500/25 bg-warn-500/10"
                    : s.type === "accent"
                    ? "border-accent-500/25 bg-accent-500/10"
                    : "border-steel-200 bg-steel-800"
                }`}
              >
                <p className={`font-mono text-xl font-semibold leading-none tabular-nums ${
                  s.type === "ok" ? "text-ok-300" :
                  s.type === "warn" ? "text-warn-300" :
                  s.type === "accent" ? "text-accent-400" :
                  "text-gray-200"
                }`}>
                  {s.value}
                </p>
                <p className="mt-1 text-[9px] uppercase tracking-[0.18em] text-steel-500">
                  {s.label}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </header>
  );
}

function MainApp() {
  const { user, logout } = useAuth();
  const [activeView, setActiveView] = useState("epp-live");
  const [savedDetections, setSavedDetections] = useState([]);
  const [savedDetectionsLoading, setSavedDetectionsLoading] = useState(false);
  const [savedDetectionsError, setSavedDetectionsError] = useState(null);
  const [selectedSavedDetectionId, setSelectedSavedDetectionId] = useState(null);
  const [notification, setNotification] = useState(null);
  const [eppHistory, setEppHistory] = useState([]);
  const [eppCurrentEntryId, setEppCurrentEntryId] = useState(null);
  const [eppModelClasses, setEppModelClasses] = useState([]);

  const refreshSavedDetections = useCallback(async () => {
    setSavedDetectionsLoading(true);
    setSavedDetectionsError(null);
    try {
      const items = await getSavedDetectionsFromBackend();
      setSavedDetections(Array.isArray(items) ? items : []);
    } catch {
      setSavedDetectionsError("No se pudo cargar imágenes guardadas.");
    } finally {
      setSavedDetectionsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSavedDetections();
  }, [refreshSavedDetections]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/epp/classes`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.all) setEppModelClasses(data.all);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (savedDetections.length === 0) {
      setSelectedSavedDetectionId(null);
      return;
    }
    const selectedExists = savedDetections.some((item) => item.id === selectedSavedDetectionId);
    if (!selectedExists) {
      setSelectedSavedDetectionId(savedDetections[0].id);
    }
  }, [savedDetections, selectedSavedDetectionId]);

  const pushNotification = (payload) => {
    setNotification({ id: createId(), ...payload });
  };

  const handleNavigate = (view) => {
    setActiveView(view);
    if (view === "saved") refreshSavedDetections();
  };

  const handleReviewSave = async (entry, payload) => {
    const nombre = payload?.nombre?.trim();
    if (!nombre) throw new Error("Nombre obligatorio para guardar detección.");

    const descripcion = payload?.descripcion?.trim() || null;
    const imagen = entry.annotatedPreviewUrl || entry.previewUrl;
    if (!imagen) throw new Error("No se encontró imagen para guardar.");

    const duplicate = savedDetections.find((item) => item.imagen === imagen);
    if (duplicate) {
      setSelectedSavedDetectionId(duplicate.id);
      pushNotification({
        type: "warning",
        message: "Imagen ya fue guardada previamente.",
        action: "open-saved",
        actionLabel: "Ver imagen",
        savedId: duplicate.id,
      });
      return duplicate;
    }

    try {
      const saved = await saveDetectionToBackend({ nombre, imagen, descripcion });
      setSavedDetections((current) => {
        const exists = current.some((item) => item.id === saved.id);
        if (exists) return current;
        return [saved, ...current];
      });
      setSelectedSavedDetectionId(saved.id);
      pushNotification({
        type: "success",
        message: "Imagen guardada en base de datos.",
        action: "open-saved",
        actionLabel: "Abrir guardadas",
        savedId: saved.id,
      });
      return saved;
    } catch (error) {
      const existing = error?.payload?.detail?.existing;
      if (error?.status === 409 && existing) {
        setSavedDetections((current) => {
          const exists = current.some((item) => item.id === existing.id);
          if (exists) return current;
          return [existing, ...current];
        });
        setSelectedSavedDetectionId(existing.id);
        pushNotification({
          type: "warning",
          message: "Imagen ya fue guardada previamente.",
          action: "open-saved",
          actionLabel: "Ver imagen",
          savedId: existing.id,
        });
        return existing;
      }
      throw new Error(error instanceof Error ? error.message : "No se pudo guardar detección");
    }
  };

  const handleDeleteSavedDetection = async (id) => {
    try {
      await deleteSavedDetectionFromBackend(id);
      setSavedDetections((current) => current.filter((item) => item.id !== id));
      pushNotification({ type: "success", message: "Registro eliminado de base de datos." });
    } catch (error) {
      pushNotification({
        type: "warning",
        message: error instanceof Error ? error.message : "No se pudo eliminar el registro.",
      });
    }
  };

  const handleNotificationAction = () => {
    if (!notification) return;
    if (notification.action === "open-saved") {
      setActiveView("saved");
      if (notification.savedId) setSelectedSavedDetectionId(notification.savedId);
    }
    setNotification(null);
  };

  const handleEppDetection = useCallback(({ image, result }) => {
    const entry = buildEppImageEntry(image, result);
    setEppHistory((current) => [entry, ...current].slice(0, 20));
  }, []);

  const handleEppCameraDetection = useCallback((frameData) => {
    const entry = buildEppCameraEntry(frameData);
    setEppHistory((current) => [entry, ...current].slice(0, 20));
  }, []);

  const handleEppHistorySelect = (entry) => {
    setEppCurrentEntryId(entry.id);
    setActiveView("epp-review");
  };

  const handleEppReviewBack = () => setActiveView("epp-history");

  const handleEppReviewPrev = () => {
    const idx = eppHistory.findIndex((h) => h.id === eppCurrentEntryId);
    if (idx > 0) setEppCurrentEntryId(eppHistory[idx - 1].id);
  };

  const handleEppReviewNext = () => {
    const idx = eppHistory.findIndex((h) => h.id === eppCurrentEntryId);
    if (idx < eppHistory.length - 1) setEppCurrentEntryId(eppHistory[idx + 1].id);
  };

  const handleEppReviewDelete = (id) => {
    setEppHistory((current) => {
      const next = current.filter((item) => item.id !== id);
      if (next.length === 0) {
        setActiveView("epp-history");
      } else {
        const deletedIdx = current.findIndex((item) => item.id === id);
        const nextIdx = Math.min(deletedIdx, next.length - 1);
        setEppCurrentEntryId(next[nextIdx].id);
      }
      return next;
    });
  };

  const isSaved = activeView === "saved";
  const isEppImage = activeView === "epp-image";
  const isEppLive = activeView === "epp-live";
  const isEppHistory = activeView === "epp-history";
  const isEppReview = activeView === "epp-review";
  const isInvite = activeView === "invite";

  const eppCurrentIdx = eppHistory.findIndex((h) => h.id === eppCurrentEntryId);
  const eppCurrentEntry = eppCurrentIdx >= 0 ? eppHistory[eppCurrentIdx] : null;

  const compliantCount = eppHistory.filter(
    (h) => h.result === "cumple" || h.result === "con"
  ).length;
  const nonCompliantCount = eppHistory.filter(
    (h) => h.result === "no cumple" || h.result === "sin"
  ).length;

  return (
    <div className="flex h-screen flex-col overflow-hidden text-white" style={{ background: '#0D0D0E' }}>
      <TopNav activeView={activeView} onNavigate={handleNavigate} user={user} onLogout={logout} />

      <main className="relative flex-1 overflow-y-auto scrollbar-thin" style={{ background: '#0D0D0E' }}>
        <div className="mx-auto flex w-full max-w-screen-2xl flex-col gap-5 px-5 py-5">

          {isSaved && (
            <div className="animate-fadeUp flex flex-col gap-5">
              <ViewHeader
                {...VIEW_META.saved}
                stats={[{ value: savedDetections.length, label: "guardadas", type: "accent" }]}
              />
              <SavedDetectionsPanel
                items={savedDetections}
                selectedId={selectedSavedDetectionId}
                onSelect={setSelectedSavedDetectionId}
                loading={savedDetectionsLoading}
                error={savedDetectionsError}
                onRetry={refreshSavedDetections}
                onDelete={handleDeleteSavedDetection}
              />
            </div>
          )}

          {isEppImage && (
            <div className="animate-fadeUp flex flex-col gap-5">
              <ViewHeader {...VIEW_META["epp-image"]} badge="Modelo activo" />
              <div className="grid gap-5 xl:grid-cols-[minmax(0,1.65fr)_380px]">
                <section className="flex flex-col gap-5">
                  <EppImageDetector onEppDetection={handleEppDetection} />
                </section>
                <aside className="xl:sticky xl:top-5 xl:h-[calc(100vh-5rem)]">
                  <DetectionList
                    items={eppHistory}
                    onSelectItem={handleEppHistorySelect}
                    formatTimestamp={formatTimestamp}
                  />
                </aside>
              </div>
            </div>
          )}

          {/* EPP Live — always mounted so camera persists */}
          <div className={isEppLive ? "animate-fadeUp" : "hidden"}>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1.65fr)_380px]" style={{ minHeight: 'calc(100vh - 4.5rem)' }}>
              <section className="flex flex-col h-full">
                <EppLiveCamera active={true} onEppCameraDetection={handleEppCameraDetection} />
              </section>
              <aside className="xl:sticky xl:top-0 xl:h-[calc(100vh-4.5rem)]">
                <DetectionList
                  items={eppHistory}
                  onSelectItem={handleEppHistorySelect}
                  formatTimestamp={formatTimestamp}
                />
              </aside>
            </div>
          </div>

          {isEppHistory && (
            <div className="animate-fadeUp flex flex-col gap-5">
              <ViewHeader
                {...VIEW_META["epp-history"]}
                stats={
                  eppHistory.length > 0
                    ? [
                        { value: eppHistory.length, label: "total", type: "neutral" },
                        { value: compliantCount, label: "cumplen", type: "ok" },
                        { value: nonCompliantCount, label: "no cumplen", type: "warn" },
                      ]
                    : undefined
                }
              />
              <DetectionList
                items={eppHistory}
                onSelectItem={handleEppHistorySelect}
                formatTimestamp={formatTimestamp}
                fullWidth
              />
            </div>
          )}

          {isInvite && user?.rol === "owner" && (
            <div className="animate-fadeUp flex flex-col gap-5">
              <ViewHeader {...VIEW_META.invite} />
              <InvitePanel />
            </div>
          )}

          {isEppReview && eppCurrentEntry && (
            <div className="animate-fadeUp">
              <DetectionReview
                entry={eppCurrentEntry}
                currentIndex={eppCurrentIdx}
                totalCount={eppHistory.length}
                formatTimestamp={formatTimestamp}
                onPrev={handleEppReviewPrev}
                onNext={handleEppReviewNext}
                onSave={handleReviewSave}
                onDelete={handleEppReviewDelete}
                onBack={handleEppReviewBack}
                eppModelClasses={eppModelClasses}
              />
            </div>
          )}

        </div>
      </main>

      {notification && (
        <div
          key={notification.id}
          className="fixed bottom-5 right-5 z-[60] w-full max-w-[340px] animate-slideInRight overflow-hidden rounded-2xl border border-steel-200 bg-steel-700 shadow-glow"
        >
          <div className={`h-0.5 w-full ${notification.type === "warning" ? "bg-warn-500" : "bg-ok-500"}`} />
          <div className="p-4">
            <div className="flex items-start gap-3">
              {notification.type === "warning" ? (
                <svg className="mt-0.5 h-4 w-4 shrink-0 text-warn-400" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
                </svg>
              ) : (
                <svg className="mt-0.5 h-4 w-4 shrink-0 text-ok-400" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                </svg>
              )}
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-white">{notification.message}</p>
                {notification.actionLabel && (
                  <button
                    type="button"
                    onClick={handleNotificationAction}
                    className="mt-2 text-xs font-semibold uppercase tracking-[0.2em] text-accent-500 transition hover:text-accent-400"
                  >
                    {notification.actionLabel} →
                  </button>
                )}
              </div>
              <button
                type="button"
                onClick={() => setNotification(null)}
                className="rounded-lg p-1 text-steel-400 transition hover:bg-steel-300 hover:text-white"
                aria-label="Cerrar"
              >
                <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div
        className="flex h-screen items-center justify-center text-sm text-steel-400"
        style={{ background: "#0D0D0E" }}
      >
        Cargando…
      </div>
    );
  }

  if (!user) return <AuthScreen />;

  return <MainApp />;
}
