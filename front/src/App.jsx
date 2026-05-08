import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import DetectionList from "./components/DetectionList";
import DetectionReview from "./components/DetectionReview";
import SavedDetectionsPanel from "./components/SavedDetectionsPanel";
import EppImageDetector from "./components/EppImageDetector";
import EppLiveCamera from "./components/EppLiveCamera";
import {
  getSavedDetectionsFromBackend,
  saveDetectionToBackend,
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
  const confidence =
    dets.length > 0 ? dets.reduce((s, d) => s + d.confidence, 0) / dets.length : 0;
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
    modelName: "yolo26_epp",
  };
};

const VIEW_TITLES = {
  saved: "Imágenes guardadas",
  "epp-image": "Detección en imagen",
  "epp-live": "Detección en tiempo real",
  "epp-history": "Actividad reciente",
  "epp-review": "Detalles de detecciónes",
};

const VIEW_SUBTITLES = {
  saved: "Consulta registros persistidos en base de datos y abre imagen guardada en un click.",
  "epp-image": "Modelo que detecta directamente sobre una imagen .",
  "epp-live": "Modelo que detecta en tiempo real desde la cámara existente.",
  "epp-history": "Revisa todas las detecciones registradas.",
  "epp-review": "Detalle de detección EPP. Guarda en base de datos o elimina del historial.",
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeView, setActiveView] = useState("epp-image");
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
      .then((r) => r.ok ? r.json() : null)
      .then((data) => { if (data?.all) setEppModelClasses(data.all); })
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
    setNotification({
      id: createId(),
      ...payload,
    });
  };

  const handleNavigate = (view) => {
    setActiveView(view);
    if (view === "saved") {
      refreshSavedDetections();
    }
  };

  const handleReviewSave = async (entry, payload) => {
    const nombre = payload?.nombre?.trim();
    if (!nombre) {
      throw new Error("Nombre obligatorio para guardar detección.");
    }

    const descripcion = payload?.descripcion?.trim() || null;

    const imagen = entry.annotatedPreviewUrl || entry.previewUrl;
    if (!imagen) {
      throw new Error("No se encontró imagen para guardar.");
    }

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
      const saved = await saveDetectionToBackend({
        nombre,
        imagen,
        descripcion,
      });

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

  const handleNotificationAction = () => {
    if (!notification) return;

    if (notification.action === "open-saved") {
      setActiveView("saved");
      if (notification.savedId) {
        setSelectedSavedDetectionId(notification.savedId);
      }
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

  const handleEppReviewBack = () => {
    setActiveView("epp-history");
  };

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

  const eppCurrentIdx = eppHistory.findIndex((h) => h.id === eppCurrentEntryId);
  const eppCurrentEntry = eppCurrentIdx >= 0 ? eppHistory[eppCurrentIdx] : null;

  return (
    <div className="h-screen overflow-hidden bg-steel-950 text-steel-50">
      <div className="flex h-full">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((v) => !v)}
          activeView={activeView}
          onNavigate={handleNavigate}
        />

        <main className="flex-1 overflow-y-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="mx-auto flex max-w-7xl flex-col gap-6">

            {isSaved && (
              <div className="animate-fadeUp flex flex-col gap-6">
                <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
                  <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                  <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    {VIEW_TITLES.saved}
                  </h1>
                  <p className="mt-2 text-sm leading-6 text-steel-300">{VIEW_SUBTITLES.saved}</p>
                  <div className="mt-3 inline-flex items-center gap-2 rounded-xl border border-white/10 bg-steel-950/70 px-3 py-1.5 text-xs text-steel-300">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-300" />
                    {savedDetections.length} registros guardados
                  </div>
                </header>

                <SavedDetectionsPanel
                  items={savedDetections}
                  selectedId={selectedSavedDetectionId}
                  onSelect={setSelectedSavedDetectionId}
                  loading={savedDetectionsLoading}
                  error={savedDetectionsError}
                  onRetry={refreshSavedDetections}
                />
              </div>
            )}

            {isEppImage && (
              <div className="animate-fadeUp flex flex-col gap-6">
                <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
                  <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                  <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    {VIEW_TITLES["epp-image"]}
                  </h1>
                  <p className="mt-2 text-sm leading-6 text-steel-300">{VIEW_SUBTITLES["epp-image"]}</p>
                  <div className="mt-3 inline-flex items-center gap-2 rounded-xl border border-ok-500/30 bg-ok-500/10 px-3 py-1.5 text-xs text-ok-300">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-ok-400" />
                    Modelo activo
                  </div>
                </header>

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1.65fr)_360px]">
                  <section className="flex flex-col gap-6">
                    <EppImageDetector onEppDetection={handleEppDetection} />
                  </section>

                  <aside className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
                    <DetectionList
                      items={eppHistory}
                      onSelectItem={handleEppHistorySelect}
                      formatTimestamp={formatTimestamp}
                    />
                  </aside>
                </div>
              </div>
            )}

            {/* EPP Live — always mounted so camera persists across navigation */}
            <div className={isEppLive ? "animate-fadeUp flex flex-col gap-6" : "hidden"}>
              <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
                <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                  {VIEW_TITLES["epp-live"]}
                </h1>
                <p className="mt-2 text-sm leading-6 text-steel-300">{VIEW_SUBTITLES["epp-live"]}</p>
                <div className="mt-3 inline-flex items-center gap-2 rounded-xl border border-ok-500/30 bg-ok-500/10 px-3 py-1.5 text-xs text-ok-300">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ok-400" />
                  Modelo activo
                </div>
              </header>

              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.65fr)_360px]">
                <section className="flex flex-col gap-6">
                  <EppLiveCamera
                    active={true}
                    onEppCameraDetection={handleEppCameraDetection}
                  />
                </section>

                <aside className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
                  <DetectionList
                    items={eppHistory}
                    onSelectItem={handleEppHistorySelect}
                    formatTimestamp={formatTimestamp}
                  />
                </aside>
              </div>
            </div>

            {isEppHistory && (
              <div className="animate-fadeUp flex flex-col gap-6">
                <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
                  <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                  <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    {VIEW_TITLES["epp-history"]}
                  </h1>
                  <p className="mt-2 text-sm leading-6 text-steel-300">{VIEW_SUBTITLES["epp-history"]}</p>
                </header>
                <DetectionList
                  items={eppHistory}
                  onSelectItem={handleEppHistorySelect}
                  formatTimestamp={formatTimestamp}
                  fullWidth
                />
              </div>
            )}

            {isEppReview && eppCurrentEntry && (
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
            )}

            {notification && (
              <div className="fixed bottom-5 right-5 z-[60] w-full max-w-sm rounded-2xl border border-white/10 bg-steel-900/95 p-4 shadow-glow backdrop-blur-xl">
                <div className="flex items-start gap-3">
                  <span
                    className={`mt-0.5 inline-flex h-2.5 w-2.5 shrink-0 rounded-full ${
                      notification.type === "warning" ? "bg-warn-400" : "bg-ok-400"
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-white">{notification.message}</p>
                    {notification.actionLabel && (
                      <button
                        type="button"
                        onClick={handleNotificationAction}
                        className="mt-2 text-xs font-semibold uppercase tracking-[0.2em] text-accent-300 transition hover:text-accent-200"
                      >
                        {notification.actionLabel}
                      </button>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => setNotification(null)}
                    className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-xs text-steel-300 transition hover:bg-white/10"
                    aria-label="Cerrar notificación"
                  >
                    ✕
                  </button>
                </div>
              </div>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}
