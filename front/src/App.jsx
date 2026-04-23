import { useCallback, useEffect, useMemo, useState } from "react";
import Sidebar from "./components/Sidebar";
import ImageUploader from "./components/ImageUploader";
import DetectionViewer from "./components/DetectionViewer";
import DetectionList from "./components/DetectionList";
import DetectionReview from "./components/DetectionReview";
import SavedDetectionsPanel from "./components/SavedDetectionsPanel";
import StatsPanel from "./components/StatsPanel";
import LiveCamera from "./components/LiveCamera";
import { analyzeImage } from "./services/detectionService";
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

const buildHistoryEntry = (image, result) => ({
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
  detections: result.detections,
  processingTimeMs: result.processingTimeMs,
  modelName: result.modelName,
});

const buildCameraHistoryEntry = (frameData) => {
  const boxes = frameData.result.boxes || [];
  const confidence =
    boxes.length > 0
      ? boxes.reduce((sum, b) => sum + b.confidence, 0) / boxes.length
      : 0;

  return {
    id: createId(),
    name: `Frame ${new Date(frameData.timestamp).toLocaleTimeString("es-ES")}`,
    file: null,
    timestamp: frameData.timestamp,
    previewUrl: frameData.frameDataUrl,
    annotatedPreviewUrl: frameData.frameDataUrl,
    naturalWidth: frameData.width,
    naturalHeight: frameData.height,
    result: "sin casco",
    confidence: Number(confidence.toFixed(2)),
    detections: boxes.map((box, i) => ({
      id: `cam-${i}-${box.personId}`,
      bbox: {
        x: box.bbox_pixels[0],
        y: box.bbox_pixels[1],
        width: box.bbox_pixels[2] - box.bbox_pixels[0],
        height: box.bbox_pixels[3] - box.bbox_pixels[1],
      },
      helmetDetected: Boolean(box.helmetDetected),
      label: box.label,
      confidence: Number(box.confidence),
      personIndex: box.personId + 1,
    })),
    processingTimeMs: 0,
    modelName: "live-cam",
  };
};

const VIEW_TITLES = {
  image: "Detección en imágenes",
  live: "Detección en tiempo real",
  history: "Actividad reciente",
  saved: "Imágenes guardadas",
};

const VIEW_SUBTITLES = {
  image: "Cargar una imagen y el modelo procesa el resultado.",
  live: "Activar la cámara para detectar cascos automáticamente.",
  history: "Revisa todas las detecciones registradas. Haz click en una para ver el detalle.",
  saved: "Consulta registros persistidos en base de datos y abre imagen guardada en un click.",
};

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeView, setActiveView] = useState("image");
  const [currentEntryId, setCurrentEntryId] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processError, setProcessError] = useState("");
  const [history, setHistory] = useState([]);
  const [savedDetections, setSavedDetections] = useState([]);
  const [savedDetectionsLoading, setSavedDetectionsLoading] = useState(false);
  const [savedDetectionsError, setSavedDetectionsError] = useState(null);
  const [selectedSavedDetectionId, setSelectedSavedDetectionId] = useState(null);
  const [notification, setNotification] = useState(null);
  const [keepAliveCamera, setKeepAliveCamera] = useState(false);
  const [isCameraRunning, setIsCameraRunning] = useState(false);

  const stats = useMemo(() => {
    const total = history.length;
    const helmet = history.reduce(
      (sum, item) => sum + item.detections.filter((d) => d.helmetDetected).length,
      0,
    );
    const noHelmet = history.reduce(
      (sum, item) => sum + item.detections.filter((d) => !d.helmetDetected).length,
      0,
    );
    return { total, helmet, noHelmet };
  }, [history]);

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

  const handleSelectImage = async (file) => {
    if (!file) return;
    const previewUrl = URL.createObjectURL(file);
    const naturalSize = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = reject;
      image.src = previewUrl;
    });
    setSelectedImage({
      id: createId(),
      file,
      name: file.name,
      previewUrl,
      processedPreviewUrl: null,
      naturalWidth: naturalSize.width,
      naturalHeight: naturalSize.height,
      uploadedAt: new Date().toISOString(),
    });
    setDetections([]);
    setProcessError("");
  };

  const handleProcessImage = async () => {
    if (!selectedImage?.file || isProcessing) return;
    setIsProcessing(true);
    setProcessError("");
    try {
      const result = await analyzeImage(selectedImage.file, selectedImage);
      const nextEntry = buildHistoryEntry(selectedImage, result);
      setDetections(result.detections);
      setSelectedImage((current) => ({
        ...current,
        modelName: result.modelName,
        processingTimeMs: result.processingTimeMs,
        result: result.summary.result,
        confidence: result.summary.confidence,
        detections: result.detections,
        processedPreviewUrl: result.annotatedImage || current.previewUrl,
      }));
      setHistory((current) => [nextEntry, ...current].slice(0, 20));
    } catch (error) {
      setProcessError(
        error instanceof Error ? error.message : "Error inesperado procesando la imagen",
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setDetections([]);
    setProcessError("");
  };

  const handleCameraDetection = (frameData) => {
    const entry = buildCameraHistoryEntry(frameData);
    setHistory((current) => [entry, ...current].slice(0, 20));
  };

  const handleHistorySelect = (entry) => {
    setCurrentEntryId(entry.id);
    setActiveView("review");
  };

  const handleReviewBack = () => {
    setActiveView("history");
  };

  const handleReviewPrev = () => {
    const idx = history.findIndex((h) => h.id === currentEntryId);
    if (idx > 0) setCurrentEntryId(history[idx - 1].id);
  };

  const handleReviewNext = () => {
    const idx = history.findIndex((h) => h.id === currentEntryId);
    if (idx < history.length - 1) setCurrentEntryId(history[idx + 1].id);
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

  const handleReviewDelete = (id) => {
    setHistory((current) => {
      const next = current.filter((item) => item.id !== id);
      if (next.length === 0) {
        setActiveView("history");
      } else {
        const deletedIdx = current.findIndex((item) => item.id === id);
        const nextIdx = Math.min(deletedIdx, next.length - 1);
        setCurrentEntryId(next[nextIdx].id);
      }
      return next;
    });
  };

  const isWorkView = activeView === "image" || activeView === "live";
  const isHistory = activeView === "history";
  const isSaved = activeView === "saved";
  const isReview = activeView === "review";
  const cameraBackground = isCameraRunning && activeView !== "live";


  const currentIdx = history.findIndex((h) => h.id === currentEntryId);
  const currentEntry = currentIdx >= 0 ? history[currentIdx] : null;

  return (
    <div className="h-screen overflow-hidden bg-steel-950 text-steel-50">
      <div className="flex h-full">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((v) => !v)}
          activeView={activeView}
          onNavigate={handleNavigate}
          cameraBackground={cameraBackground}
        />

        <main className="flex-1 overflow-y-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="mx-auto flex max-w-7xl flex-col gap-6">

            {}
            <div className={!isWorkView ? "hidden" : ""}>
              <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl animate-fadeUp">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                    <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                      {VIEW_TITLES[activeView] ?? VIEW_TITLES.image}
                    </h1>
                    <p className="mt-2 max-w-3xl text-sm leading-6 text-steel-300">
                      {VIEW_SUBTITLES[activeView] ?? VIEW_SUBTITLES.image}
                    </p>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center sm:min-w-[360px]">
                    <div className="rounded-2xl border border-white/8 bg-steel-900/70 px-4 py-3">
                      <p className="text-xs uppercase tracking-[0.25em] text-steel-400">Procesadas</p>
                      <p className="mt-2 text-2xl font-semibold text-white">{stats.total}</p>
                    </div>
                    <div className="rounded-2xl border border-ok-500/20 bg-ok-500/10 px-4 py-3">
                      <p className="text-xs uppercase tracking-[0.25em] text-ok-200">Con casco</p>
                      <p className="mt-2 text-2xl font-semibold text-ok-300">{stats.helmet}</p>
                    </div>
                    <div className="rounded-2xl border border-warn-500/20 bg-warn-500/10 px-4 py-3">
                      <p className="text-xs uppercase tracking-[0.25em] text-warn-200">Sin casco</p>
                      <p className="mt-2 text-2xl font-semibold text-warn-300">{stats.noHelmet}</p>
                    </div>
                  </div>
                </div>
              </header>

              <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.65fr)_360px]">
                <section className="flex flex-col gap-6">
                
                  <div className={activeView !== "image" ? "hidden" : ""}>
                    <div className="flex flex-col gap-6">
                      <ImageUploader
                        image={selectedImage}
                        onSelectImage={handleSelectImage}
                        onClearImage={handleClearImage}
                      />
                      <DetectionViewer
                        image={selectedImage}
                        detections={detections}
                        isProcessing={isProcessing}
                        processError={processError}
                        onProcess={handleProcessImage}
                        onNavigateHistory={() => handleNavigate("history")}
                      />
                    </div>
                  </div>

                  <div className={activeView !== "live" ? "hidden" : ""}>
                    <LiveCamera
                      active={activeView === "live"}
                      keepAlive={keepAliveCamera}
                      onKeepAliveChange={setKeepAliveCamera}
                      onIsActiveChange={setIsCameraRunning}
                      onCameraDetection={handleCameraDetection}
                    />
                  </div>

                  <StatsPanel history={history} />
                </section>

                <aside className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
                  <DetectionList
                    items={history}
                    onSelectItem={handleHistorySelect}
                    formatTimestamp={formatTimestamp}
                  />
                </aside>
              </div>
            </div>

            
            {isHistory && (
              <div className="animate-fadeUp flex flex-col gap-6">
                <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
                  <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                  <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    {VIEW_TITLES.history}
                  </h1>
                  <p className="mt-2 text-sm leading-6 text-steel-300">{VIEW_SUBTITLES.history}</p>
                  {cameraBackground && (
                    <div className="mt-3 inline-flex items-center gap-2 rounded-xl border border-ok-500/30 bg-ok-500/10 px-3 py-1.5 text-xs text-ok-300">
                      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ok-400" />
                      Cámara activa en segundo plano · detectando
                    </div>
                  )}
                </header>
                <DetectionList
                  items={history}
                  onSelectItem={handleHistorySelect}
                  formatTimestamp={formatTimestamp}
                  fullWidth
                />
              </div>
            )}

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

            {isReview && currentEntry && (
              <DetectionReview
                entry={currentEntry}
                currentIndex={currentIdx}
                totalCount={history.length}
                formatTimestamp={formatTimestamp}
                onPrev={handleReviewPrev}
                onNext={handleReviewNext}
                onSave={handleReviewSave}
                onDelete={handleReviewDelete}
                onBack={handleReviewBack}
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
