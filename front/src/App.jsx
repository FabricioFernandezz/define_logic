import { useMemo, useState } from "react";
import Sidebar from "./components/Sidebar";
import ImageUploader from "./components/ImageUploader";
import DetectionViewer from "./components/DetectionViewer";
import DetectionList from "./components/DetectionList";
import DetectionReview from "./components/DetectionReview";
import StatsPanel from "./components/StatsPanel";
import LiveCamera from "./components/LiveCamera";
import { analyzeImage } from "./services/detectionService";

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
  image: "Detección de casco en imágenes",
  live: "Detección en tiempo real",
  history: "Historial de detecciones",
};

const VIEW_SUBTITLES = {
  image: "Carga una imagen, procesa el resultado del modelo y revisa las últimas detecciones.",
  live: "Activa la cámara para detectar automáticamente personas sin casco cada segundo.",
  history: "Revisa todas las detecciones registradas. Haz click en una para ver el detalle.",
};

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeView, setActiveView] = useState("image");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedImage, setSelectedImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processError, setProcessError] = useState("");
  const [history, setHistory] = useState([]);
  const [savedDetections, setSavedDetections] = useState([]);

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

  const handleNavigate = (view) => {
    setActiveView(view);
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
    const index = history.findIndex((h) => h.id === entry.id);
    setCurrentIndex(index >= 0 ? index : 0);
    setActiveView("review");
  };

  const handleReviewBack = () => {
    setActiveView("history");
  };

  const handleReviewPrev = () => {
    setCurrentIndex((i) => Math.max(0, i - 1));
  };

  const handleReviewNext = () => {
    setCurrentIndex((i) => Math.min(history.length - 1, i + 1));
  };

  const handleReviewSave = (entry) => {
    setSavedDetections((current) => [entry, ...current]);
    setActiveView("history");
  };

  const handleReviewDelete = (id) => {
    setHistory((current) => {
      const next = current.filter((item) => item.id !== id);
      if (currentIndex >= next.length && next.length > 0) {
        setCurrentIndex(next.length - 1);
      }
      return next;
    });
    setActiveView("history");
  };

  const isReview = activeView === "review";
  const isHistory = activeView === "history";
  const isWorkView = activeView === "image" || activeView === "live";

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

            {/* ── REVIEW VIEW ── */}
            {isReview && history[currentIndex] && (
              <DetectionReview
                entry={history[currentIndex]}
                currentIndex={currentIndex}
                totalCount={history.length}
                formatTimestamp={formatTimestamp}
                onPrev={handleReviewPrev}
                onNext={handleReviewNext}
                onSave={handleReviewSave}
                onDelete={handleReviewDelete}
                onBack={handleReviewBack}
              />
            )}

            {/* ── HISTORY VIEW ── */}
            {isHistory && (
              <div className="animate-fadeUp flex flex-col gap-6">
                <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
                  <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                  <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    {VIEW_TITLES.history}
                  </h1>
                  <p className="mt-2 text-sm leading-6 text-steel-300">{VIEW_SUBTITLES.history}</p>
                </header>
                <DetectionList
                  items={history}
                  onSelectItem={handleHistorySelect}
                  formatTimestamp={formatTimestamp}
                  fullWidth
                />
              </div>
            )}

            {/* ── IMAGE / LIVE VIEW ── */}
            {isWorkView && (
              <>
                <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl animate-fadeUp">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">DefineLogic</p>
                      <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                        {VIEW_TITLES[activeView]}
                      </h1>
                      <p className="mt-2 max-w-3xl text-sm leading-6 text-steel-300">
                        {VIEW_SUBTITLES[activeView]}
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

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1.65fr)_360px]">
                  <section className="flex flex-col gap-6">
                    {activeView === "image" ? (
                      <>
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
                      </>
                    ) : (
                      <LiveCamera onCameraDetection={handleCameraDetection} />
                    )}
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
              </>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}
