import { useMemo, useState } from "react";
import Sidebar from "./components/Sidebar";
import ImageUploader from "./components/ImageUploader";
import DetectionViewer from "./components/DetectionViewer";
import DetectionList from "./components/DetectionList";
import StatsPanel from "./components/StatsPanel";
import { initialDetectionHistory } from "./data/mockDetections";
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

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processError, setProcessError] = useState("");
  const [history, setHistory] = useState(initialDetectionHistory);

  const stats = useMemo(() => {
    const total = history.length;
    const helmet = history.reduce((sum, item) => sum + item.detections.filter((d) => d.helmetDetected).length, 0);
    const noHelmet = history.reduce((sum, item) => sum + item.detections.filter((d) => !d.helmetDetected).length, 0);

    return { total, helmet, noHelmet };
  }, [history]);

  const handleSelectImage = async (file) => {
    if (!file) {
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    const naturalSize = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        resolve({ width: image.naturalWidth, height: image.naturalHeight });
      };
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
    if (!selectedImage?.file || isProcessing) {
      return;
    }

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
      setHistory((current) => [nextEntry, ...current].slice(0, 8));
    } catch (error) {
      setProcessError(error instanceof Error ? error.message : "Error inesperado procesando la imagen");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleHistorySelect = (entry) => {
    setSelectedImage({
      id: entry.id,
      file: entry.file,
      name: entry.name,
      previewUrl: entry.previewUrl,
      processedPreviewUrl: entry.annotatedPreviewUrl || null,
      naturalWidth: entry.naturalWidth,
      naturalHeight: entry.naturalHeight,
      result: entry.result,
      confidence: entry.confidence,
      detections: entry.detections,
      modelName: entry.modelName,
      processingTimeMs: entry.processingTimeMs,
      timestamp: entry.timestamp,
    });
    setDetections(entry.detections);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setDetections([]);
    setProcessError("");
  };

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="min-h-screen bg-steel-950 text-steel-50">
      <div className="flex min-h-screen">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((value) => !value)}
          onNavigate={scrollToSection}
        />

        <main className="flex-1 px-4 py-4 sm:px-6 lg:px-8">
          <div className="mx-auto flex max-w-7xl flex-col gap-6">
            <header className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl animate-fadeUp">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-accent-300/80">
                    Helmet Vision Dashboard
                  </p>
                  <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    Detección de casco en imágenes estáticas
                  </h1>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-steel-300">
                    Carga una imagen, procesa el resultado del modelo y revisa las últimas detecciones desde un panel moderno, limpio y escalable.
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
                  onNavigateHistory={() => scrollToSection("history-panel")}
                />

                <StatsPanel history={history} />
              </section>

              <aside id="history-panel" className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
                <DetectionList
                  items={history}
                  onSelectItem={handleHistorySelect}
                  formatTimestamp={formatTimestamp}
                />
              </aside>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}