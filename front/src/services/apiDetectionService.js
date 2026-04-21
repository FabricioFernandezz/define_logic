const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const toNormalizedDetections = (response, imageMeta) => {
  const { width, height } = imageMeta;
  return (response.detections || []).map((item, index) => ({
    id: `api-${index}-${item.personId}`,
    bbox: {
      x: item.bbox_pixels[0],
      y: item.bbox_pixels[1],
      width: item.bbox_pixels[2] - item.bbox_pixels[0],
      height: item.bbox_pixels[3] - item.bbox_pixels[1],
    },
    helmetDetected: Boolean(item.helmetDetected),
    label: item.label,
    confidence: Number(item.confidence ?? 0),
    personIndex: item.personId + 1,
    bboxNorm: item.bbox_norm || [0, 0, 0, 0],
    imageWidth: width,
    imageHeight: height,
  }));
};

export const analyzeImageWithBackend = async (file, imageMeta = {}) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/detect-image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "No se pudo procesar la imagen en el backend");
  }

  const data = await response.json();
  const detections = toNormalizedDetections(data, imageMeta);

  return {
    modelName: data.modelName || "vit_epp_best.pt",
    processingTimeMs: data.processingTimeMs ?? 0,
    imageSize: data.imageSize || {
      width: imageMeta.naturalWidth || imageMeta.width || 0,
      height: imageMeta.naturalHeight || imageMeta.height || 0,
    },
    detections,
    summary: {
      helmetCount: data.summary?.helmetCount ?? detections.filter((item) => item.helmetDetected).length,
      noHelmetCount: data.summary?.noHelmetCount ?? detections.filter((item) => !item.helmetDetected).length,
      result: data.summary?.result ?? (detections.some((item) => !item.helmetDetected) ? "mixto" : "con casco"),
      confidence: data.summary?.confidence ?? 0,
    },
    annotatedImage: data.annotatedImage,
  };
};

export const pingBackend = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("Backend no disponible");
  }
  return response.json();
};
