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

export const getSavedDetectionsFromBackend = async () => {
  const response = await fetch(`${API_BASE_URL}/api/saved-detections`);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "No se pudo obtener imágenes guardadas");
  }
  return response.json();
};

export const getZoneConfig = async () => {
  const response = await fetch(`${API_BASE_URL}/api/epp/zone-config`);
  if (!response.ok) throw new Error("No se pudo obtener configuración de zonas");
  return response.json();
};

export const saveZoneConfig = async ({ zones, defaultZoneEpp, defaultZoneActive, defaultZoneRequirePerson }) => {
  const response = await fetch(`${API_BASE_URL}/api/epp/zone-config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ zones, defaultZoneEpp, defaultZoneActive, defaultZoneRequirePerson }),
  });
  if (!response.ok) throw new Error("No se pudo guardar configuración de zonas");
  return response.json();
};

export const generateAiDescription = async ({ imageDataUrl, detections, personCount, result, alertingZones, defaultZoneResult }) => {
  const response = await fetch(`${API_BASE_URL}/api/epp/generate-description`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_data_url: imageDataUrl,
      detections: detections || [],
      person_count: personCount || 0,
      result: result || "",
      alerting_zones: alertingZones || null,
      default_zone_result: defaultZoneResult || null,
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => null);
    throw new Error(err?.detail || "No se pudo generar descripción");
  }
  const data = await response.json();
  return data.description;
};

export const saveDetectionToBackend = async ({ nombre, imagen, descripcion }) => {
  const response = await fetch(`${API_BASE_URL}/api/saved-detections`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      nombre,
      imagen,
      descripcion,
    }),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => null);
    const detail = errorPayload?.detail;
    const message = typeof detail === "string" ? detail : (detail?.message || "No se pudo guardar la detección");
    const error = new Error(message);
    error.status = response.status;
    error.payload = errorPayload;
    throw error;
  }

  return response.json();
};
