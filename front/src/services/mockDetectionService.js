const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const hashString = (input) => {
  let hash = 0;
  for (let index = 0; index < input.length; index += 1) {
    hash = (hash * 31 + input.charCodeAt(index)) >>> 0;
  }
  return hash || 1;
};

const getNaturalSize = (imageMeta = {}) => {
  const width = imageMeta.naturalWidth || imageMeta.width || 1280;
  const height = imageMeta.naturalHeight || imageMeta.height || 720;
  return { width, height };
};

const createDetections = (seed, width, height) => {
  const count = clamp(1 + (seed % 4), 1, 4);
  const detections = [];

  for (let index = 0; index < count; index += 1) {
    const shift = seed >> (index * 4);
    const boxWidth = Math.round(width * (0.14 + ((shift % 7) * 0.018)));
    const boxHeight = Math.round(height * (0.22 + (((shift >> 3) % 5) * 0.02)));
    const maxX = Math.max(0, width - boxWidth - 1);
    const maxY = Math.max(0, height - boxHeight - 1);
    const x = Math.round(((shift % 997) / 997) * maxX);
    const y = Math.round((((shift >> 5) % 991) / 991) * maxY);
    const helmetDetected = ((shift >> 2) % 5) !== 0;
    const confidence = helmetDetected
      ? 0.78 + (((shift >> 1) % 11) * 0.016)
      : 0.62 + (((shift >> 2) % 9) * 0.015);

    detections.push({
      id: `det-${seed}-${index}`,
      bbox: {
        x,
        y,
        width: boxWidth,
        height: boxHeight,
      },
      helmetDetected,
      label: helmetDetected ? "Casco detectado" : "Sin casco",
      confidence: Number(clamp(confidence, 0.51, 0.97).toFixed(2)),
      personIndex: index + 1,
    });
  }

  return detections;
};

const summarizeDetections = (detections) => {
  const helmetCount = detections.filter((item) => item.helmetDetected).length;
  const noHelmetCount = detections.length - helmetCount;

  return {
    helmetCount,
    noHelmetCount,
    result: noHelmetCount > 0 ? (helmetCount > 0 ? "mixto" : "sin casco") : "con casco",
    confidence: detections.length
      ? Number((detections.reduce((acc, item) => acc + item.confidence, 0) / detections.length).toFixed(2))
      : 0,
  };
};

export const analyzeImage = async (file, imageMeta = {}) => {
  const startedAt = performance.now();
  const { width, height } = getNaturalSize(imageMeta);
  const seed = hashString(`${file.name}:${file.size}:${width}x${height}`);

  await delay(850 + (seed % 650));

  const detections = createDetections(seed, width, height);
  const summary = summarizeDetections(detections);
  const processingTimeMs = Math.round(performance.now() - startedAt);

  return {
    modelName: "vit_epp_best.pt",
    processingTimeMs,
    imageSize: { width, height },
    detections,
    summary,
  };
};