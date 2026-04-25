import { analyzeImage as analyzeImageMock } from "./mockDetectionService";
import { analyzeImageWithBackend, pingBackend } from "./apiDetectionService";

let backendAvailable = null;
const ENABLE_MOCK_FALLBACK = import.meta.env.VITE_ENABLE_MOCK_FALLBACK === "true";

const hasBackend = async () => {
  if (backendAvailable === true) {
    return true;
  }

  try {
    await pingBackend();
    backendAvailable = true;
  } catch {
    backendAvailable = false;
  }

  return backendAvailable;
};

export const analyzeImage = async (file, imageMeta = {}) => {
  if (await hasBackend()) {
    try {
      return await analyzeImageWithBackend(file, imageMeta);
    } catch (error) {
      console.error("Error procesando en backend:", error);
      backendAvailable = false;

      if (!ENABLE_MOCK_FALLBACK) {
        throw new Error(
          "No se pudo procesar con el backend real. "
        );
      }
    }
  }

  if (!ENABLE_MOCK_FALLBACK) {
    throw new Error(
      "Error: No se pudo conectar al backend"
    );
  }

  return analyzeImageMock(file, imageMeta);
};
