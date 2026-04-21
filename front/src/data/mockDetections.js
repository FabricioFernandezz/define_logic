const makeSvgPreview = ({ title, subtitle, accent = "#06b6d4", secondAccent = "#f97316" }) => {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 520">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#0f172a" />
          <stop offset="100%" stop-color="#020617" />
        </linearGradient>
        <linearGradient id="shine" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="${accent}" stop-opacity="0.85" />
          <stop offset="100%" stop-color="${secondAccent}" stop-opacity="0.85" />
        </linearGradient>
      </defs>
      <rect width="800" height="520" rx="36" fill="url(#bg)" />
      <circle cx="650" cy="90" r="130" fill="${accent}" fill-opacity="0.15" />
      <circle cx="120" cy="420" r="170" fill="${secondAccent}" fill-opacity="0.12" />
      <rect x="72" y="72" width="656" height="376" rx="30" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" />
      <rect x="120" y="130" width="210" height="230" rx="26" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.08)" />
      <rect x="370" y="108" width="270" height="270" rx="32" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.08)" />
      <rect x="410" y="156" width="110" height="64" rx="20" fill="url(#shine)" opacity="0.85" />
      <text x="72" y="488" font-family="Inter, Segoe UI, sans-serif" font-size="36" fill="#f8fafc" font-weight="700">${title}</text>
      <text x="72" y="520" font-family="Inter, Segoe UI, sans-serif" font-size="22" fill="#cbd5e1">${subtitle}</text>
    </svg>
  `;

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

export const initialDetectionHistory = [
  {
    id: "hist-01",
    name: "obra_norte_001.jpg",
    timestamp: "2026-04-20T09:15:00Z",
    previewUrl: makeSvgPreview({
      title: "obra_norte_001",
      subtitle: "Resultado: 3/4 con casco",
      accent: "#06b6d4",
      secondAccent: "#10b981",
    }),
    result: "mixto",
    confidence: 0.88,
    detections: [
      {
        id: "hist-01-1",
        bbox: { x: 110, y: 72, width: 140, height: 220 },
        helmetDetected: true,
        label: "Casco detectado",
        confidence: 0.93,
      },
      {
        id: "hist-01-2",
        bbox: { x: 286, y: 82, width: 150, height: 228 },
        helmetDetected: true,
        label: "Casco detectado",
        confidence: 0.89,
      },
      {
        id: "hist-01-3",
        bbox: { x: 494, y: 90, width: 146, height: 232 },
        helmetDetected: false,
        label: "Sin casco",
        confidence: 0.81,
      },
      {
        id: "hist-01-4",
        bbox: { x: 634, y: 86, width: 118, height: 216 },
        helmetDetected: true,
        label: "Casco detectado",
        confidence: 0.91,
      },
    ],
  },
  {
    id: "hist-02",
    name: "almacen_entrada_044.png",
    timestamp: "2026-04-20T08:48:00Z",
    previewUrl: makeSvgPreview({
      title: "almacen_entrada_044",
      subtitle: "Resultado: sin casco",
      accent: "#f97316",
      secondAccent: "#ef4444",
    }),
    result: "sin casco",
    confidence: 0.77,
    detections: [
      {
        id: "hist-02-1",
        bbox: { x: 156, y: 120, width: 188, height: 270 },
        helmetDetected: false,
        label: "Sin casco",
        confidence: 0.87,
      },
      {
        id: "hist-02-2",
        bbox: { x: 458, y: 102, width: 174, height: 260 },
        helmetDetected: false,
        label: "Sin casco",
        confidence: 0.79,
      },
    ],
  },
  {
    id: "hist-03",
    name: "patio_sur_122.jpeg",
    timestamp: "2026-04-20T08:12:00Z",
    previewUrl: makeSvgPreview({
      title: "patio_sur_122",
      subtitle: "Resultado: 2/2 con casco",
      accent: "#10b981",
      secondAccent: "#06b6d4",
    }),
    result: "con casco",
    confidence: 0.94,
    detections: [
      {
        id: "hist-03-1",
        bbox: { x: 160, y: 92, width: 168, height: 244 },
        helmetDetected: true,
        label: "Casco detectado",
        confidence: 0.95,
      },
      {
        id: "hist-03-2",
        bbox: { x: 470, y: 110, width: 164, height: 238 },
        helmetDetected: true,
        label: "Casco detectado",
        confidence: 0.92,
      },
    ],
  },
];