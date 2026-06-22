import { getToken } from "./authService";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Opens the persistent EPP detection WebSocket. Auth travels as a query param
// because browsers can't set Authorization headers on a WebSocket handshake.
export function openEppSocket({ onMessage, onError, onClose } = {}) {
  const token = getToken();
  const wsBase = API_BASE_URL.replace(/^http/i, "ws");
  const ws = new WebSocket(`${wsBase}/ws/epp/detect?token=${encodeURIComponent(token || "")}`);

  ws.onmessage = (ev) => {
    try {
      onMessage?.(JSON.parse(ev.data));
    } catch {
      /* ignore malformed frame */
    }
  };
  if (onError) ws.onerror = onError;
  if (onClose) ws.onclose = onClose;

  return ws;
}
