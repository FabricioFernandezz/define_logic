const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const TOKEN_KEY = "dl_token";

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (token) => localStorage.setItem(TOKEN_KEY, token);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

export const authHeaders = () => {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

const parseError = async (response, fallback) => {
  const payload = await response.json().catch(() => null);
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  return fallback;
};

const post = async (path, body) => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await parseError(response, "No se pudo completar la solicitud"));
  }
  return response.json();
};

export const loginRequest = (email, password) =>
  post("/api/auth/login", { email, password });

export const registerOwnerRequest = (nombre, email, password, industriaNombre) =>
  post("/api/auth/register-owner", { nombre, email, password, industriaNombre });

export const registerMemberRequest = (nombre, email, password) =>
  post("/api/auth/register", { nombre, email, password });

export const fetchMe = async () => {
  const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
    headers: { ...authHeaders() },
  });
  if (!response.ok) throw new Error("No autenticado");
  return response.json();
};

// --- Whitelist (owner) ---
export const listAllowedEmails = async () => {
  const response = await fetch(`${API_BASE_URL}/api/auth/allowed-emails`, {
    headers: { ...authHeaders() },
  });
  if (!response.ok) throw new Error(await parseError(response, "No se pudo cargar la lista"));
  return response.json();
};

export const addAllowedEmail = async (email) => {
  const response = await fetch(`${API_BASE_URL}/api/auth/allowed-emails`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ email }),
  });
  if (!response.ok) throw new Error(await parseError(response, "No se pudo agregar el email"));
  return response.json();
};

export const removeAllowedEmail = async (id) => {
  const response = await fetch(`${API_BASE_URL}/api/auth/allowed-emails/${id}`, {
    method: "DELETE",
    headers: { ...authHeaders() },
  });
  if (!response.ok) throw new Error(await parseError(response, "No se pudo eliminar"));
  return response.json();
};
