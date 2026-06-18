import { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  clearToken,
  fetchMe,
  getToken,
  loginRequest,
  registerMemberRequest,
  registerOwnerRequest,
  setToken,
} from "../services/authService";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Restaurar sesion al montar
  useEffect(() => {
    const token = getToken();
    if (!token) {
      setLoading(false);
      return;
    }
    fetchMe()
      .then((u) => setUser(u))
      .catch(() => clearToken())
      .finally(() => setLoading(false));
  }, []);

  const handleAuthSuccess = useCallback((data) => {
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);

  const login = useCallback(
    async (email, password) => handleAuthSuccess(await loginRequest(email, password)),
    [handleAuthSuccess]
  );

  const registerOwner = useCallback(
    async (nombre, email, password, industriaNombre) =>
      handleAuthSuccess(await registerOwnerRequest(nombre, email, password, industriaNombre)),
    [handleAuthSuccess]
  );

  const registerMember = useCallback(
    async (nombre, email, password) =>
      handleAuthSuccess(await registerMemberRequest(nombre, email, password)),
    [handleAuthSuccess]
  );

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{ user, loading, login, registerOwner, registerMember, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth debe usarse dentro de AuthProvider");
  return ctx;
};
