import { useState } from "react";
import { useAuth } from "../context/AuthContext";

const MODES = {
  login: { title: "Iniciar sesión", cta: "Entrar" },
  "register-owner": { title: "Crear industria", cta: "Registrar industria" },
  "register-member": { title: "Unirme a una industria", cta: "Crear cuenta" },
};

const inputClass =
  "w-full rounded-xl border border-steel-200 bg-steel-800 px-4 py-2.5 text-sm text-white placeholder-steel-500 outline-none transition focus:border-accent-500";

export default function AuthScreen() {
  const { login, registerOwner, registerMember } = useAuth();
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({
    nombre: "",
    email: "",
    password: "",
    industriaNombre: "",
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const meta = MODES[mode];
  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const switchMode = (next) => {
    setMode(next);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "login") {
        await login(form.email, form.password);
      } else if (mode === "register-owner") {
        await registerOwner(form.nombre, form.email, form.password, form.industriaNombre);
      } else {
        await registerMember(form.nombre, form.email, form.password);
      }
      // AuthProvider actualiza user -> App se re-renderiza
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error inesperado");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="flex min-h-screen items-center justify-center px-4 text-white"
      style={{ background: "#0D0D0E" }}
    >
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center gap-2.5">
          <div
            className="flex h-9 w-9 items-center justify-center rounded-xl text-xs font-black text-white shadow"
            style={{ background: "#F97316" }}
          >
            DL
          </div>
          <span className="text-base font-semibold">DefineLogic</span>
        </div>

        <div className="rounded-2xl border border-steel-200 bg-steel-700 p-6 shadow-glow">
          <h1 className="mb-1 text-xl font-bold">{meta.title}</h1>
          <p className="mb-5 text-sm text-steel-400">
            {mode === "login"
              ? "Accede a la configuración de tu industria."
              : mode === "register-owner"
              ? "Registra tu empresa y serás el encargado dueño."
              : "Tu email debe haber sido invitado por el encargado."}
          </p>

          <form onSubmit={handleSubmit} className="flex flex-col gap-3">
            {mode !== "login" && (
              <input
                className={inputClass}
                placeholder="Nombre completo"
                value={form.nombre}
                onChange={set("nombre")}
                required
              />
            )}
            {mode === "register-owner" && (
              <input
                className={inputClass}
                placeholder="Nombre de la industria / empresa"
                value={form.industriaNombre}
                onChange={set("industriaNombre")}
                required
              />
            )}
            <input
              className={inputClass}
              type="email"
              placeholder="Email"
              value={form.email}
              onChange={set("email")}
              required
            />
            <input
              className={inputClass}
              type="password"
              placeholder="Contraseña"
              value={form.password}
              onChange={set("password")}
              minLength={mode === "login" ? undefined : 6}
              required
            />

            {error && (
              <div className="rounded-lg border border-warn-500/30 bg-warn-500/10 px-3 py-2 text-xs text-warn-300">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="mt-1 rounded-xl px-4 py-2.5 text-sm font-semibold text-white transition disabled:opacity-60"
              style={{ background: "#F97316" }}
            >
              {loading ? "Procesando…" : meta.cta}
            </button>
          </form>

          <div className="mt-5 space-y-1.5 text-center text-xs text-steel-400">
            {mode === "login" ? (
              <>
                <p>
                  ¿Nueva empresa?{" "}
                  <button
                    type="button"
                    onClick={() => switchMode("register-owner")}
                    className="font-semibold text-accent-500 hover:text-accent-400"
                  >
                    Crear industria
                  </button>
                </p>
                <p>
                  ¿Te invitaron?{" "}
                  <button
                    type="button"
                    onClick={() => switchMode("register-member")}
                    className="font-semibold text-accent-500 hover:text-accent-400"
                  >
                    Unirme a una industria
                  </button>
                </p>
              </>
            ) : (
              <p>
                ¿Ya tienes cuenta?{" "}
                <button
                  type="button"
                  onClick={() => switchMode("login")}
                  className="font-semibold text-accent-500 hover:text-accent-400"
                >
                  Iniciar sesión
                </button>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
