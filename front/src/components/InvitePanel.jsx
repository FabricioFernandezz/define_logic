import { useCallback, useEffect, useState } from "react";
import {
  addAllowedEmail,
  listAllowedEmails,
  removeAllowedEmail,
} from "../services/authService";

export default function InvitePanel() {
  const [emails, setEmails] = useState([]);
  const [newEmail, setNewEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setEmails(await listAllowedEmails());
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo cargar la lista");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAdd = async (e) => {
    e.preventDefault();
    const email = newEmail.trim().toLowerCase();
    if (!email) return;
    setSubmitting(true);
    setError(null);
    try {
      const added = await addAllowedEmail(email);
      setEmails((cur) => [added, ...cur]);
      setNewEmail("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo agregar");
    } finally {
      setSubmitting(false);
    }
  };

  const handleRemove = async (id) => {
    setError(null);
    try {
      await removeAllowedEmail(id);
      setEmails((cur) => cur.filter((e) => e.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo eliminar");
    }
  };

  return (
    <div className="rounded-2xl border border-steel-200 bg-steel-700 p-6 shadow-glow">
      <h2 className="text-lg font-bold text-white">Invitar encargados</h2>
      <p className="mt-1 text-sm text-steel-400">
        Agrega los emails autorizados. Solo esas personas podrán registrarse y acceder
        a la configuración de tu industria.
      </p>

      <form onSubmit={handleAdd} className="mt-4 flex gap-2">
        <input
          type="email"
          placeholder="email@empresa.com"
          value={newEmail}
          onChange={(e) => setNewEmail(e.target.value)}
          required
          className="flex-1 rounded-xl border border-steel-200 bg-steel-800 px-4 py-2.5 text-sm text-white placeholder-steel-500 outline-none transition focus:border-accent-500"
        />
        <button
          type="submit"
          disabled={submitting}
          className="rounded-xl px-4 py-2.5 text-sm font-semibold text-white transition disabled:opacity-60"
          style={{ background: "#F97316" }}
        >
          {submitting ? "…" : "Agregar"}
        </button>
      </form>

      {error && (
        <div className="mt-3 rounded-lg border border-warn-500/30 bg-warn-500/10 px-3 py-2 text-xs text-warn-300">
          {error}
        </div>
      )}

      <div className="mt-5">
        {loading ? (
          <p className="text-sm text-steel-400">Cargando…</p>
        ) : emails.length === 0 ? (
          <p className="text-sm text-steel-500">Aún no invitaste a nadie.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {emails.map((item) => (
              <li
                key={item.id}
                className="flex items-center justify-between rounded-xl border border-steel-200 bg-steel-800 px-4 py-2.5"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm text-white">{item.email}</p>
                  <p className="text-[11px] text-steel-500">
                    {item.used ? "Cuenta creada" : "Pendiente de registro"}
                  </p>
                </div>
                <span
                  className={`ml-3 rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    item.used
                      ? "bg-ok-500/15 text-ok-300"
                      : "bg-accent-500/15 text-accent-400"
                  }`}
                >
                  {item.used ? "activo" : "pendiente"}
                </span>
                {!item.used && (
                  <button
                    type="button"
                    onClick={() => handleRemove(item.id)}
                    className="ml-3 rounded-lg p-1.5 text-steel-400 transition hover:bg-steel-300 hover:text-white"
                    aria-label="Eliminar invitación"
                  >
                    <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
                      <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                    </svg>
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
