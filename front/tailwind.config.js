/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Syne", "sans-serif"],
        sans: ["DM Sans", "sans-serif"],
      },
      boxShadow: {
        glow:    "0 0 0 1px rgba(255,255,255,0.05), 0 4px 32px rgba(0,0,0,0.6)",
        "glow-sm": "0 0 0 1px rgba(255,255,255,0.04), 0 2px 12px rgba(0,0,0,0.5)",
        "glow-orange": "0 0 0 1px rgba(249,115,22,0.12), 0 4px 24px rgba(249,115,22,0.15)",
      },
      colors: {
        // Superficies oscuras — tema industrial
        steel: {
          50:  "#111113",   // dark surface (replaces white cards)
          100: "#0D0D0E",   // body/app bg
          200: "#2A2A2E",   // borders
          300: "#333338",   // hover overlay
          400: "#C0C7D4",   // muted text (legible on dark bg)
          500: "#9CA3AF",   // secondary text
          600: "#6B7280",   // dim text
          700: "#161618",   // card/surface bg
          800: "#111113",   // deep surface
          900: "#0A0A0C",   // darkest bg
          950: "#080809",   // absolute dark
        },
        // Naranja industrial — acento principal
        accent: {
          50:  "#1A0F06",
          100: "#2D1A0B",
          200: "#7C2D12",
          300: "#FDBA74",
          400: "#FB923C",
          500: "#F97316",
          600: "#EA580C",
          700: "#C2410C",
        },
        // Verde — cumple / seguro
        ok: {
          50:  "#F0FDF4",
          100: "#DCFCE7",
          200: "#BBF7D0",
          300: "#86EFAC",
          400: "#4ADE80",
          500: "#22C55E",
          600: "#16A34A",
        },
        // Rojo — no cumple / riesgo
        warn: {
          50:  "#FFF1F2",
          100: "#FFE4E6",
          200: "#FECDD3",
          300: "#FDA4AF",
          400: "#F87171",
          500: "#EF4444",
          600: "#DC2626",
        },
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideInRight: {
          "0%":   { opacity: "0", transform: "translateX(16px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0.4" },
        },
      },
      animation: {
        fadeUp:       "fadeUp 0.4s ease-out forwards",
        slideInRight: "slideInRight 0.28s ease-out forwards",
      },
    },
  },
  plugins: [],
};
