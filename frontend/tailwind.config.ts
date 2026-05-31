import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        // Draft room dark theme
        surface: {
          DEFAULT: "#0f172a", // slate-950 — main background
          card: "#1e293b",    // slate-800 — card backgrounds
          border: "#334155",  // slate-700 — borders
        },
      },
    },
  },
  plugins: [],
};

export default config;
