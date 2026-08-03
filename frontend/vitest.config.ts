import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

/**
 * Vitest config for the frontend's pure logic.
 *
 * Scope is deliberately narrow: the helpers in src/lib are plain functions with
 * real domain rules (snake-draft slot math, ADP value/reach direction, search
 * matching) and no DOM, so they can be tested with no jsdom, no React testing
 * library, and no component rendering setup.
 *
 * This exists because `adpValue` shipped with its value/reach direction
 * inverted — a three-line test would have caught it, but there was nowhere to
 * put one.
 */
export default defineConfig({
  resolve: {
    alias: {
      // Mirrors the "@/*" -> "./src/*" path mapping in tsconfig.json.
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
