import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// build 產物進 dist/（由 FastAPI 服務）；dev 時前端 5173、後端 8000。
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
});
