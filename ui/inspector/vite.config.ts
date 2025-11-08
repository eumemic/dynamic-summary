import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_RAGZOOM_API_URL ?? "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        "/documents": {
          target,
          changeOrigin: true,
          secure: false
        },
        "/query": {
          target,
          changeOrigin: true,
          secure: false
        },
        "/status": {
          target,
          changeOrigin: true,
          secure: false
        },
        "/config": {
          target,
          changeOrigin: true,
          secure: false
        },
        "/index": {
          target,
          changeOrigin: true,
          secure: false
        },
        "/pin": {
          target,
          changeOrigin: true,
          secure: false
        },
        "/clear": {
          target,
          changeOrigin: true,
          secure: false
        }
      }
    }
  };
});
