/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // plan §P1.6: dev server proxies /api to the FastAPI backend (:8000) so
    // the frontend never hardcodes an absolute API origin. `/health`
    // (app/api/health.py) lives outside the /api prefix, so it needs its
    // own rule -- without it, GET /health hits Vite's own SPA fallback
    // (200 text/html) instead of the backend, and useHealth's `.catch`
    // silently reports "degraded" forever regardless of real ES/Milvus/
    // Mongo status (found 2026-08-15: banner stayed stuck on "Elasticsearch
    // không khả dụng" even with a fully healthy backend).
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/setupTests.ts',
  },
})
