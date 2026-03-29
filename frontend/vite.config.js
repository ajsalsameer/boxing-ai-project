import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // WebSocket — punch recognition stream
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
      // All REST routes FastAPI exposes
      '/status': { target: 'http://localhost:8000', changeOrigin: true },
      '/users':  { target: 'http://localhost:8000', changeOrigin: true },
      '/api':    { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})