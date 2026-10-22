import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// Preserve the legacy / app; add an isolated /pose-monitor.html entry.
// This monitor does not import App.tsx or its flight-control event emitters.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        poseMonitor: fileURLToPath(new URL('./pose-monitor.html', import.meta.url)),
      },
    },
  },
})
