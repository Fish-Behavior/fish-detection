import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Host and ports come from the repository-root .env (FISHLAB_*); real environment variables win.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', 'FISHLAB_')
  const host = env.FISHLAB_HOST || '127.0.0.1'
  if (!['127.0.0.1', 'localhost', '::1'].includes(host)) throw new Error('FISHLAB_HOST must be a loopback address; the app has no authentication.')
  const api = `http://${host}:${env.FISHLAB_API_PORT || 8008}`
  return { plugins: [react()], server: {
    host, port: Number(env.FISHLAB_PORT || 8000), strictPort: true,
    proxy: { '/api': api, '/docs': api, '/openapi.json': api },
  } }
})
