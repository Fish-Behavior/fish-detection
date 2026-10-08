import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Host and ports come from the repository-root .env (FISHLAB_*); real environment variables win.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', 'FISHLAB_')
  for (const name of ['FISHLAB_HOST', 'FISHLAB_PORT', 'FISHLAB_API_PORT']) if (!env[name]) throw new Error(`Set ${name} in the repository-root .env.`)
  const host = env.FISHLAB_HOST
  if (!['127.0.0.1', 'localhost', '::1'].includes(host)) throw new Error('FISHLAB_HOST must be a loopback address; the app has no authentication.')
  const api = `http://${host}:${env.FISHLAB_API_PORT}`
  return { plugins: [react()], server: {
    host, port: Number(env.FISHLAB_PORT), strictPort: true,
    proxy: { '/api': api, '/docs': api, '/openapi.json': api },
  } }
})
