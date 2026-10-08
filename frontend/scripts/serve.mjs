import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { createServer } from 'node:net'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const frontend = fileURLToPath(new URL('..', import.meta.url))
const root = resolve(frontend, '..')
const python = resolve(root, '.venv/bin/python')
if (existsSync(resolve(root, '.env'))) process.loadEnvFile(resolve(root, '.env')) // real env vars win over .env
const host = process.env.FISHLAB_HOST || '127.0.0.1'
function portFrom(name, fallback) {
  const text = process.env[name] || String(fallback), value = Number(text)
  if (!/^\d+$/.test(text) || value < 1 || value > 65535) { console.error(`FishLab could not start: ${name} must be a port number, got "${text}".`); process.exit(1) }
  return value
}
if (!['127.0.0.1', 'localhost', '::1'].includes(host)) { console.error('FishLab could not start: FISHLAB_HOST must be a loopback address; the app has no authentication.'); process.exit(1) }
const port = portFrom('FISHLAB_PORT', 8000)
const devApiPort = portFrom('FISHLAB_API_PORT', 8008)
const address = `http://${host}:${port}`
const instance = randomUUID()
const children = new Set()
let stopping = false

function signalChild(child, signal) {
  if (!child.pid) return
  try { process.kill(-child.pid, signal) } catch (error) {
    if (error.code !== 'ESRCH') console.error(`Could not stop an owned process: ${error.message}`)
  }
}
function stop(code) {
  if (stopping) return
  stopping = true
  process.exitCode = code
  for (const child of children) signalChild(child, 'SIGTERM')
  setTimeout(() => {
    for (const child of children) signalChild(child, 'SIGKILL')
  }, 3000).unref()
}
process.on('SIGINT', () => stop(0))
process.on('SIGTERM', () => stop(0))

function launch(command, args, label, persistent = true, cwd = root) {
  const child = spawn(command, args, { cwd, stdio: 'inherit', detached: true,
    env: { ...process.env, FISHLAB_INSTANCE_ID: instance } })
  children.add(child)
  child.once('error', (error) => {
    console.error(`${label} could not start: ${error.message}`)
    stop(1)
  })
  child.once('exit', (code, signal) => {
    children.delete(child)
    if (persistent && !stopping) {
      console.error(`${label} stopped (${signal ?? code}). Check the error above.`)
      stop(1)
    }
  })
  return child
}
async function build() {
  const child = launch('npm', ['run', 'build'], 'Frontend build', false, frontend)
  await new Promise((done, reject) => {
    child.once('error', reject)
    child.once('exit', (code) => code === 0 ? done() : reject(new Error('Frontend build failed; the app was not started.')))
  })
}
async function requireFreePort(port) {
  await new Promise((done, reject) => {
    const probe = createServer()
    probe.once('error', (error) => reject(new Error(error.code === 'EADDRINUSE'
      ? `Port ${port} is already in use. Stop the existing FishLab command before starting another; no process was stopped.`
      : `Cannot use port ${port}: ${error.message}`)))
    probe.listen(port, host, () => probe.close(done))
  })
}
async function waitForApi(port) {
  const deadline = Date.now() + 30000
  while (!stopping && Date.now() < deadline) {
    try {
      const response = await fetch(`http://${host}:${port}/api/health`, { signal: AbortSignal.timeout(1000) })
      const health = await response.json()
      if (!stopping && response.ok && health.ok && health.instance_id === instance) return
    } catch { /* The owned service may still be starting. */ }
    await new Promise((done) => setTimeout(done, 200))
  }
  if (!stopping) throw new Error('The API did not become ready within 30 seconds. Check the backend error above.')
}
async function main() {
  const args = process.argv.slice(2)
  const dev = args.includes('--dev'), preview = args.includes('--preview')
  if (dev && preview) throw new Error('Choose dev or preview, not both.')
  const options = []
  const accepted = ['--chat-url', '--predictions', '--classifier-root', '--model-run']
  for (let i = 0; i < args.length; i++) {
    if (['--dev', '--preview'].includes(args[i])) continue
    const [key, ...inline] = args[i].split('=')
    if (!accepted.includes(key)) throw new Error(`Unsupported option ${key}. These commands use ${address}; optional review settings: ${accepted.join(', ')}.`)
    const value = inline.length ? inline.join('=') : args[++i]
    if (!value || value.startsWith('--')) throw new Error(`${key} needs a value.`)
    options.push(key, value)
  }
  if (!existsSync(python)) throw new Error('Project Python environment is missing. Create .venv and install the project dependencies first.')
  if (preview && !existsSync(resolve(frontend, 'dist/index.html'))) throw new Error('No frontend build exists. Run npm run build, or use npm start to build and start together.')
  await requireFreePort(port)
  if (dev) await requireFreePort(devApiPort)
  if (!dev && !preview) await build()
  if (stopping) return
  const apiPort = dev ? devApiPort : port
  launch(python, ['-u', '-m', 'prepds', 'review', '--host', host, '--port', String(apiPort),
    '--frontend-dir', resolve(frontend, 'dist'), ...options], 'Python API')
  await waitForApi(apiPort)
  if (stopping) return
  if (dev) {
    launch(process.execPath, [resolve(frontend, 'node_modules/vite/bin/vite.js')], 'Development frontend', true, frontend)
    await waitForApi(port)
  }
  if (!stopping) console.log(`\nFishLab ready: ${address}\nPress Ctrl+C to stop the app.\n`)
}
main().catch((error) => {
  if (!stopping) console.error(`FishLab could not start: ${error.message}`)
  stop(1)
})
