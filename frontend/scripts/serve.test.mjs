import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { copyFile, mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const frontend = fileURLToPath(new URL('..', import.meta.url))
const root = resolve(frontend, '..')
const launcher = resolve(frontend, 'scripts/serve.mjs')
const delay = (ms) => new Promise((done) => setTimeout(done, ms))

async function fixture() {
  const directory = await mkdtemp(resolve(tmpdir(), 'fishlab-startup-'))
  await mkdir(resolve(directory, 'videos'))
  const probes = [createServer(), createServer()]
  for (const probe of probes) { probe.listen(0, '127.0.0.1'); await once(probe, 'listening') }
  const [port, apiPort] = probes.map((probe) => probe.address().port)
  await Promise.all(probes.map((probe) => new Promise((done) => probe.close(done))))
  return { directory, env: { ...process.env, PDS_VIDEO_DIR: resolve(directory, 'videos'),
    PDS_OUTPUT_DIR: resolve(directory, 'outputs'), PDS_ACCEPTED_DIR: resolve(directory, 'accepted'),
    FISHLAB_HOST: '127.0.0.1', FISHLAB_PORT: String(port), FISHLAB_API_PORT: String(apiPort),
    FISHLAB_CHAT_URL: ' ' } }
}
function start(args, env, script = launcher) {
  const child = spawn(process.execPath, [script, ...args], { cwd: frontend, env, stdio: ['ignore', 'pipe', 'pipe'] })
  const running = { child, output: '', closed: false, exit: null }
  child.stdout.on('data', (data) => { running.output += data })
  child.stderr.on('data', (data) => { running.output += data })
  running.exit = once(child, 'close').then(([code]) => { running.closed = true; return code })
  return running
}
async function ready(running) {
  for (let i = 0; i < 200 && !running.closed; i++) {
    if (running.output.includes('FishLab ready:')) return
    await delay(100)
  }
  assert.fail(running.output || 'Startup did not become ready.')
}
async function finish(running) {
  let timer
  try {
    return await Promise.race([running.exit, new Promise((_done, reject) => {
      timer = setTimeout(() => reject(new Error('Owned launcher did not stop.')), 10000)
    })])
  } finally { clearTimeout(timer) }
}
async function stop(running) {
  if (!running.closed) running.child.kill('SIGTERM')
  await finish(running)
}
async function assertStopped(port) {
  await assert.rejects(fetch(`http://127.0.0.1:${port}/api/health`, { signal: AbortSignal.timeout(1000) }))
}
async function copiedLauncher(directory) {
  const project = resolve(directory, 'project')
  await mkdir(project)
  await symlink(resolve(root, '.venv'), resolve(project, '.venv'), 'dir')
  await mkdir(resolve(project, 'frontend/scripts'), { recursive: true })
  const script = resolve(project, 'frontend/scripts/serve.mjs')
  await copyFile(launcher, script)
  return { project, script }
}

test('start, dev and preview serve frontend plus API at one URL and stop their owned services', { timeout: 60000 }, async () => {
  const { directory, env } = await fixture()
  const address = `http://127.0.0.1:${env.FISHLAB_PORT}`
  try {
    for (const args of [[], ['--dev'], ['--preview', '--chat-url=http://127.0.0.1:8002']]) {
      const running = start(args, env)
      try {
        await ready(running)
        assert.ok(running.output.includes(`FishLab ready: ${address}`))
        const response = await fetch(`${address}/`)
        assert.equal(response.status, 200)
        assert.match(await response.text(), /FishLab/)
        const health = await (await fetch(`${address}/api/health`)).json()
        assert.equal(health.ok, true)
        assert.ok(health.instance_id)
        assert.equal(health.chat_configured, args.includes('--preview'))
        assert.deepEqual(await (await fetch(`${address}/api/sessions`)).json(), [])
        assert.equal((await fetch(`${address}/docs`)).status, 200)
        const missing = await fetch(`${address}/api/unknown`)
        assert.equal(missing.status, 404)
        assert.deepEqual(await missing.json(), { detail: 'Not Found' })
      } finally { await stop(running) }
      await assertStopped(env.FISHLAB_PORT)
      if (args.includes('--dev')) await assertStopped(env.FISHLAB_API_PORT)
    }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('occupied public or private port fails without killing or borrowing another server', async () => {
  const { directory, env } = await fixture()
  try {
    for (const port of [Number(env.FISHLAB_PORT), Number(env.FISHLAB_API_PORT)]) {
      const other = createServer((_request, response) => response.end('unrelated service'))
      other.listen(port, '127.0.0.1'); await once(other, 'listening')
      try {
        const running = start(['--dev'], env)
        assert.equal(await finish(running), 1)
        assert.match(running.output, new RegExp(`Port ${port} is already in use`))
        assert.equal(await (await fetch(`http://127.0.0.1:${port}`)).text(), 'unrelated service')
      } finally { await new Promise((done) => other.close(done)) }
    }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('unsafe override spellings and missing builds fail before starting a service', async () => {
  const { directory, env } = await fixture()
  try {
    const { script } = await copiedLauncher(directory)
    for (const args of [['--host', '0.0.0.0'], ['--port=9000'], ['-p', '9000'], ['--', '--port', '9000'], ['--dev', '--preview']]) {
      const running = start(args, env, script)
      assert.equal(await finish(running), 1)
      assert.match(running.output, /Unsupported option|Choose dev or preview/)
    }
    const missing = start(['--preview'], env, script)
    assert.equal(await finish(missing), 1)
    assert.match(missing.output, /No frontend build exists/)
    await assertStopped(env.FISHLAB_PORT)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('failed build never starts the API; failed development frontend stops the owned API', async () => {
  const { directory, env } = await fixture()
  try {
    const { project, script } = await copiedLauncher(directory)
    await writeFile(resolve(project, 'frontend/package.json'), JSON.stringify({ scripts: { build: 'node -e "process.exit(9)"' } }))
    const failedBuild = start([], env, script)
    assert.equal(await finish(failedBuild), 1)
    assert.match(failedBuild.output, /Frontend build failed/)
    await assertStopped(env.FISHLAB_PORT)
    await mkdir(resolve(project, 'frontend/node_modules/vite/bin'), { recursive: true })
    await writeFile(resolve(project, 'frontend/node_modules/vite/bin/vite.js'), 'process.exit(7)')
    const failedFrontend = start(['--dev'], env, script)
    assert.equal(await finish(failedFrontend), 1)
    assert.match(failedFrontend.output, /Development frontend stopped/)
    await assertStopped(env.FISHLAB_API_PORT)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
