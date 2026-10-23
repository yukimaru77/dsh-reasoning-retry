// Development only: link the DSH packages this plugin imports into node_modules,
// from a source checkout (packages/...) or an installed release (node_modules).
import { access, mkdir, rm, symlink } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const base = resolve(process.argv[2] ?? process.env.DSH_CHECKOUT ?? join(process.env.HOME, 'deepseek-harness'))
const candidates = [join(base, 'packages/llm/llm'), join(base, '@deepseek-ai/dsh-llm'), join(base, 'node_modules/@deepseek-ai/dsh-llm')]
let target
for (const candidate of candidates) {
  try { await access(join(candidate, 'package.json')); target = candidate; break } catch {}
}
if (!target) throw new Error(`@deepseek-ai/dsh-llm not found under ${base}`)
await mkdir(join(root, 'node_modules', '@deepseek-ai'), { recursive: true })
const destination = join(root, 'node_modules', '@deepseek-ai', 'dsh-llm')
await rm(destination, { force: true })
await symlink(target, destination)
console.log(`Linked @deepseek-ai/dsh-llm from ${target}`)
