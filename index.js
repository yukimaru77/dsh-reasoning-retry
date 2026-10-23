// Output-limit stops, handled by re-posing the input instead of limiting thought.
//
// When a turn ends because a reply hit its output token limit (reasoning counts
// toward it), this plugin removes the failed step from the model-visible history
// (logged surface replacement; the log keeps everything), asks the model to
// rewrite the input that led to it, and re-runs the step with the rewritten
// input. A second failure is rewritten again from the whole chain: the original
// input and reasoning, the rewritten input and its reasoning. The input is
// whatever the failed step received: a user or goal prompt, or tool results,
// which are kept as they are and get a framing note instead.
import { createUserMessage } from '@deepseek-ai/dsh-llm'

export const name = 'reasoning-retry'
export const inject = ['agents', 'sessions', 'goals', 'llm', 'systemPrompt', 'commands']

export const DEFAULTS = {
  enabled: true,
  maxRetries: 2,
  // Models (provider/model, case-insensitive) that get the standing step-by-step guidance.
  guidanceModels: 'deepseek',
  // Characters of each input and reasoning shown to the rewriter (head and tail kept).
  rewriteInputChars: 12000,
  rewriteReasoningChars: 12000,
  rewriter: { maxTokens: 16384, timeoutMs: 180000 },
  resumeGoalAfterRetry: true,
  blockGoalAfterRetries: true,
  notifyOnGiveUp: true,
  // Undo an output cap that a plugin once forced and that now sticks to the
  // session (the next request reuses the previous request's explicit values).
  restoreAdapterMaxTokens: true,
}

const RETRY_SOURCE = 'max-tokens-retry'
// Messages of earlier versions of this plugin, hidden by /clean-failed-steps.
const OLD_RETRY_SOURCES = new Set(['reasoning-retry', 'reasoning-retry-note'])
const MARKER_COMMAND = 'ai-hidden'
// Context the harness injects into a turn; it is not the input a step responds to.
const OMITTED_SOURCE = 'max-tokens-omitted'
const CONTEXT_SOURCES = new Set(['time-context', 'runtime-context', 'agent-instructions', 'skill-catalog', OMITTED_SOURCE])
// Stands in for removed history. A user/message (not an empty system/message) because only
// user messages may replace surface nodes outside a turn in the stored log format.
const OMITTED_TEXT = '[omitted: an earlier attempt that was cut off at the output limit]'

/**
 * The output cap a session last ran with as the adapter default, when its
 * current explicit cap is lower. DSH reuses the previous request's explicit
 * values, so a cap lowered once by a plugin otherwise sticks for good.
 */
export function staleMaxTokens(headers, proposed) {
  if (!Number.isInteger(proposed)) return undefined
  let adapterDefault
  for (const header of headers) {
    if (header?.adapterDefaults?.maxTokens === true && Number.isInteger(header.config?.maxTokens)) adapterDefault = header.config.maxTokens
  }
  return adapterDefault !== undefined && proposed < adapterDefault ? adapterDefault : undefined
}

/** Standing guidance: work in visible steps instead of one long reasoning pass. */
export function renderGuidance(maxTokens) {
  const budget = Number.isInteger(maxTokens) ? `${maxTokens} tokens` : 'limited'
  return [
    `Your output budget per reply is ${budget}, and your reasoning counts toward it.`,
    'Think as much as the next step needs, but do not try to solve the whole task in one reply.',
    'Work in steps: decide the next step, act on it (a tool call or visible text), and carry the rest of the analysis into later steps.',
    'For long analyses, write intermediate conclusions or a short plan as visible text (or in todo_write or a file) as you go, so progress survives a cut-off.',
  ].join('\n')
}

const blockText = (block) => {
  if (!block || typeof block !== 'object') return ''
  if (typeof block.text === 'string') return block.text
  if (block.type === 'tool-call') return `[tool call ${block.name ?? ''} ${JSON.stringify(block.arguments ?? block.input ?? {}).slice(0, 500)}]`
  if (block.type === 'image') return '[image]'
  return ''
}
const messageText = (message) => (Array.isArray(message?.content) ? message.content.map(blockText).filter(Boolean).join('\n') : '')
const reasoningOf = (message) => (Array.isArray(message?.content)
  ? message.content.filter(block => block?.type === 'reasoning').map(block => block.text ?? '').join('\n') : '')
const visibleOf = (message) => (Array.isArray(message?.content)
  ? message.content.filter(block => block?.type !== 'reasoning').map(blockText).filter(Boolean).join('\n') : '')

/** Keep the head and tail of long text. */
export function clip(text, max) {
  const value = String(text ?? '')
  if (value.length <= max) return value
  const head = Math.floor(max * 0.35)
  return `${value.slice(0, head)}\n… [${value.length - max} characters omitted] …\n${value.slice(-(max - head))}`
}

/**
 * The input of the step that produced `failedSeq`: the surface nodes after the
 * previous assistant message. Tool results make it a tool input; otherwise the
 * non-context user messages form a prompt input.
 * @param surface - visible surface events in order ({ seq, type, data }).
 */
export function stepInput(surface, failedSeq) {
  const index = surface.findIndex(event => event.seq === failedSeq)
  if (index < 0) return undefined
  const nodes = []
  for (let i = index - 1; i >= 0; i--) {
    const event = surface[i]
    if (event.type === 'assistant/message' || event.type === 'system/message') break
    nodes.unshift(event)
  }
  const tools = nodes.filter(event => event.type === 'tool/result')
  if (tools.length > 0) {
    return { kind: 'tool', text: tools.map(event => messageText(event.data.message)).join('\n\n---\n\n'), deleteFrom: failedSeq }
  }
  const prompts = nodes.filter(event => event.type === 'user/message' && !CONTEXT_SOURCES.has(event.data.source?.kind))
  if (prompts.length === 0) return { kind: 'tool', text: '', deleteFrom: failedSeq }
  return { kind: 'prompt', text: prompts.map(event => messageText(event.data)).join('\n\n'), deleteFrom: prompts[0].seq }
}

/**
 * Visible nodes to hide when cleaning a session: the last reply of every turn
 * that ended at the output limit, and earlier versions' retry messages.
 * @param surface - visible surface events in order.
 * @param events - the whole log (a turn's last reply is judged on it, not on what is still visible).
 */
export function failedNodes(surface, events) {
  const turnEnds = new Map()
  const lastReply = new Map()
  for (const event of events) {
    if (event.type === 'turn/end') turnEnds.set(event.data.turn, event.data.reason?.kind)
    else if (event.type === 'assistant/message') lastReply.set(event.data.turn, event.seq)
  }
  const replies = []
  const notices = []
  for (const event of surface) {
    if (event.type === 'assistant/message' && turnEnds.get(event.data.turn) === 'max-tokens' && lastReply.get(event.data.turn) === event.seq) {
      replies.push(event)
    } else if (event.type === 'user/message' && OLD_RETRY_SOURCES.has(event.data.source?.kind)) {
      notices.push(event)
    }
  }
  return { replies, notices }
}

const turnRange = (turns) => {
  const list = [...new Set(turns)].sort((a, b) => a - b)
  return list.length === 0 ? '' : list.length === 1 ? `turn ${list[0]}` : `turn ${list[0]}〜${list.at(-1)}`
}

/** The rewriter's request, built from every attempt so far. */
export function renderRewriteRequest(kind, attempts, { inputChars = 12000, reasoningChars = 12000 } = {}) {
  const lines = [
    'An AI coding agent is working on a task. Its reply hit the output token limit before it finished; its reasoning counts toward that limit. That reply has been removed and the step will be run again.',
    kind === 'prompt'
      ? 'Rewrite the input below so the agent can make real progress within one reply. Keep every fact, constraint and goal of the original input; change how it is posed. For example: split it into ordered steps and ask for the first one, state the decision to make now, settle what the reasoning already established, resolve the ambiguity it kept circling, or name the part of the problem to look at first.'
      : 'The input below is tool output, which stays as it is. Write a short note to place right after it that poses the next step well, so the agent can make real progress within one reply. For example: point to the part of the output that matters, state what was already established, state the decision to make now, or split the remaining work into ordered steps and ask for the first one.',
    'Do not tell the agent to think less, to stop thinking, or to answer briefly; change the way the work is posed. Write in the language of the original input. Output only the text of the new message, with no preface.',
    '',
  ]
  attempts.forEach((attempt, i) => {
    const tag = i === 0 && kind === 'tool' ? 'tool-output' : 'input'
    const label = i === 0 ? 'Original input' : `Rewritten input (attempt ${i + 1})`
    lines.push(`<attempt number="${i + 1}">`, `<${tag} label="${label}">`, clip(attempt.input, inputChars), `</${tag}>`)
    if (attempt.note) lines.push('<note-after-tool-output>', clip(attempt.note, inputChars), '</note-after-tool-output>')
    lines.push('<reasoning-until-cut-off>', clip(attempt.reasoning || '(no reasoning text was returned)', reasoningChars), '</reasoning-until-cut-off>')
    if (attempt.visible) lines.push('<partial-visible-output>', clip(attempt.visible, 4000), '</partial-visible-output>')
    lines.push('</attempt>')
  })
  if (attempts.length > 1) lines.push('', 'The earlier rewrites also hit the limit. Use what each attempt reasoned to pose the work differently again.')
  return lines.join('\n')
}

function resolveConfig(input = {}) {
  const config = { ...DEFAULTS, ...input, rewriter: { ...DEFAULTS.rewriter, ...(input.rewriter ?? {}) } }
  if (typeof config.enabled !== 'boolean') throw new Error('reasoning-retry: enabled must be a boolean')
  if (!Number.isInteger(config.maxRetries) || config.maxRetries < 0) throw new Error('reasoning-retry: maxRetries must be a non-negative integer')
  if (typeof config.guidanceModels !== 'string') throw new Error('reasoning-retry: guidanceModels must be a string')
  if (!Number.isInteger(config.rewriter.maxTokens) || config.rewriter.maxTokens < 1) throw new Error('reasoning-retry: rewriter.maxTokens must be a positive integer')
  return config
}

export function apply(ctx, config = {}) {
  const cfg = resolveConfig(config)
  if (!cfg.enabled) return
  const states = new Map()

  const log = (level, message) => { if (typeof ctx.logger?.[level] === 'function') ctx.logger[level](`reasoning-retry: ${message}`) }
  const goalOf = (agent) => { try { return ctx.goals.get(agent) } catch { return undefined } }
  const maxTokensOf = (agent) => {
    try {
      const value = agent.session?.requestHeader?.()?.config?.maxTokens
      return Number.isInteger(value) ? value : undefined
    } catch { return undefined }
  }

  function observe(state, event) {
    switch (event.type) {
      case 'assistant/message':
        state.lastAssistant = { seq: event.seq, turn: event.data.turn, message: event.data.message }
        return
      case 'request/header':
        if (event.data.header?.adapterDefaults?.maxTokens === true && Number.isInteger(event.data.header.config?.maxTokens)) {
          state.adapterMaxTokens = event.data.header.config.maxTokens
        }
        return
      case 'user/message': {
        const kind = event.data.source?.kind
        if (kind === RETRY_SOURCE) {
          state.retrySeq = event.seq
          state.inRetryTurn = true
        } else if (kind === 'user' || kind === 'goal' || kind === 'relay') {
          // New work from the user or the goal ends a ladder.
          state.ladder = undefined
          state.pending = false
        }
        return
      }
      default:
    }
  }

  function stateFor(agent) {
    let state = states.get(agent)
    if (!state) {
      state = { lastAssistant: undefined, retrySeq: undefined, inRetryTurn: false, ladder: undefined, pending: false, adapterMaxTokens: undefined }
      try {
        for (const event of agent.session?.snapshotEvents?.() ?? []) observe(state, event)
      } catch { /* the log only seeds state */ }
      state.inRetryTurn = false
      states.set(agent, state)
    }
    return state
  }

  function resumeGoal(agent) {
    if (!cfg.resumeGoalAfterRetry) return
    const goal = goalOf(agent)
    if (goal === undefined || goal.phase !== 'active' || goal.activation !== 'disarmed') return
    try { ctx.goals.resume(agent, { id: goal.id, revision: goal.revision }) } catch (error) { log('warn', `could not resume goal: ${error}`) }
  }

  function giveUp(agent, state) {
    state.ladder = undefined
    state.pending = false
    const reason = `The output limit was hit ${cfg.maxRetries + 1} times in a row; rewriting the input did not help.`
    log('warn', reason)
    const goal = goalOf(agent)
    if (cfg.blockGoalAfterRetries && goal !== undefined && goal.phase === 'active') {
      try { ctx.goals.block(agent, { id: goal.id, revision: goal.revision }, { code: 'reasoning-retry-exhausted', message: reason }) } catch (error) { log('warn', `could not block goal: ${error}`) }
    }
    if (cfg.notifyOnGiveUp) {
      const message = createUserMessage({
        content: [{ type: 'text', text: `${reason} Please review the last input and send a more specific instruction, or resume the goal.` }],
        source: { kind: 'reasoning-retry-note', form: 'notice', summary: 'retry exhausted' },
      })
      try { if (agent.status === 'idle') agent.followup(message); else agent.inject(message) } catch (error) { log('warn', `could not queue the note: ${error}`) }
    }
  }

  function visibleSurface(agent) {
    const session = agent.session
    return [...(session?.surface?.nodes ?? [])].map(seq => session.eventAt(seq)).filter(Boolean)
  }

  function handleTurnEnd(agent, state, data) {
    const wasRetry = state.inRetryTurn
    state.inRetryTurn = false
    if (data.reason?.kind !== 'max-tokens') {
      if (wasRetry && state.ladder) {
        // The rewritten input got through: the ladder is done.
        state.ladder = undefined
        resumeGoal(agent)
      }
      return
    }
    const failed = state.lastAssistant
    if (!failed || failed.turn !== data.turn) return
    const attempt = { failedSeq: failed.seq, reasoning: reasoningOf(failed.message), visible: visibleOf(failed.message) }
    if (wasRetry && state.ladder) {
      const previous = state.ladder.attempts.at(-1)
      state.ladder.attempts.push({
        ...attempt,
        input: state.ladder.kind === 'prompt' ? previous.next : state.ladder.attempts[0].input,
        note: state.ladder.kind === 'tool' ? previous.next : undefined,
        deleteFrom: state.retrySeq,
      })
    } else {
      const input = stepInput(visibleSurface(agent), failed.seq)
      if (!input) return
      state.ladder = { kind: input.kind, attempts: [{ ...attempt, input: input.text, deleteFrom: input.deleteFrom }] }
    }
    if (state.ladder.attempts.length > cfg.maxRetries) { giveUp(agent, state); return }
    state.pending = true
  }

  /** Replace [fromSeq..toSeq] in the model-visible history with a one-line placeholder (the log keeps it). */
  function removeRange(agent, fromSeq, toSeq) {
    const nodes = [...(agent.session.surface?.nodes ?? [])]
    let start = nodes.indexOf(fromSeq)
    const end = nodes.indexOf(toSeq)
    if (start < 0 || end < start) throw new Error(`range ${fromSeq}..${toSeq} is not visible`)
    // Fold into a placeholder right before, so repeated retries leave a single line.
    if (start > 0 && agent.session.eventAt(nodes[start - 1])?.data?.source?.kind === OMITTED_SOURCE) start -= 1
    fromSeq = nodes[start]
    agent.session.append('user/message', createUserMessage({ content: [{ type: 'text', text: OMITTED_TEXT }], source: { kind: OMITTED_SOURCE } }), {
      surfaceOp: { op: 'replace', startSeq: fromSeq, endSeq: toSeq },
      sourceEventSeqs: nodes.slice(start, end + 1),
    })
  }

  async function rewrite(agent, ladder) {
    const route = agent.session?.requestHeader?.()?.config ?? agent.options
    if (!route?.provider || !route?.model) return undefined
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(new Error('rewriter timeout')), cfg.rewriter.timeoutMs)
    try {
      const stream = ctx.llm.stream({
        provider: route.provider,
        model: route.model,
        maxTokens: cfg.rewriter.maxTokens,
        messages: [createUserMessage({
          content: [{ type: 'text', text: renderRewriteRequest(ladder.kind, ladder.attempts, { inputChars: cfg.rewriteInputChars, reasoningChars: cfg.rewriteReasoningChars }) }],
          source: { kind: 'reasoning-retry', form: 'notice', summary: 'rewrite request' },
        })],
        system: 'You re-pose inputs for an AI coding agent so that each step can be completed within one reply. Output only the new message text.',
        signal: controller.signal,
      })
      let text = ''
      for await (const chunk of stream) {
        if (chunk.type === 'text-delta') text += chunk.text
      }
      return text.trim() || undefined
    } catch (error) {
      log('warn', `rewriter failed: ${error instanceof Error ? error.message : String(error)}`)
      return undefined
    } finally { clearTimeout(timer) }
  }

  function fallback(kind, attempts) {
    const steps = 'Work on this in steps: write the ordered list of steps that remain as visible text, then carry out the first one; continue with the rest in later steps.'
    return kind === 'prompt' ? `${attempts[0].input}\n\n${steps}` : steps
  }

  /** A card in the transcript (a command result, never sent to the model). */
  async function mark(agent, text) {
    try {
      await ctx.commands.execute(agent, `/${MARKER_COMMAND} ${text}`, [], new AbortController().signal)
    } catch (error) {
      log('warn', `could not add the AI-hidden marker: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  ctx.effect(() => ctx.commands.register({
    name: MARKER_COMMAND,
    description: 'Marker: the part just above is hidden from the AI (added by reasoning-retry)',
    recordInput: false,
    handler: ({ rawInput }) => ({ kind: 'success', text: rawInput.trim() }),
  }))

  ctx.effect(() => ctx.commands.register({
    name: 'clean-failed-steps',
    description: 'Hide replies cut off at the output limit and old retry notices from the AI (the log keeps them)',
    async handler({ agent }) {
      if (agent.status !== 'idle') return { kind: 'error', text: 'Run this while the session is idle.' }
      const events = agent.session.snapshotEvents()
      const { replies, notices } = failedNodes(visibleSurface(agent), events)
      if (replies.length + notices.length === 0) return { kind: 'success', text: 'AIから見えている失敗した返答や古いやり直しの指示はありません。' }
      // One placeholder per run of adjacent hidden nodes.
      const hide = new Set([...replies, ...notices].map(e => e.seq))
      const runs = []
      let prevHidden = false
      for (const seq of agent.session.surface.nodes) {
        if (!hide.has(seq)) { prevHidden = false; continue }
        if (prevHidden) runs.at(-1)[1] = seq
        else runs.push([seq, seq])
        prevHidden = true
      }
      for (const [from, to] of runs) removeRange(agent, from, to)
      const stateNow = stateFor(agent)
      stateNow.ladder = undefined
      stateNow.pending = false
      return {
        kind: 'success',
        text: `🚫 AI非表示：出力上限で切れた返答 ${replies.length}件（${turnRange(replies.map(e => e.data.turn))}）と、古いやり直しの指示 ${notices.length}件を、AIから見えなくしました。画面とログには残っていますが、AIには送られません。`,
      }
    },
  }))

  async function retry(agent, state) {
    state.pending = false
    const ladder = state.ladder
    if (!ladder) return
    const goal = goalOf(agent)
    if (goal !== undefined && (goal.phase === 'paused' || goal.phase === 'blocked')) { state.ladder = undefined; return }
    const current = ladder.attempts.at(-1)
    const text = (await rewrite(agent, ladder)) ?? fallback(ladder.kind, ladder.attempts)
    if (agent.status !== 'idle' || state.ladder !== ladder) return
    try {
      removeRange(agent, current.deleteFrom, current.failedSeq)
    } catch (error) {
      log('warn', `could not remove the failed step: ${error instanceof Error ? error.message : String(error)}`)
      state.ladder = undefined
      return
    }
    current.next = text
    const hidden = ladder.attempts.length > 1 ? '直前のやり直しの入力と返答' : ladder.kind === 'prompt' ? '直前の入力と返答' : '直前の返答'
    await mark(agent, `🚫 AI非表示：${hidden}（turn ${state.lastAssistant?.turn ?? '?'}）は出力上限で切れたため、AIから見えなくしました。入力の出し方を書き換えて再実行します（${ladder.attempts.length}/${cfg.maxRetries}）。`)
    agent.followup(createUserMessage({
      content: [{ type: 'text', text }],
      source: { kind: RETRY_SOURCE, summary: `rewritten ${ladder.kind === 'prompt' ? 'input' : 'note'} ${ladder.attempts.length}/${cfg.maxRetries}` },
    }))
    log('info', `re-ran the step with a rewritten ${ladder.kind === 'prompt' ? 'input' : 'note'} (${ladder.attempts.length}/${cfg.maxRetries})`)
  }

  ctx.on('session/event', (session, event) => {
    const agent = ctx.agents.get(session.id)
    if (agent === undefined || agent.session !== session) return
    const state = stateFor(agent)
    observe(state, event)
    if (event.type === 'turn/end') handleTurnEnd(agent, state, event.data)
  })

  ctx.on('agent/status', ({ agent, status }) => {
    if (status !== 'idle') return
    const state = states.get(agent)
    if (state?.pending) void retry(agent, state)
  })

  if (cfg.restoreAdapterMaxTokens) {
    ctx.on('agent/request', async ({ agent }, next) => {
      const config = await next()
      const state = stateFor(agent)
      const restored = staleMaxTokens(state.adapterMaxTokens === undefined ? [] : [{ config: { maxTokens: state.adapterMaxTokens }, adapterDefaults: { maxTokens: true } }], config?.maxTokens)
      if (restored === undefined) return config
      log('info', `dropped a stale maxTokens ${config.maxTokens}; the adapter default (${restored}) applies again`)
      const { maxTokens: _stale, ...rest } = config
      return rest
    })
  }

  // Standing guidance for matching models: their output budget and working in visible steps.
  const guidancePattern = cfg.guidanceModels ? new RegExp(cfg.guidanceModels, 'i') : undefined
  if (guidancePattern !== undefined && ctx.systemPrompt?.context) {
    ctx.effect(() => ctx.systemPrompt.context({
      name: 'reasoning-budget',
      order: 130,
      text: ({ scope }) => {
        const route = scope?.session?.requestHeader?.()?.config ?? scope?.options
        if (!route?.model || !guidancePattern.test(`${route.provider ?? ''}/${route.model}`)) return ''
        return renderGuidance(maxTokensOf(scope))
      },
    }))
  }

  ctx.on('agent/created', ({ agent }) => { stateFor(agent) })
  ctx.on('agent/disposed', ({ agent }) => { states.delete(agent) })
}
