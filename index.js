// Reasoning-only output-limit stops, handled through instructions.
//
// The model can spend its whole output budget (reasoning included) thinking and
// emit no visible text or tool call. This plugin never limits thinking. It
// (1) tells matching models their output budget and to work in visible steps,
// and (2) when the stop happens anyway, retries with the end of the lost
// reasoning so the model continues from it and makes its progress visible.
import { createUserMessage } from '@deepseek-ai/dsh-llm'

export const name = 'reasoning-retry'
export const inject = ['agents', 'sessions', 'goals', 'systemPrompt']

export const DEFAULTS = {
  enabled: true,
  maxRetries: 2,
  // Models (provider/id, case-insensitive) that get the step-by-step guidance.
  guidanceModels: 'deepseek',
  // Characters of the cut-off reasoning carried into the retry.
  reasoningTailChars: 4000,
  repetition: { ngram: 8, minCount: 5, minRatio: 0.25 },
  retryOnCompletedReasoningOnly: true,
  resumeGoalAfterRetry: true,
  blockGoalAfterRetries: true,
  notifyOnGiveUp: true,
  // Undo an output cap that a plugin once forced and that now sticks to the
  // session (the next request reuses the previous request's explicit values).
  restoreAdapterMaxTokens: true,
}

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

/** Whether an assistant message contains only reasoning blocks. */
export function isReasoningOnly(message) {
  if (!message || !Array.isArray(message.content) || message.content.length === 0) return false
  return message.content.every(block => block?.type === 'reasoning')
}

/** Count repeated word n-grams; used only to make the retry brief concrete. */
export function repetitionStats(text, ngram = 8) {
  const words = String(text ?? '').toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []
  if (words.length < ngram) return { max: 0, maxKey: '', ratio: 0, windows: 0, words: words.length }
  const counts = new Map()
  for (let i = 0; i + ngram <= words.length; i++) {
    const key = words.slice(i, i + ngram).join(' ')
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  let max = 0
  let maxKey = ''
  let repeated = 0
  for (const [key, count] of counts) {
    if (count > max) {
      max = count
      maxKey = key
    }
    if (count > 1) repeated++
  }
  return {
    max,
    maxKey,
    ratio: counts.size === 0 ? 0 : repeated / counts.size,
    windows: counts.size,
    words: words.length,
  }
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

/** Retry message: continue from the lost reasoning and make the progress visible. */
export function renderRetryMessage({ stage, maxRetries, maxTokens, reasoningTail, stats, minCount = 5 }) {
  const budget = Number.isInteger(maxTokens) ? ` (${maxTokens} tokens)` : ''
  const lines = [
    `<retry stage="${stage + 1}/${maxRetries}">`,
    `Your previous reply used its whole output budget${budget} on reasoning and produced no visible text or tool call, so that work was not saved.`,
    'Below is the end of that reasoning. Continue from it instead of starting over.',
    '<previous-reasoning-tail>',
    reasoningTail || '(no reasoning text was returned)',
    '</previous-reasoning-tail>',
    'Think as needed, but this time make progress visible within this reply: state your current conclusion or plan as text, or take the next step with a tool call. Leave the remaining analysis for later steps.',
  ]
  if (stats?.max >= minCount) {
    lines.push(`You repeated "${stats.maxKey}" ${stats.max} times. Settle that point (pick an option, or check it with a tool) and move on.`)
  }
  if (stage > 0) {
    lines.push('This is a repeated cut-off: split the work smaller. First write a short plan of the remaining steps as text, then do only the first one.')
  }
  lines.push('</retry>')
  return lines.join('\n')
}

function resolveConfig(input = {}) {
  const config = {
    ...DEFAULTS,
    ...input,
    repetition: { ...DEFAULTS.repetition, ...(input.repetition ?? {}) },
  }
  if (typeof config.enabled !== 'boolean') throw new Error('reasoning-retry: enabled must be a boolean')
  if (!Number.isInteger(config.maxRetries) || config.maxRetries < 0) {
    throw new Error('reasoning-retry: maxRetries must be a non-negative integer')
  }
  if (!Number.isInteger(config.reasoningTailChars) || config.reasoningTailChars < 0) {
    throw new Error('reasoning-retry: reasoningTailChars must be a non-negative integer')
  }
  if (typeof config.guidanceModels !== 'string') throw new Error('reasoning-retry: guidanceModels must be a string')
  if (!Number.isInteger(config.repetition.ngram) || config.repetition.ngram < 2) {
    throw new Error('reasoning-retry: repetition.ngram must be an integer >= 2')
  }
  return config
}

function textOf(message) {
  if (!message || !Array.isArray(message.content)) return ''
  return message.content.filter(block => block?.type === 'text').map(block => block.text).join('\n')
}

function truncate(text, max = 2000) {
  const value = String(text ?? '')
  return value.length <= max ? value : `${value.slice(0, max)}…`
}

function isRetrySource(source) {
  return source?.kind === 'reasoning-retry'
}

/** Sources that represent a new turn, not context inserted into one. */
function isSemanticUserSource(source) {
  const kind = source?.kind
  return kind === 'user' || kind === 'goal' || kind === 'relay'
    || kind === 'reasoning-retry' || kind === 'reasoning-retry-note'
}

export function apply(ctx, config = {}) {
  const cfg = resolveConfig(config)
  if (!cfg.enabled) return

  const states = new Map()

  function log(level, message) {
    const logger = ctx.logger
    if (typeof logger?.[level] === 'function') logger[level](`reasoning-retry: ${message}`)
  }

  /** Remember the last adapter-default output cap this session ran with. */
  function noteHeader(state, header) {
    if (header?.adapterDefaults?.maxTokens === true && Number.isInteger(header.config?.maxTokens)) {
      state.adapterMaxTokens = header.config.maxTokens
    }
  }

  function stateFor(agent) {
    let state = states.get(agent)
    if (state === undefined) {
      state = {
        lastAssistant: undefined,
        lastToolResults: [],
        lastUserSource: undefined,
        retryActive: false,
        retryTurn: undefined,
        retryPending: false,
        retryMessageId: undefined,
        stage: 0,
        pendingRetry: undefined,
      }
      // A reload can happen mid-turn; seed from the durable log so the retry turn is recognized.
      try {
        for (const event of agent.session?.snapshotEvents?.() ?? []) {
          switch (event.type) {
            case 'assistant/message':
              state.lastAssistant = {
                turn: event.data.turn,
                step: event.data.step,
                message: event.data.message,
              }
              break
            case 'tool/result':
              state.lastToolResults.push(textOf(event.data.message))
              while (state.lastToolResults.length > 3) state.lastToolResults.shift()
              break
            case 'request/header':
              noteHeader(state, event.data.header)
              break
            case 'user/message':
              // time-context and other context messages are inserted into a
              // turn; they must not overwrite the semantic user source.
              if (isSemanticUserSource(event.data.source)) state.lastUserSource = event.data.source
              break
            default:
              break
          }
        }
      } catch {
        // The session log is advisory for seeding; live events still arrive.
      }
      states.set(agent, state)
    }
    return state
  }

  function goalOf(agent) {
    try {
      return ctx.goals.get(agent)
    } catch {
      return undefined
    }
  }

  function resumeGoal(agent) {
    if (!cfg.resumeGoalAfterRetry) return
    const goal = goalOf(agent)
    if (goal === undefined || goal.phase !== 'active' || goal.activation !== 'disarmed') return
    try {
      ctx.goals.resume(agent, { id: goal.id, revision: goal.revision })
      log('info', `resumed goal ${goal.id} after a successful retry`)
    } catch (error) {
      log('warn', `could not resume goal ${goal.id}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  function blockGoal(agent, reason) {
    if (!cfg.blockGoalAfterRetries) return
    const goal = goalOf(agent)
    if (goal === undefined || goal.phase !== 'active') return
    try {
      ctx.goals.block(agent, { id: goal.id, revision: goal.revision }, {
        code: 'reasoning-retry-exhausted',
        message: reason,
      })
      log('warn', `blocked goal ${goal.id}: ${reason}`)
    } catch (error) {
      log('warn', `could not block goal ${goal.id}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  function notifyGiveUp(agent, reason) {
    if (!cfg.notifyOnGiveUp) return
    const message = createUserMessage({
      content: [{ type: 'text', text: reason }],
      source: { kind: 'reasoning-retry-note', form: 'notice', summary: 'retry exhausted' },
    })
    try {
      if (agent.status === 'idle') agent.followup(message)
      else agent.inject(message)
    } catch (error) {
      log('warn', `could not queue the give-up note: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  function giveUp(agent, reason) {
    const state = stateFor(agent)
    state.retryActive = false
    state.retryPending = false
    state.pendingRetry = undefined
    state.retryTurn = undefined
    log('warn', reason)
    blockGoal(agent, reason)
    notifyGiveUp(agent, `The previous attempt hit the output limit while only reasoning. I retried ${cfg.maxRetries} time(s), but the model still produced no visible text or tool call. Please review the last tool result and send a more specific instruction, or resume the goal.`)
  }

  /** The request's output cap, when the session recorded one. */
  function maxTokensOf(agent) {
    try {
      const value = agent.session?.requestHeader?.()?.config?.maxTokens
      return Number.isInteger(value) ? value : undefined
    } catch { return undefined }
  }

  async function queueRetry(agent, state, stage, last) {
    if (state.retryPending) return
    if (agent.status !== 'idle') return
    if (agent.inbox.nextTurn.length > 0 || agent.inbox.nextStep.length > 0) return
    const goal = goalOf(agent)
    if (goal !== undefined && (goal.phase === 'paused' || goal.phase === 'blocked')) return
    if (stage >= cfg.maxRetries) {
      giveUp(agent, `Output limit hit ${cfg.maxRetries} time(s) while only reasoning; no visible text or tool call.`)
      return
    }
    const reasoning = last.message.content
      .filter(block => block?.type === 'reasoning')
      .map(block => block.text ?? '')
      .join('\n')
    const message = createUserMessage({
      content: [{
        type: 'text',
        text: renderRetryMessage({
          stage,
          maxRetries: cfg.maxRetries,
          maxTokens: maxTokensOf(agent),
          reasoningTail: cfg.reasoningTailChars > 0 ? reasoning.slice(-cfg.reasoningTailChars) : '',
          stats: repetitionStats(reasoning, cfg.repetition.ngram),
          minCount: cfg.repetition.minCount,
        }),
      }],
      source: { kind: 'reasoning-retry', form: 'notice', summary: `retry ${stage + 1}/${cfg.maxRetries}` },
    })
    state.retryPending = true
    state.retryActive = false
    state.stage = stage
    state.retryMessageId = message.id
    try {
      agent.followup(message)
      log('info', `queued retry ${stage + 1}/${cfg.maxRetries}`)
    } catch (error) {
      state.retryPending = false
      log('warn', `could not queue retry: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  function handleTurnEnd(agent, state, data) {
    const last = state.lastAssistant
    const reasoningOnly = last !== undefined && last.turn === data.turn && isReasoningOnly(last.message)
    const wasRetry = state.retryActive || isRetrySource(state.lastUserSource)
    if (state.retryActive) {
      state.retryActive = false
      state.retryPending = false
      state.retryTurn = undefined
    }
    if (!reasoningOnly) {
      state.stage = 0
      state.pendingRetry = undefined
      // Any visible progress ends the retry ladder. If the goal-round-driver
      // disarmed the goal on a previous max-tokens stop, rearm it so the goal
      // continues; resumeGoal() is a no-op when the goal is already armed.
      resumeGoal(agent)
      return
    }
    if (data.reason.kind === 'aborted' || data.reason.kind === 'error') {
      state.stage = 0
      state.pendingRetry = undefined
      return
    }
    const goal = goalOf(agent)
    const driverWillContinue = goal !== undefined && goal.phase === 'active' && goal.activation === 'armed'
    if (data.reason.kind === 'completed') {
      if (!cfg.retryOnCompletedReasoningOnly) {
        state.stage = 0
        state.pendingRetry = undefined
        return
      }
      // A non-retry completed turn with an armed goal is already followed by a
      // goal round; let the goal-round-driver own that retry.
      if (!wasRetry && driverWillContinue) {
        state.stage = 0
        state.pendingRetry = undefined
        return
      }
    }
    if (data.reason.kind !== 'max-tokens' && data.reason.kind !== 'completed') {
      state.stage = 0
      state.pendingRetry = undefined
      return
    }
    const stage = wasRetry ? state.stage + 1 : 0
    if (stage >= cfg.maxRetries) {
      giveUp(agent, `Output limit hit ${cfg.maxRetries} time(s) while only reasoning; no visible text or tool call.`)
      return
    }
    state.stage = stage
    state.pendingRetry = { stage, last, goal }
  }

  ctx.on('session/event', (session, event) => {
    const agent = ctx.agents.get(session.id)
    if (agent === undefined || agent.session !== session) return
    const state = stateFor(agent)
    switch (event.type) {
      case 'assistant/message':
        state.lastAssistant = { turn: event.data.turn, step: event.data.step, message: event.data.message }
        return
      case 'tool/result':
        state.lastToolResults.push(textOf(event.data.message))
        while (state.lastToolResults.length > 3) state.lastToolResults.shift()
        return
      case 'request/header':
        noteHeader(state, event.data.header)
        return
      case 'user/message': {
        const source = event.data.source
        if (!isSemanticUserSource(source)) return
        state.lastUserSource = source
        if (!isRetrySource(source)) {
          state.stage = 0
          state.retryActive = false
          state.retryPending = false
          state.pendingRetry = undefined
        }
        return
      }
      case 'turn/start':
        // turn/start is appended before the admitted user/message, so the
        // queued retry reservation is the reliable signal here.
        state.retryActive = state.retryPending === true
        state.retryTurn = state.retryActive ? event.data.turn : undefined
        return
      case 'turn/end':
        handleTurnEnd(agent, state, event.data)
        return
      default:
        return
    }
  })

  ctx.on('agent/status', ({ agent, status }) => {
    if (status !== 'idle') return
    const state = states.get(agent)
    if (state?.pendingRetry === undefined) return
    const pending = state.pendingRetry
    state.pendingRetry = undefined
    void queueRetry(agent, state, pending.stage, pending.last)
  })

  // Standing guidance for matching models: tell them their output budget and to
  // work in visible steps, so a long reasoning pass is less likely to be lost.
  const guidancePattern = cfg.guidanceModels ? new RegExp(cfg.guidanceModels, 'i') : undefined
  if (guidancePattern !== undefined && ctx.systemPrompt?.context) {
    ctx.effect(() => ctx.systemPrompt.context({
      name: 'reasoning-budget',
      order: 130,
      text: ({ scope }) => {
        const agent = scope
        const route = agent?.session?.requestHeader?.()?.config ?? agent?.options
        const key = `${route?.provider ?? ''}/${route?.model ?? ''}`
        if (!route?.model || !guidancePattern.test(key)) return ''
        return renderGuidance(maxTokensOf(agent))
      },
    }))
  }

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

  ctx.on('agent/inbox/discarded', ({ agent, message }) => {
    const state = states.get(agent)
    if (state?.retryPending === true && message.id === state.retryMessageId) {
      state.retryPending = false
      state.retryActive = false
      state.retryTurn = undefined
    }
  })

  ctx.on('agent/created', ({ agent }) => { stateFor(agent) })
  ctx.on('agent/disposed', ({ agent }) => { states.delete(agent) })
}
