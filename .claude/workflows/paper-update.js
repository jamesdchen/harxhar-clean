export const meta = {
  name: 'paper-update',
  description: 'Paper text update per the paper-prose skill: optional draft, code checks, three parallel auditors, fixes, round-2 audit, compile',
  whenToUse: 'A section of the harxhar paper was drafted or must be drafted, and has to pass paper-prose Step 3 before the coordinator commits. args: {file, base, range?, task?, branch?, macroFiles?, facts?, regen?, tag?}',
  phases: [
    { title: 'Draft', detail: 'optional drafting agent (paper-prose branch)' },
    { title: 'Checks', detail: 'prose_check lint + diff; base text saved for the scope auditor' },
    { title: 'Audit', detail: 'claims, sentences and scope auditors in parallel' },
    { title: 'Fix', detail: 'confirmed findings as logged named operations' },
    { title: 'Audit 2', detail: 'only slices with confirmed gate violations' },
    { title: 'Compile', detail: 'regen.py --compile' },
  ],
}

// 2026-10-30 (the author: "make it a workflow"): the text side of a paper update, the paper-prose skill's Step 3
// (.claude/skills/paper-prose/SKILL.md) run deterministically.  Out of scope: cluster work, git commit, the Overleaf push
// (the coordinator and the post-commit hook).  args:
//   file        repo-relative .tex (or generator) path                         (required)
//   base        git revision before the edit, for the diff and the scope auditor (required)
//   range       target line range "START-END" (required unless task is given; the drafter returns it)
//   task        drafting brief; omitted = the text is already written, audit only
//   branch      paper-prose branch letter(s) for the drafter, e.g. "A" or "A, C"
//   macroFiles  generated macro files the range uses
//   facts       facts only the coordinator's session holds (e.g. a timeline an auditor needs)
//   regen       generators to rerun before the compile (regen.py --only), e.g. "ablations,macro_summary"
//   tag         a short unique tag for this run's temp files
const A = args || {}
if (!A.file || !A.base) throw new Error('args.file and args.base are required')
if (!A.range && !A.task) throw new Error('args.range is required when no task is given')
const REPO = 'C:\\Users\\james\\CC Allowed\\harxhar-0dte-professor'
const TMP = 'C:\\Users\\james\\AppData\\Local\\Temp\\paper_update'
const TAG = A.tag || 'run'
const EDITLOG = `${TMP}\\${TAG}_editlog.jsonl`
const BASEFILE = `${TMP}\\${TAG}_base_${A.file.split('/').pop()}`
const MACROS = (A.macroFiles || []).join(', ') || 'the writeup/generated/*.tex files the range uses'
const FACTS = A.facts ? `\nFacts from the coordinator's session record (not in the repo; treat as given):\n${A.facts}\n` : ''
const COMMON = `Repo: ${REPO} (Windows; the Bash tool is Git Bash; prose_check runs with C:\\Users\\james\\miniconda3\\python.exe, generators with C:\\Users\\james\\miniconda3\\envs\\285J\\python.exe). Do not commit, push or SSH. The PC is short on memory: nothing heavy locally. In Git Bash heredocs backslashes collapse; write files with the Write tool.`

const DRAFT_SCHEMA = {
  type: 'object',
  properties: { range: { type: 'string' }, notes: { type: 'string' } },
  required: ['range', 'notes'],
}
const ROW = {
  type: 'object',
  properties: {
    id: { type: 'string' }, file: { type: 'string' }, line: { type: 'number' }, severity: { type: 'string' },
    candidate: { type: 'boolean' }, match: { type: 'string' }, message: { type: 'string' },
  },
  required: ['id', 'line', 'severity', 'candidate', 'message'],
}
const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    lint_exit: { type: 'number' }, diff_exit: { type: 'number' }, base_saved: { type: 'boolean' },
    rows: { type: 'array', items: ROW },
  },
  required: ['lint_exit', 'diff_exit', 'base_saved', 'rows'],
}
const VIOL = {
  type: 'object',
  properties: {
    id: { type: 'string' }, file: { type: 'string' }, line: { type: 'number' }, quote: { type: 'string' },
    severity: { type: 'string' }, confirmed: { type: 'boolean' }, why: { type: 'string' }, fix: { type: 'string' },
  },
  required: ['id', 'line', 'severity', 'confirmed', 'why', 'fix'],
}
const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    slice: { type: 'string' }, clean: { type: 'boolean' },
    violations: { type: 'array', items: VIOL },
    dismissed: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, line: { type: 'number' }, match: { type: 'string' }, reason: { type: 'string' } }, required: ['id', 'reason'] } },
  },
  required: ['slice', 'clean', 'violations', 'dismissed'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    range: { type: 'string' },
    applied: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, line: { type: 'number' }, op: { type: 'string' } }, required: ['id', 'op'] } },
    not_applied: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, line: { type: 'number' }, reason: { type: 'string' } }, required: ['id', 'reason'] } },
  },
  required: ['range', 'applied', 'not_applied'],
}
const COMPILE_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' }, pages: { type: 'number' }, errors: { type: 'number' }, undefined_refs: { type: 'number' },
    macros_changed: { type: 'string' }, summary: { type: 'string' },
  },
  required: ['ok', 'errors', 'undefined_refs', 'summary'],
}

const SLICES = [
  { key: 'claims', type: 'prose-auditor-claims', has: id => ['V2', 'V4', 'V5', 'V6', 'V7'].includes(id) },
  { key: 'sentences', type: 'prose-auditor-sentences', has: id => ['V1', 'V3', 'V8', 'V9', 'V10', 'V11'].includes(id) || id.startsWith('AP-') },
  { key: 'scope', type: 'prose-auditor-scope', has: id => ['V12', 'V13', 'V14', 'V15', 'V16'].includes(id) },
]

let range = A.range
if (A.task) {
  phase('Draft')
  const d = await agent(
    `Draft or update text of the harxhar JFEc paper. Use the paper-prose skill: read .claude/skills/paper-prose/SKILL.md, then the pattern files of branch ${A.branch || 'A'}. You are a drafting subagent: run Step 0-2 and stop; the audit (Step 3) runs after you.\nTask: ${A.task}\nFile: ${A.file}${range ? ` (target range ${range})` : ''}. Base revision: ${A.base}. Edit only the target. Log each edit as one JSON line {"file": ..., "line": ..., "op": "<named operation>", "rule_id": "<pattern id>"} to ${EDITLOG} (create the directory if needed). Every number from a generated macro (${MACROS}).${FACTS}\n${COMMON}\nReturn the final target line range as "START-END" and short notes (doubts for the auditors).`,
    { label: 'draft', phase: 'Draft', schema: DRAFT_SCHEMA },
  )
  range = d.range
  log(`drafted ${A.file}:${range}`)
}

async function checks(round) {
  return agent(
    `Run the paper-prose Step 3(a) code checks and save the base text. ${COMMON}\n1. mkdir -p the directory of ${BASEFILE}; git show ${A.base}:${A.file} > ${BASEFILE} (the whole base file, for the scope auditor; set base_saved).\n2. C:\\Users\\james\\miniconda3\\python.exe writeup/style/prose_check.py lint ${A.file}:${range} --json\n3. C:\\Users\\james\\miniconda3\\python.exe writeup/style/prose_check.py diff --base ${A.base} --target ${A.file}:${range} [--log ${EDITLOG} only if that file exists] --json ${A.file}\nReturn both exit codes and every row of both JSON outputs except severity "rate" (field names as in the JSON: id, file, line, severity, candidate, match, message). Round ${round}.`,
    { label: `checks:${round}`, phase: 'Checks', schema: CHECK_SCHEMA, effort: 'low' },
  )
}

function auditPrompt(s, rows, round, fixed) {
  const hits = rows.filter(r => s.has(r.id))
  const hitText = hits.length
    ? hits.map(r => `- ${r.id} line ${r.line} ${r.severity} ${r.candidate ? 'candidate' : 'CONFIRMED BY CODE'}: ${r.match || ''} | ${r.message}`).join('\n')
    : 'none'
  const scopeExtra = s.key === 'scope'
    ? `\nBase revision ${A.base}; the whole base file is saved at ${BASEFILE} (you cannot run git). Edit log: ${EDITLOG} (if it exists). No layer is locked unless writeup/layer_state.json says so.`
    : ''
  const fixedText = fixed && fixed.length ? `\nRound-1 fixes applied (named operations): ${fixed.map(f => `${f.id} line ${f.line ?? '?'}: ${f.op}`).join('; ')}. Judge the whole range afresh.` : ''
  return `Step 3(b) of the paper-prose skill, ${s.key} slice, audit round ${round} of at most 2. Repo ${REPO}. Audit ${A.file} lines ${range}.${scopeExtra}\nMacro files: ${MACROS}.${FACTS}${fixedText}\nCode hits of your slice (prose_check):\n${hitText}\nReturn your JSON as your instructions specify.`
}

async function audit(slices, rows, round, fixed) {
  const out = await parallel(slices.map(s => () =>
    agent(auditPrompt(s, rows, round, fixed), { label: `audit${round}:${s.key}`, phase: round === 1 ? 'Audit' : 'Audit 2', agentType: s.type, schema: AUDIT_SCHEMA })))
  return out.map((r, i) => r || { slice: slices[i].key, clean: false, violations: [], dismissed: [], failed: true })
}

async function fix(violations, round) {
  const list = violations.map(v => `- ${v.id} (${v.severity}) ${A.file} line ${v.line}: "${v.quote || ''}" -- ${v.why} FIX: ${v.fix}`).join('\n')
  return agent(
    `Apply audit findings to text of the harxhar JFEc paper, as the paper-prose skill's Step 3(c)(1) says: each confirmed violation fixed with its one named operation (V12), each logged as one JSON line {"file": ..., "line": ..., "op": ..., "rule_id": ...} appended to ${EDITLOG}. Edit only ${A.file} within ${range} (a fix that would need text outside it goes to not_applied). Keep every number a macro; do not change a claim beyond what the finding asks.${FACTS}\n${COMMON}\nFindings (audit round ${round}):\n${list}\nThen rerun C:\\Users\\james\\miniconda3\\python.exe writeup/style/prose_check.py lint ${A.file}:<new range> and fix any new non-candidate gate hit you introduced; rewrap lines over 120 characters. Return the new target range, the operations applied and anything not applied with the reason.`,
    { label: `fix:${round}`, phase: 'Fix', schema: FIX_SCHEMA },
  )
}

phase('Checks')
const c1 = await checks(1)
if (!c1.base_saved) log('warning: base text not saved; the scope auditor works without it')
phase('Audit')
const r1 = await audit(SLICES, c1.rows, 1, null)
const count = rs => Object.fromEntries(rs.map(r => [r.slice, {
  gate: r.violations.filter(v => v.confirmed && v.severity.startsWith('gate')).length,
  review: r.violations.filter(v => v.confirmed && !v.severity.startsWith('gate')).length,
  unconfirmed: r.violations.filter(v => !v.confirmed).length,
  dismissed: r.dismissed.length,
}]))
const summary = { round1: count(r1) }
log(`round 1: ${JSON.stringify(summary.round1)}`)

const conf1 = r1.flatMap(r => r.violations.filter(v => v.confirmed))
const open = r1.flatMap(r => r.violations.filter(v => !v.confirmed).map(v => ({ ...v, slice: r.slice, round: 1 })))
let applied = []
let r2 = []
if (conf1.length) {
  phase('Fix')
  const f1 = await fix(conf1, 1)
  if (f1) { range = f1.range; applied = applied.concat(f1.applied); f1.not_applied.forEach(n => open.push({ ...n, round: 1, note: 'not applied' })) }
  const gateSlices = SLICES.filter(s => r1.find(r => r.slice === s.key && r.violations.some(v => v.confirmed && v.severity.startsWith('gate'))))
  if (gateSlices.length) {
    phase('Checks')
    const c2 = await checks(2)
    phase('Audit 2')
    r2 = await audit(gateSlices, c2.rows, 2, applied)
    summary.round2 = count(r2)
    log(`round 2 (${gateSlices.map(s => s.key).join(', ')}): ${JSON.stringify(summary.round2)}`)
    const conf2 = r2.flatMap(r => r.violations.filter(v => v.confirmed))
    r2.forEach(r => r.violations.filter(v => !v.confirmed).forEach(v => open.push({ ...v, slice: r.slice, round: 2 })))
    if (conf2.length) {
      phase('Fix')
      const f2 = await fix(conf2, 2)
      if (f2) { range = f2.range; applied = applied.concat(f2.applied); f2.not_applied.forEach(n => open.push({ ...n, round: 2, note: 'not applied' })) }
      // gate violations confirmed in round 2 are fixed but not re-audited (two rounds at most): listed for the coordinator
      conf2.filter(v => v.severity.startsWith('gate')).forEach(v => open.push({ ...v, round: 2, note: 'gate fixed after the last round; not re-audited' }))
    }
  }
}

phase('Checks')
const cf = await checks(3)
const gateLeft = cf.rows.filter(r => r.severity === 'gate' && !r.candidate)
phase('Compile')
const comp = await agent(
  `Compile-check the harxhar paper. ${COMMON}\nRun from the repo root with REGEN_PYTHON=C:\\Users\\james\\miniconda3\\envs\\285J\\python.exe: C:\\Users\\james\\miniconda3\\python.exe writeup/tools/regen.py ${A.regen ? `--only ${A.regen} --compile` : '--no-run --compile'}. Report the compile line (pages, errors, undefined references/citations/control sequences) and the "macros vs HEAD" line; ok = 0 errors and 0 undefined of any kind.`,
  { label: 'compile', phase: 'Compile', schema: COMPILE_SCHEMA, effort: 'low' },
)

return {
  file: A.file,
  range,
  base: A.base,
  editlog: EDITLOG,
  audit: summary,
  applied,
  open,
  dismissed: r1.concat(r2).flatMap(r => r.dismissed.map(d => ({ ...d, slice: r.slice }))),
  final_code_gate_hits: gateLeft,
  compile: comp,
  ready_to_commit: gateLeft.length === 0 && open.length === 0 && !!(comp && comp.ok),
}
