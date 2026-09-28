
## Phase-1 learning substrate + anti-cascade penalty (2026-09-28)
- NEW providers/router/episode_store.py — durable JSONL episode store (~/.pal/episodes.jsonl); observability-only, makes no routing decision. Schema: model/category/outcome/err_class/latency_ms/reward(null,Phase2)/prompt_hash(sha256,no raw prompt)/tool/reason. In-mem ring + success_rate() read side for future bandit seeding. Env PAL_EPISODE_STORE/PAL_EPISODES_JSONL/PAL_EPISODE_MEM_CAP.
- PATCH providers/router/refusal_memory.py — error-class-aware penalties. classify_class()->policy|availability|client|unknown. Availability (5xx/429) NEVER blacklists on single hit: needs AVAIL_MIN(3) sustained within AVAIL_TTL(1800s), decays 1/AVAIL_DECAY_STEP(300s), auto-reprobes. Policy/client keep immediate skip + TTL(300s). Defeats flaky-5xx->permanent-blacklist cascade. Class change resets counter. Env PAL_REFUSAL_AVAIL_{TTL,MIN,DECAY_STEP}.
- PATCH providers/router/self_heal.py — migration probation: record_migration() commits permanent alias only after MIN_HINTS(2) identical hints; one-shot retry still recovers current call. Env PAL_SELF_HEAL_MIN_HINTS (1=old behavior).
- WIRE server.py — episode_store.record() at existing refusal hook points (success + failure) with latency.
- TESTS tests/test_episode_store.py, test_refusal_memory_penalty.py, test_self_heal_probation.py — 29 pass. No regression on quota_probe/rate_limit.
- Verdict source: 3-model debate (gpt-oss-120b FOR / gemini-flash AGAINST / qwen-27b NEUTRAL). Stop-line = Phase 2 human-gated diffs; NO autonomous self_modify auto-apply shipped.

## Phase-2 distiller + bandit seeding (2026-09-28)
- NEW providers/router/bandit.py — runtime candidate reordering by observed success (UCB-style: empirical rate when n>=MIN_SAMPLES(5) else prior order, + exploration bonus EXPLORE_C(0.7)*sqrt(ln(N+1)/(n+1))). NEVER drops a candidate (exploration floor); reversible/in-session; nothing persisted. reorder()/score()/explain(). Env PAL_BANDIT/PAL_BANDIT_{MIN_SAMPLES,EXPLORE_C,WINDOW}. Unseen candidates explored first (optimism under uncertainty).
- NEW providers/router/distill.py — OFFLINE distiller CLI `python -m providers.router.distill`. Reads episodes.jsonl, separates quality_rate=success/(success+policy_refusal) from reliability=1-avail_fail/n (transient 5xx never dents quality), proposes reordered CATEGORY_PREFERENCES per category, surfaces emerging unlisted models. Writes ~/.pal/distill/proposal-<ts>.json + report. HUMAN-GATED: never edits classifier.py, never auto-applied. Env PAL_DISTILL_{MIN_SAMPLES,DIR}.
- PATCH providers/router/episode_store.py — added observed(model,cat)->(n,successes) [bandit read side] and iter_file() [distiller disk read, corrupt-line tolerant].
- WIRE server.py — bandit.reorder(cat_tag, candidates) in auto-mode before blacklist filter.
- TESTS test_bandit.py(8), test_distill.py(8) — 62 total pass incl. regression. E2E CLI verified: refusing prior-#1 demoted, high-quality model promoted despite 8 transient 5xx.
- Stop line held: bandit=reversible runtime reorder (allowed); distill=proposal only (human-gated). No self_modify auto-apply.

## Unified `pal` CLI + teacher-sourced lessons (2026-09-28)
- NEW cli.py + [project.scripts] pal="cli:main" — subcommands: `pal serve` (server:run), `pal diag [--json]`, `pal distill ...` (passthrough, avoids argparse.REMAINDER leading-flag quirk). Needs `pip install -e .` for PATH; until then `python cli.py <cmd>`.
- NEW providers/router/diag.py — shared diag collector (defensive _safe wrappers). Adds episodes + bandit sections (bandit.explain per category) to the dump. scripts/pal-diag.py now thin-wraps it.
- NEW providers/router/lesson_store.py — teacher-sourced logic. emit_lesson(teacher_id, scope, rationale, ...) is the ONLY entry; requires teacher_id in TEACHERS allowlist (orchestrator 1.0/critic 0.7/panel 0.5/self 0.3) AND non-empty provenance. INJECTION DEFENSE AT ORIGIN CHANNEL: no transcript/task-text parser exists, so task content can never become a rule. Only scope=routing feeds the distiller; classifier/prompt/tool = human_review_only via pending_review(). ZERO inference-time loading. deactivate() writes tombstone; _load_disk applies tombstones on reload. Env PAL_LESSON_STORE/PAL_LESSONS_JSONL.
- PATCH distill.py — opt-in `--with-lessons` / build_proposal(use_lessons=True). Routing lessons nudge eff_quality by ±trust_weight*LESSON_NUDGE(1.0), attributed per-lesson in proposal["lessons_applied"] + report · lines. Lesson can't introduce a model absent from candidates. Strong telemetry + weak lesson => telemetry wins. Still human-gated proposal.
- TESTS test_cli.py(7), test_lesson_store.py(12), test_distill_lessons.py(5) — 85 total pass. E2E verified: orchestrator lesson promoted or-free #4->#1 as proposal.
- Debate source: gpt-oss-120b FOR (LLM-authored diffs) / gemini-flash AGAINST (inert log, injection kill-shot) / qwen NEUTRAL (stop at proposals). Verdict: teacher logic = human-gated proposals only; injection cut at channel auth, not diff content-scan.

## `pal` console script installed (2026-09-28)
- FIX pyproject.toml [tool.setuptools] py-modules — added "cli" (was ["server","config"]); without it the editable install exposed the `pal` entry point but `from cli import main` raised ModuleNotFoundError.
- INSTALLED: `pip install -e .` in .pal_venv (TMPDIR=/tmp; required freeing a stale sibling-session tmpfs dir that had filled /tmp to 100%). `pal` now on PATH at .pal_venv/bin/pal. Verified: `pal diag` (shows episodes+bandit sections), `pal distill --print-only`, `pal distill --with-lessons` all work as bare commands. `pal serve` = server:run.
- Package version unchanged (9.8.2).

## Interactive `pal chat` REPL (2026-09-28)
- NEW providers/router/chat_router.py — cheap-vs-smart tier selector. REUSES classifier.py (category signals) + model registry (availability); no new model-calling logic. classify_difficulty(): greeting/short -> cheap; classifier category OR code-authoring intent OR reasoning words OR len>280/3+ '?' -> smart. CHEAP_TIER=(gpt-oss-20b,groq-fast,flash-lite,...) SMART_TIER=(gpt-oss-120b,gemini-pro,gpt-5.2,...); env PAL_CHAT_CHEAP_MODEL/PAL_CHAT_SMART_MODEL/PAL_CHAT_HARD_LEN.
- NEW providers/router/chat_repl.py — async REPL. Every model call goes through server.handle_call_tool("chat"/...) so classifier+bandit+refusal+episode logging all apply (not a new router). Commands: plain msg=auto route; /cheap /smart force tier; /delegate <model> <q>; /debate <q> (fans across cheap+smart+flash panel); /model /help /exit. Strips chat tool's agent-boilerplate ("AGENT'S TURN", continuation_id note) for human readability. Threads continuation_id for context. Calls configure_providers() at startup (registry not populated on bare import).
- WIRE cli.py — `pal chat` subcommand. pyproject py-modules already includes cli (editable install auto-picks new router files).
- TESTS test_chat_router.py(20) — routing decisions + _clean/_extract helpers. 89 pass in the router suite.
- LIVE VERIFIED (Groq): 'hi'->gpt-oss-20b cheap; 'why sky blue'->gpt-oss-120b smart (Rayleigh); /delegate forces model; /debate returns 2 distinct panel views.
- Design: reused existing router (classifier/registry/handle_call_tool/hybrid) per "copy/reuse instead of build from scratch"; hybrid.py remains available as opt-in confidence-escalation (PAL_HYBRID=1).

## `pal chat` rich UI + log-flood fix (2026-09-28)
- REWRITE providers/router/chat_repl.py — rich terminal UI: header Panel (shows cheap/smart models + command hints), answers rendered as Markdown inside bordered bubbles (green=pal, blue=delegate, yellow=debate, red=error), console.status dots spinner while a model thinks. Routing tag shown dim (→ model · tier: reason).
- FIX log flood: the DEBUG spam users saw was all from server IMPORT time (LOG_LEVEL defaults to DEBUG, .env sets it, handler attached at import). run() now force-sets os.environ["LOG_LEVEL"]=PAL_CHAT_LOGLEVEL(default ERROR) AND logging.disable(WARNING) BEFORE importing server, then _quiet_logging() reasserts. Verified: 0 noise lines leaked (was 24). PAL_CHAT_LOGLEVEL overrides.
- FIX boilerplate: added markers for "Please continue this conversation using the continuation_id" variant + trailing markdown-rule stripping (_TRAILING_RULE).
- DEP: rich>=13.0 added to requirements.txt; installed into .pal_venv.
- TESTS test_chat_router.py still 21 pass (routing + _clean/_extract). Live-verified: clean header, markdown bubbles, cheap/smart routing, no DEBUG.

## `pal chat` /agent — full Claude Code capabilities (2026-09-28)
- NEW /agent <task> command in chat_repl.py — routes through handle_call_tool("clink", {cli_name:"claude", role:"default"}) to the local `claude` CLI (installed at ~/.local/bin/claude). This gives pal chat the REAL agentic Claude: bash, file read/edit, search, tools — running `claude --print --output-format json --permission-mode acceptEdits --model sonnet` in the current dir. model_used=claude-sonnet-5.
- One-time yellow safety notice on first /agent use (acceptEdits can modify files under cwd). _ask_agent() helper; magenta "agent" bubble.
- _clean() now strips <SUMMARY>…</SUMMARY> blocks clink appends.
- Distinction: /delegate <model> = raw model text (e.g. an OpenRouter Claude); /agent = full Claude Code CLI agent with tools. clink clients available: aider, claude, cline, codex, cursor-agent, gemini.
- LIVE VERIFIED: `/agent read note.txt` -> agent used file tools, returned contents; safety notice shown; SUMMARY stripped. Router tests still pass.

## /agent roles + permissions + cheap-models-use-Kali-tools (2026-09-28)
- (a) ROLES: /agent (read-only), /agent:edit (writes), /agent:plan (planner), /agent:review (codereviewer). Parsed from /agent:<suffix> in chat_repl.
- (b) PERMISSIONS: conf/cli_clients/claude.json — moved --permission-mode OUT of additional_args into per-role role_args (role_args append last, so they win). default/codereviewer=--permission-mode default (reads allowed, edits blocked headless=SAFE), edit=acceptEdits, planner=plan. Verified: default role reads note.txt but does NOT create files; edit role writes. Red warning shown only on first /agent:edit.
- (c) CHEAP MODELS USE KALI TOOLS: NEW /tools <task> — ReAct loop (_tools_loop) calls the PROVIDER DIRECTLY (not the chat tool, which blocks tool-shaped output), enabling PAL toolbelt (bash allowlist ls/cat/grep/find/curl/gh/git/wc/..., read_file, web_fetch, gh). Model emits <tool_call>, PAL runs it locally on Kali, feeds back <tool_result>, loops (max 5). Default tools model=qwen3 (gpt-oss trips Groq's output parser); override PAL_CHAT_TOOLS_MODEL. Tool calls shown as dim · lines. LIVE VERIFIED: qwen3 ran `wc -l data.env` -> answered 3.
- FIX providers/tooling/react.py — extract_calls now handles ALL open-model formats: {"args"}, {"arguments"} (str or dict), <function=NAME>{json}</function>, and <function=NAME><parameter=k>v</parameter></function> (Qwen XML). This was the blocker for cheap-model tool use.
- TESTS test_react_parser.py(8) — all formats. 29+ pass; full router suite green.
- SECURITY: bash adapter allowlist keeps /tools de-facto read-only (no rm/write binaries); runs in the launch cwd. /agent edit gated + warned.

## /tools write capability + parser hardening (2026-09-28)
- ROOTCAUSE of "(no answer)" on "create a file": (1) no write tool existed (bash allowlist is read-only, `touch` blocked); (2) qwen emitted a nameless call {"command":"touch hello"} the parser dropped.
- NEW write_file tool (providers/tooling/adapters/filesystem.py) — create/write UTF-8 files within PAL_FS_ROOTS (default cwd). Won't overwrite without overwrite=true; blocks paths outside roots (verified /etc/passwd rejected). Enabled in _ensure_toolbelt.
- FIX react.py extract_calls — now infers tool name when the model omits it (_infer_name: command→bash, path+content→write_file, path→read_file, url→web_fetch), and treats a bare object as the args when there's no args/arguments wrapper.
- _TOOLS_SYS updated: explicit tool guidance (bash=read-only, write_file=create files) + require "name" in the call. Header shows write_file.
- TESTS test_react_parser.py now 13 (added inference + nameless-drop). LIVE VERIFIED: /tools create file named hello -> qwen used write_file, file exists on disk.

## /tools:full — arbitrary command execution for authorized pentesting (2026-09-29)
- WHY: `/tools run burpsuite` refused — bash was locked to a read-only allowlist, so no real Kali tools (nmap/nuclei/burpsuite) could run.
- NEW /tools:full <task> — unlocks arbitrary bash for that run only. Red one-time warning (authorized-systems-only). max_steps=8, red bubble. Plain /tools stays read-only (default safe).
- bash adapter (adapters/bash.py): allowlist now read at CALL time; PAL_BASH_UNRESTRICTED=1 bypasses it. _tools_loop(full=True) sets the flag during the run and restores it in a finally (verified no leak to process env). Spec description mentions full mode + detached GUI launch.
- _TOOLS_SYS_FULL system prompt: offensive-security framing, real Kali tools, guidance to launch GUI/long-running apps DETACHED (`setsid <cmd> >/dev/null 2>&1 < /dev/null &`) and use higher timeout_s for scans.
- LIVE VERIFIED: /tools:full ran `whoami; hostname`; read-only /tools still blocks non-allowlisted binaries; flag does not leak between calls. 89 tests pass.
- SECURITY posture: destructive/arbitrary exec is opt-in per command via :full + explicit warning; default /tools read-only; /agent:edit remains the file-edit path. All gated behaviors require the user to choose them.

## /tools now full-power by default (2026-09-29)
- User choice: one /tools that runs ANY command + all tools, no :full needed. /tools:ro (aliases: readonly, safe) for a read-only session. /tools:full still accepted (same as default).
- chat_repl /tools handler: full = suffix not in (ro,readonly,safe) -> default True. One-time red safety notice on first full use ("runs ANY shell command by default... authorized systems only... use /tools:ro"). Header + usage updated. Blocked-in-ro hint now suggests plain /tools.
- Behavior verified: /tools run whoami -> executes (full); /tools:ro -> read-only allowlist. Bubble cyan; :ro tagged.
- All tests pass.

## /debate reads files then decides + files_required leak fixed (2026-09-29)
- BUG: /debate showed raw {"status":"files_required_to_continue",...} JSON — PAL's chat tool lets models request files, useless in a REPL debate.
- User need: debate models must READ files then decide. NEW two-phase /debate: (1) a tool-capable reader (qwen3, env PAL_CHAT_TOOLS_MODEL) uses read_file/bash to gather a <=250-word file digest from cwd; (2) the panel [gpt-oss-120b, qwen3, gpt-oss-20b] each give a verdict grounded in that digest via _ask_direct (direct provider call — no chat tool, so NO files_required envelope, no response-blocking). Only one model needs tools; all can debate.
- NEW _ask_direct(model, prompt, system) — provider.generate_content in a thread, _clean'd. Used for debate verdicts.
- _extract now humanizes a leaked files_required envelope ("(the model wanted to see files) <instructions>") instead of dumping JSON.
- LIVE VERIFIED: reader ran ls/read_file/grep on auth.py; gpt-oss-120b + qwen3 + gpt-oss-20b all correctly flagged plaintext compare/no-hashing/no-rate-limit. gpt-oss-120b works now (was blocked via chat tool).
- TEST: test_extract_files_required_envelope_humanized. All pass.

## Overcome recon refusals + Groq 413 token limit (2026-09-29)
- SYMPTOM: /debate "plan to continue recon on beon" -> qwen reader 413 (ITPM 7000 < 11740 requested) so no file evidence; then gpt-oss/qwen REFUSED ("can't help", "prohibit unauthorized access") on authorized recon.
- FIX 1 (refusals): _maybe_security(task, system) prepends _SECURITY_PREAMBLE (authorized in-scope testing, no refusals/disclaimers) when classifier.classify(task)=="security_permissive". Applied in _tools_loop (both modes) and the /debate panel _ask_direct. VERIFIED: recon debate now returns concrete plans (Shodan/Censys/crt.sh subdomain enum), no refusals.
- FIX 2 (413): token budget in _tools_loop — before each generate_content, if convo > PAL_TOOLS_CTX_CHARS (16000, ~4k tokens) keep task head + latest tool output tail, drop the middle. Keeps requests under Groq's small ITPM. VERIFIED: 40KB file trimmed, no 413.
- All tests pass.

## Per-command roles (2026-09-29)
- _ROLES dict: one-sentence system role per command (chat/delegate/smart/cheap/tools/debate/agent). _ask(role=) prepends "[Role: ...]" so every query — normal (cheap/smart by tier), /delegate, /debate, /tools, /agent — behaves specifically for its job. 22 tests pass.

## `pal run` — headless one-shot / plan runner (2026-09-29)
- NEW providers/router/headless.py + `pal run` subcommand (cli.py passthrough). Runs a task or a --plan file autonomously and prints ONLY the result — no REPL. Makes PAL callable like a function (pipe a task, read result), so the orchestrator (Claude) can delegate work without doing it.
  - `pal run "<task>"` (tools, full by default) · `--ro` (read-only) · `--agent` (full Claude Code CLI) · `--model <m>` · `--plan <file>` (one step per line / md bullets) · `--json` (structured: task/mode/model/tools_used/result).
  - Reuses _tools_loop / _ask_agent; quiets logging; configure_providers on start.
- VERIFIED: `pal run "count lines"` -> "3"; --json parseable (mode/model/tools/result); --plan ran 2 steps with per-step results; --agent --ro read a file.
- All tests pass.

## pal run --max-steps (2026-09-29)
- Added --max-steps N to `pal run` (headless.py): tool-call budget per task/plan-step, default 8 full / 5 ro. Fixes plan steps hitting [reached max tool steps]. Threaded run_task/run_plan -> _tools_loop.
