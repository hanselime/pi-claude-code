import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import * as http from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  cachedClaudeProcessCount,
  closeClaudeProcesses,
  discoverClaudeModels,
  rewriteInitialRequest,
  runClaudeRequest,
  stripUserEmail,
} from "../src/claude-cli.ts";
import { createClaudeCodeProvider } from "../src/provider.ts";

test("stabilizes Claude Code's native replay reminders and cache breakpoint", () => {
  const canonical = [{ type: "text", text: "inspect" }];
  const prefix = [{ type: "text", text: "account reminder" }];
  const initialSystemMessages = [
    {
      role: "system",
      content: [
        {
          type: "text",
          text: "environment reminder\ndate reminder",
          cache_control: { type: "ephemeral", ttl: "1h" },
        },
      ],
    },
  ];
  const current = {
    messages: [
      { role: "user", content: [{ type: "text", text: "changed reminder" }, ...canonical] },
      { role: "system", content: "environment reminder" },
      { role: "assistant", content: [{ type: "text", text: "first reply" }] },
      { role: "user", content: [...prefix, { type: "text", text: "follow up" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "old result" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "read", input: {} }] },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "t2", content: "current result\naccount reminder" }],
      },
      {
        role: "system",
        content: [
          {
            type: "text",
            text: "date reminder",
            cache_control: { type: "ephemeral", ttl: "1h" },
          },
        ],
      },
    ],
  };

  const rewritten = JSON.parse(
    rewriteInitialRequest(Buffer.from(JSON.stringify(current)), prefix, canonical, initialSystemMessages).toString("utf8"),
  ) as typeof current;
  assert.deepEqual(rewritten.messages[0]?.content, [...prefix, ...canonical]);
  assert.equal(
    ((rewritten.messages[1]?.content as Array<Record<string, unknown>>)[0]?.text as string),
    "environment reminder\ndate reminder",
  );
  assert.equal(
    ((rewritten.messages[1]?.content as Array<Record<string, unknown>>)[0]?.cache_control),
    undefined,
  );
  assert.equal(rewritten.messages.length, 8);
  assert.deepEqual(rewritten.messages[3]?.content, [{ type: "text", text: "follow up" }]);
  assert.equal(
    ((rewritten.messages[5]?.content as Array<Record<string, unknown>>)[0]?.content as string),
    "old result",
  );
  assert.deepEqual(
    (rewritten.messages[5]?.content as Array<Record<string, unknown>>).at(-1)?.cache_control,
    { type: "ephemeral", ttl: "1h" },
  );
  assert.equal(
    ((rewritten.messages[7]?.content as Array<Record<string, unknown>>)[0]?.content as string),
    "current result",
  );
  assert.deepEqual(
    (rewritten.messages.at(-1)?.content as Array<Record<string, unknown>>).at(-1)?.cache_control,
    { type: "ephemeral", ttl: "1h" },
  );
});

// Shape of Claude Code 2.1.308's context reminder, with a fake address.
const EMAIL_SECTION =
  "# userEmail\nThe user's email address is user@example.com. Use it only to identify the user.\n";
const contextReminder = (sections: string) =>
  "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n" +
  `${sections}\nClaude Code attached this context automatically; it isn't part of the user's message.\n</system-reminder>\n`;

test("strips the account email from a first request and keeps other context", () => {
  const body = {
    model: "claude-sonnet-5",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: contextReminder(`${EMAIL_SECTION}# currentDate\nToday's date is 2026-11-02.\n`) },
          { type: "text", text: "inspect" },
        ],
      },
    ],
  };
  const out = JSON.parse(stripUserEmail(Buffer.from(JSON.stringify(body))).toString("utf8")) as typeof body;
  assert.deepEqual(out.messages[0]?.content, [
    { type: "text", text: contextReminder("# currentDate\nToday's date is 2026-11-02.\n") },
    { type: "text", text: "inspect" },
  ]);
  assert.equal(out.model, "claude-sonnet-5");
});

test("drops an email-only reminder and leaves requests without one untouched", () => {
  const body = {
    messages: [
      { role: "user", content: [{ type: "text", text: contextReminder(EMAIL_SECTION) }, { type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: `done\n${contextReminder(EMAIL_SECTION)}` },
        ],
      },
    ],
  };
  const out = JSON.parse(stripUserEmail(Buffer.from(JSON.stringify(body))).toString("utf8")) as typeof body;
  assert.deepEqual(out.messages[0]?.content, [{ type: "text", text: "hi" }]);
  assert.equal((out.messages[2]?.content as Array<Record<string, unknown>>)[0]?.content, "done");
  assert.doesNotMatch(JSON.stringify(out), /userEmail|user@example\.com/);

  const plain = Buffer.from(JSON.stringify({ messages: [{ role: "user", content: "hi" }] }));
  assert.equal(stripUserEmail(plain), plain);
});

test("removes the email sentence even under a renamed heading", () => {
  const text = contextReminder("# accountEmail\nThe user's email address is user@example.com.\n# currentDate\nToday.\n");
  const body = Buffer.from(JSON.stringify({ messages: [{ role: "user", content: [{ type: "text", text }] }] }));
  const out = stripUserEmail(body).toString("utf8");
  assert.doesNotMatch(out, /user@example\.com/);
  assert.match(out, /# currentDate/);
});

test("a replay rebuilds the first message exactly as the first request sent it", () => {
  const reminder = contextReminder(`${EMAIL_SECTION}# currentDate\nToday's date is 2026-11-02.\n`);
  const prefix = [{ type: "text", text: reminder }];
  const canonical = [{ type: "text", text: "inspect" }];
  const initialSystemMessages = [{ role: "system", content: "environment" }];
  const opening = { role: "user", content: [...prefix, ...canonical] };
  const parse = (body: Buffer) =>
    JSON.parse(body.toString("utf8"), (key, value) => (key === "cache_control" ? undefined : value)) as {
      messages: Array<{ role: string; content: unknown }>;
    };

  // What the relay sends on the conversation's first request (no saved prefix yet).
  const firstRequest = parse(stripUserEmail(Buffer.from(JSON.stringify({ messages: [opening] }))));
  // What it sends when a fresh process replays the conversation.
  const replay = parse(
    rewriteInitialRequest(
      Buffer.from(
        JSON.stringify({
          messages: [
            opening,
            { role: "system", content: "environment" },
            { role: "assistant", content: [{ type: "text", text: "first reply" }] },
            { role: "user", content: [...prefix, { type: "text", text: "follow up" }] },
          ],
        }),
      ),
      prefix,
      canonical,
      initialSystemMessages,
    ),
  );

  assert.doesNotMatch(JSON.stringify([firstRequest, replay]), /userEmail|user@example\.com/);
  assert.deepEqual(replay.messages[0], firstRequest.messages[0]);
  assert.match(JSON.stringify(replay.messages[0]), /Today's date is 2026-11-02/);
  // The repeated reminder on the follow-up is removed as a duplicate.
  assert.deepEqual(
    (replay.messages[3]?.content as Array<Record<string, unknown>>).map((block) => block.text),
    ["follow up"],
  );
  assert.match(prefix[0]!.text, /userEmail/, "caller's saved prefix is not mutated");
});

test("the relay strips the email from a new conversation's first request", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-claude-code-email-"));
  const workingRoot = await realpath(root);
  const executable = join(root, "fake-claude.mjs");
  // A stand-in for Claude Code: it sends its own request, reminder included, through the relay.
  await writeFile(
    executable,
    `#!/usr/bin/env node
import readline from "node:readline";
if (process.argv.includes("auth") && process.argv.includes("status")) {
  console.log(JSON.stringify({loggedIn: true, authMethod: "claude.ai", subscriptionType: "test"}));
  process.exit(0);
}
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  const frame = JSON.parse(line);
  if (frame.type !== "user") continue;
  const reminder = ${JSON.stringify(contextReminder(`${EMAIL_SECTION}# currentDate\nToday's date is 2026-11-02.\n`))};
  const body = {messages: [{role: "user", content: [{type: "text", text: reminder}, ...frame.message.content]}]};
  const response = await fetch(process.env.ANTHROPIC_BASE_URL + "/v1/messages", {method: "POST", body: JSON.stringify(body)});
  await response.text();
  console.log(JSON.stringify({type: "result", subtype: "success", is_error: false}));
}
`,
    "utf8",
  );
  await chmod(executable, 0o755);

  const received: string[] = [];
  const upstream = http.createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    received.push(body);
    const usage = { input_tokens: 2, output_tokens: 1, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };
    const events = [
      { type: "message_start", message: { id: "m1", role: "assistant", content: [], usage } },
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
      { type: "content_block_stop", index: 0 },
      { type: "message_delta", delta: { stop_reason: "end_turn" }, usage },
      { type: "message_stop" },
    ];
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""));
  });
  await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
  const port = (upstream.address() as { port: number }).port;

  try {
    const result = await runClaudeRequest({
      model: "sonnet",
      cwd: workingRoot,
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "hello" }] } }],
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      toolNames: new Set(),
      env: { PI_CLAUDE_CODE_COMMAND: executable },
      upstreamUrl: `http://127.0.0.1:${port}`,
      timeoutMs: 5_000,
    });
    assert.equal(result.relay.upstreamRequests, 1);
    assert.equal(received.length, 1);
    assert.doesNotMatch(received[0]!, /userEmail|user@example\.com/);
    assert.match(received[0]!, /Today's date is 2026-11-02/);
    assert.match(received[0]!, /hello/);
  } finally {
    await closeClaudeProcesses();
    await new Promise<void>((resolve) => upstream.close(() => resolve()));
    await rm(root, { recursive: true, force: true });
  }
});

test("injects native system history at assistant boundaries", () => {
  const body = {
    system: [
      { type: "text", text: "identity", cache_control: { type: "ephemeral", ttl: "1h" } },
      { type: "text", text: "prompt", cache_control: { type: "ephemeral", ttl: "1h" } },
    ],
    messages: [
      { role: "user", content: [{ type: "text", text: "first" }] },
      { role: "system", content: [{ type: "text", text: "environment" }] },
      { role: "assistant", content: [{ type: "text", text: "answer" }] },
      { role: "user", content: [{ type: "text", text: "second" }] },
      { role: "system", content: [{ type: "text", text: "token reminder" }] },
    ],
  };
  const rewritten = JSON.parse(
    rewriteInitialRequest(Buffer.from(JSON.stringify(body)), undefined, body.messages[0]!.content, undefined, [
      {
        beforeAssistant: 0,
        content: [{ type: "tool_addition", tool: { type: "tool_reference", name: "mcp__pi__beta" } }],
      },
      {
        beforeAssistant: null,
        content: [{ type: "tool_removal", tool: { type: "tool_reference", name: "mcp__pi__alpha" } }],
      },
    ]).toString("utf8"),
  ) as typeof body;

  assert.deepEqual(rewritten.messages.map((message) => message.role), [
    "user",
    "system",
    "system",
    "assistant",
    "user",
    "system",
    "system",
  ]);
  assert.equal(
    ((rewritten.messages[2]?.content as Array<Record<string, unknown>>)[0] as Record<string, unknown>).type,
    "tool_addition",
  );
  assert.equal(
    ((rewritten.messages.at(-1)?.content as Array<Record<string, unknown>>)[0] as Record<string, unknown>).type,
    "tool_removal",
  );
  assert.equal((rewritten.system[0] as Record<string, unknown>).cache_control, undefined);
  assert.deepEqual((rewritten.system[1] as Record<string, unknown>).cache_control, {
    type: "ephemeral",
    ttl: "1h",
  });
  assert.deepEqual(
    (rewritten.messages.at(-1)?.content as Array<Record<string, unknown>>).at(-1)?.cache_control,
    { type: "ephemeral", ttl: "1h" },
  );
});

test("runs a request through a stream-json Claude executable", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-claude-code-test-"));
  const workingRoot = await realpath(root);
  const executable = join(root, "fake-claude.mjs");
  await writeFile(
    executable,
    `#!/usr/bin/env node
import readline from "node:readline";
import {appendFileSync, readFileSync} from "node:fs";
if (process.argv.includes("auth") && process.argv.includes("status")) {
  console.log(JSON.stringify({loggedIn: true, authMethod: "claude.ai", subscriptionType: "test"}));
  process.exit(0);
}
if (process.cwd() !== process.env.EXPECTED_CWD) throw new Error(\`wrong cwd: \${process.cwd()}\`);
if (process.env.PROCESS_LOG) appendFileSync(process.env.PROCESS_LOG, \`\${process.pid}\\n\`);
const systemIndex = process.argv.indexOf("--system-prompt-file");
if (systemIndex >= 0 && readFileSync(process.argv[systemIndex + 1], "utf8") !== process.env.EXPECTED_SYSTEM_PROMPT) {
  throw new Error("wrong system prompt");
}
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
let queryCount = 0;
for await (const line of input) {
  const frame = JSON.parse(line);
  if (frame.type === "control_request" && frame.request?.subtype === "initialize") {
    console.log(JSON.stringify({
      type: "control_response",
      response: {
        subtype: "success",
        request_id: frame.request_id,
        response: {
          models: [
            {value: "default", resolvedModel: "claude-opus-5-5[1m]", displayName: "Default", description: "Opus with 1M context"},
            {value: "sonnet", resolvedModel: "claude-sonnet-5", displayName: "Sonnet", description: "Efficient", supportsAdaptiveThinking: true}
          ]
        }
      }
    }));
    continue;
  }
  if (frame.shouldQuery === false) {
    console.log(JSON.stringify({type: "result", num_turns: 0}));
    continue;
  }
  if (frame.type !== "user") continue;
  queryCount += 1;
  const requestedTool = frame.message?.content?.some?.(block => block.type === "text" && block.text === "call-tool");
  const text = queryCount === 1 ? "ok" : \`ok\${queryCount}\`;
  const id = queryCount === 1 ? "m_fake" : \`m_fake_\${queryCount}\`;
  const usage = {input_tokens: 3, output_tokens: 1, cache_read_input_tokens: 0, cache_creation_input_tokens: 0};
  if (requestedTool) {
    const content = [{type: "tool_use", id: "tool_fake", name: "mcp__pi__echo", input: {value: "test"}}];
    console.log(JSON.stringify({type: "stream_event", event: {type: "message_stop"}}));
    console.log(JSON.stringify({type: "assistant", message: {id, role: "assistant", content, stop_reason: "tool_use", usage}}));
    console.log(JSON.stringify({type: "result", subtype: "error_max_turns", is_error: true, usage}));
    continue;
  }
  const events = [
    {type: "message_start", message: {id, role: "assistant", content: [], usage}},
    {type: "content_block_start", index: 0, content_block: {type: "text", text: ""}},
    {type: "content_block_delta", index: 0, delta: {type: "text_delta", text}},
    {type: "content_block_stop", index: 0},
    {type: "message_delta", delta: {stop_reason: "end_turn"}, usage},
    {type: "message_stop"}
  ];
  for (const event of events) console.log(JSON.stringify({type: "stream_event", event}));
  console.log(JSON.stringify({type: "assistant", message: {id, role: "assistant", content: [{type: "text", text}], stop_reason: "end_turn", usage}}));
  console.log(JSON.stringify({type: "result", subtype: "success", is_error: false, usage}));
}
`,
    "utf8",
  );
  await chmod(executable, 0o755);

  const events: string[] = [];
  try {
    const provider = createClaudeCodeProvider();
    const authCheck = await provider.auth.apiKey?.check?.({
      ctx: {
        env: async (name) => (name === "PI_CLAUDE_CODE_COMMAND" ? executable : undefined),
        fileExists: async () => false,
      },
      signal: new AbortController().signal,
    });
    assert.equal(authCheck?.type, "oauth");
    assert.equal(provider.auth.oauth?.isSubscription, true);
    await assert.rejects(
      provider.auth.oauth!.login({
        signal: new AbortController().signal,
        prompt: async () => "",
        notify: () => {},
      }),
      /claude auth login/,
    );

    const discovered = await discoverClaudeModels(
      { PI_CLAUDE_CODE_COMMAND: executable, EXPECTED_CWD: process.cwd() },
      undefined,
      5_000,
    );
    assert.deepEqual(discovered.map((model) => model.value), ["default", "sonnet"]);

    const result = await runClaudeRequest({
      model: "sonnet",
      cwd: workingRoot,
      frames: [
        { type: "user", message: { role: "user", content: [{ type: "text", text: "old" }] }, shouldQuery: false },
        { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "old reply" }] } },
        { type: "user", message: { role: "user", content: [{ type: "text", text: "new" }] } },
      ],
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      toolNames: new Set(),
      env: {
        PI_CLAUDE_CODE_COMMAND: executable,
        EXPECTED_CWD: workingRoot,
        EXPECTED_SYSTEM_PROMPT: "system",
      },
      timeoutMs: 5_000,
      onEvent: (event) => events.push(event.type),
    });
    assert.equal(result.message.id, "m_fake");
    assert.equal(result.message.stop_reason, "end_turn");
    assert.equal(result.usage.input_tokens, 3);
    assert.equal(result.relay.upstreamRequests, 0);
    assert.equal(events.at(-1), "message_stop");

    const processLog = join(root, "processes.log");
    const common = {
      model: "sonnet",
      cwd: workingRoot,
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      toolNames: new Set<string>(),
      env: {
        PI_CLAUDE_CODE_COMMAND: executable,
        EXPECTED_CWD: workingRoot,
        EXPECTED_SYSTEM_PROMPT: "system",
        PROCESS_LOG: processLog,
      },
      timeoutMs: 5_000,
    };
    const first = await runClaudeRequest({
      ...common,
      sessionId: "stable-session",
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "one" }] } }],
    });
    const second = await runClaudeRequest({
      ...common,
      sessionId: "stable-session",
      frames: [
        {
          type: "user",
          message: { role: "user", content: [{ type: "text", text: "one" }] },
          shouldQuery: false,
        },
        { type: "assistant", message: { role: "assistant", content: first.message.content } },
        { type: "user", message: { role: "user", content: [{ type: "text", text: "two" }] } },
      ],
    });
    assert.equal(second.message.id, "m_fake_2");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 1);
    assert.equal(cachedClaudeProcessCount(), 1);

    // A branch mismatch replaces the warm process instead of risking stale
    // hidden Claude state.
    const branched = await runClaudeRequest({
      ...common,
      sessionId: "stable-session",
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "branch" }] } }],
    });
    assert.equal(branched.message.id, "m_fake");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 2);

    // Retained workers are deliberately bounded. Creating two more sessions
    // evicts the least-recently-used idle worker and leaves only two alive.
    for (const sessionId of ["second-session", "third-session"]) {
      await runClaudeRequest({
        ...common,
        sessionId,
        frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: sessionId }] } }],
      });
    }
    assert.equal(cachedClaudeProcessCount(), 2);
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 4);

    // A tool call makes Claude's hidden history diverge because its inert MCP
    // server has already returned a denial. Never retain that process; replay
    // the real Pi tool result in a fresh process on the next request.
    const toolCommon = {
      ...common,
      manifest: [
        {
          name: "echo",
          description: "Echo a value",
          inputSchema: { type: "object", properties: { value: { type: "string" } } },
        },
      ],
      extraBody: {
        tools: [
          {
            name: "mcp__pi__echo",
            description: "Echo a value",
            input_schema: { type: "object", properties: { value: { type: "string" } } },
          },
        ],
      },
      toolNames: new Set(["echo"]),
    };
    const toolCall = await runClaudeRequest({
      ...toolCommon,
      sessionId: "tool-session",
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "call-tool" }] } }],
    });
    assert.equal(toolCall.message.stop_reason, "tool_use");
    assert.equal(cachedClaudeProcessCount(), 1);
    const afterTool = await runClaudeRequest({
      ...toolCommon,
      sessionId: "tool-session",
      frames: [
        {
          type: "user",
          message: { role: "user", content: [{ type: "text", text: "call-tool" }] },
          shouldQuery: false,
        },
        { type: "assistant", message: { role: "assistant", content: toolCall.message.content } },
        {
          type: "user",
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "tool_fake", content: "echoed" }],
          },
        },
      ],
    });
    assert.equal(afterTool.message.id, "m_fake");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 6);
  } finally {
    await closeClaudeProcesses();
    await rm(root, { recursive: true, force: true });
  }
});
