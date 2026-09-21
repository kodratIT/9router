// Tests for CommandCode format token-saver support.
//
// CommandCode nests the conversation in params.messages[] with Anthropic-style
// blocks (see translator/request/openai-to-commandcode.js): tool output lives at
// `tool-result.output.value` and prose at `text.text`. None of the savers could
// see that text before — RTK skipped the shape, Headroom reported
// "unsupported commandcode request shape", and the system-prompt injectors fell
// through their messages[]/input[] sniff and silently no-oped.
import { describe, it, expect, vi, afterEach } from "vitest";
import { compressMessages, formatRtkLog } from "../../open-sse/rtk/index.js";
import { compressWithHeadroom } from "../../open-sse/rtk/headroom.js";
import { injectSystemPrompt } from "../../open-sse/rtk/systemInject.js";

afterEach(() => {
  vi.restoreAllMocks();
});

// >500 bytes (MIN_COMPRESS_SIZE) and shaped like build output so autodetect matches.
const buildOutput = [
  ...Array.from({ length: 24 }, (_, i) => `   Compiling package-${i} v1.0.${i}`),
  "    Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.34s",
].join("\n");

function commandCodeBody() {
  return {
    threadId: "thread-1",
    memory: "",
    config: { workingDir: "/repo", date: "2026-09-21", environment: "darwin" },
    params: {
      model: "deepseek/deepseek-v4-flash",
      stream: true,
      max_tokens: 8192,
      temperature: 0.3,
      system: "You are a coding agent.",
      messages: [
        { role: "user", content: [{ type: "text", text: "jalankan build" }] },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Menjalankan build." },
            { type: "tool-call", toolCallId: "call_1", toolName: "bash", input: { command: "cargo build" } },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_1",
              toolName: "bash",
              output: { type: "text", value: buildOutput },
            },
          ],
        },
      ],
    },
  };
}

describe("CommandCode format RTK support", () => {
  it("compresses params.messages[].content[].tool-result.output.value", () => {
    const body = commandCodeBody();

    const stats = compressMessages(body, true);

    expect(stats).not.toBeNull();
    expect(stats.hits.length).toBe(1);
    expect(stats.hits[0].filter).toBe("build-output");
    expect(stats.hits[0].shape).toBe("commandcode-tool-result");
    expect(stats.bytesAfter).toBeLessThan(stats.bytesBefore);

    const savedPercent = ((stats.bytesBefore - stats.bytesAfter) / stats.bytesBefore) * 100;
    expect(savedPercent).toBeGreaterThan(10);
    expect(body.params.messages[2].content[0].output.value).not.toBe(buildOutput);
    expect(formatRtkLog(stats)).toContain("build-output");
  });

  it("leaves CommandCode text and tool-call blocks untouched", () => {
    const body = commandCodeBody();

    compressMessages(body, true);

    expect(body.params.messages[0].content[0].text).toBe("jalankan build");
    expect(body.params.messages[1].content[0].text).toBe("Menjalankan build.");
    expect(body.params.messages[1].content[1]).toEqual({
      type: "tool-call",
      toolCallId: "call_1",
      toolName: "bash",
      input: { command: "cargo build" },
    });
  });

  it("still honours the RTK off switch", () => {
    const body = commandCodeBody();

    expect(compressMessages(body, false)).toBeNull();
    expect(body.params.messages[2].content[0].output.value).toBe(buildOutput);
  });

  it("reports no hits when a CommandCode body carries no tool results", () => {
    const body = commandCodeBody();
    body.params.messages[2].content = [{ type: "text", text: "done" }];

    const stats = compressMessages(body, true);

    expect(stats.hits).toEqual([]);
    expect(formatRtkLog(stats)).toBeNull();
  });

  it("handles malformed CommandCode bodies without crashing", () => {
    const stats = compressMessages(
      { params: { messages: [null, {}, { content: null }, { content: [{ type: "tool-result" }] }] } },
      true
    );

    expect(stats.hits).toEqual([]);
  });
});

describe("CommandCode format Headroom support", () => {
  it("projects blocks to OpenAI shape for /v1/compress and writes the result back", async () => {
    let payload;
    global.fetch = vi.fn(async (_url, init) => {
      payload = JSON.parse(init.body);
      return new Response(
        JSON.stringify({
          messages: [
            { role: "user", content: "jalankan build" },
            {
              role: "assistant",
              content: "Menjalankan build.",
              tool_calls: [{ id: "call_1", type: "function", function: { name: "bash", arguments: '{"command":"cargo build"}' } }],
            },
            { role: "tool", content: "compressed build output", tool_call_id: "call_1" },
          ],
          tokens_before: 900,
          tokens_after: 100,
          tokens_saved: 800,
        }),
        { status: 200 }
      );
    });

    const body = commandCodeBody();

    const stats = await compressWithHeadroom(body, {
      enabled: true,
      url: "http://localhost:8787",
      format: "commandcode",
      model: "deepseek/deepseek-v4-flash",
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8787/v1/compress",
      expect.objectContaining({ method: "POST" })
    );
    expect(payload.messages.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
    expect(payload.messages[1].tool_calls[0].function.name).toBe("bash");
    expect(payload.messages[2].content).toBe(buildOutput);
    expect(payload.messages[2].tool_call_id).toBe("call_1");
    expect(stats.tokens_saved).toBe(800);

    expect(body.params.messages[0].content[0].text).toBe("jalankan build");
    expect(body.params.messages[1].content[0].text).toBe("Menjalankan build.");
    expect(body.params.messages[2].content[0].output.value).toBe("compressed build output");
    expect(body.params.messages[2].content[0].toolCallId).toBe("call_1");
  });

  it("leaves the body untouched when the proxy changes the message count", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ messages: [{ role: "user", content: "only one" }] }), { status: 200 })
    );
    const body = commandCodeBody();
    const before = JSON.stringify(body);

    const stats = await compressWithHeadroom(body, {
      enabled: true,
      url: "http://localhost:8787",
      format: "commandcode",
    });

    expect(stats).toBeNull();
    expect(JSON.stringify(body)).toBe(before);
  });

  it("leaves the body untouched when the proxy reorders roles", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          messages: [
            { role: "user", content: "jalankan build" },
            { role: "system", content: "Menjalankan build." },
            { role: "tool", content: "compressed build output" },
          ],
        }),
        { status: 200 }
      )
    );
    const body = commandCodeBody();

    const stats = await compressWithHeadroom(body, {
      enabled: true,
      url: "http://localhost:8787",
      format: "commandcode",
    });

    expect(stats).toBeNull();
    expect(body.params.messages[2].content[0].output.value).toBe(buildOutput);
  });
});

describe("CommandCode format system-prompt injection", () => {
  it("appends the prompt to params.system, once", () => {
    const body = commandCodeBody();

    injectSystemPrompt(body, "commandcode", "TERSE_MODE_MARKER");
    expect(body.params.system).toContain("TERSE_MODE_MARKER");
    expect(body.params.system.startsWith("You are a coding agent.")).toBe(true);

    injectSystemPrompt(body, "commandcode", "TERSE_MODE_MARKER");
    expect(body.params.system.split("TERSE_MODE_MARKER").length - 1).toBe(1);
  });

  it("creates params.system when the translated body has none", () => {
    const body = commandCodeBody();
    delete body.params.system;

    injectSystemPrompt(body, "commandcode", "TERSE_MODE_MARKER");

    expect(body.params.system).toBe("TERSE_MODE_MARKER");
  });
});
