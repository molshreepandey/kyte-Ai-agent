// @ts-check
const { test, expect } = require("@playwright/test");

const BOT_REPLY = {
  answer: "The Plus plan allows five thousand dollars per day.",
  steps: [
    { kind: "rag", tool: "search_knowledge_base", args: { query: "plus plan limits" }, output: "[faq] limits..." },
    { kind: "mcp", tool: "calculate", args: { expression: "5000 * 30" }, output: "5000 * 30 = 150000" },
  ],
  latency_ms: 1234,
};

let consoleErrors;

test.beforeEach(async ({ page }) => {
  // Any console error (including Content-Security-Policy violations) fails the test.
  consoleErrors = [];
  page.on("console", (msg) => msg.type() === "error" && consoleErrors.push(msg.text()));
  page.on("pageerror", (err) => consoleErrors.push(String(err)));
  // Keep tests silent and deterministic: no real speech synthesis.
  await page.addInitScript(() => {
    // @ts-ignore
    window.speechSynthesis && (window.speechSynthesis.speak = (u) => u.onend && u.onend());
  });
});

test.afterEach(() => {
  // Error responses that a test stubs on purpose are logged by Chrome as failed loads.
  const unexpected = consoleErrors.filter((e) => !/Failed to load resource/.test(e));
  expect(unexpected, "console errors / CSP violations").toEqual([]);
});

test("page is served with a strict Content-Security-Policy", async ({ page }) => {
  const response = await page.goto("/");
  expect(response.headers()["content-security-policy"]).toContain("script-src 'self'");
  await expect(page.locator(".chips .chip")).toHaveCount(4); // scripts ran under the CSP
});

test("page loads with status badges, tools and example chips", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("AI Voice Agent");
  await expect(page.locator("h1")).toHaveText("AI Voice Agent");
  await expect(page.locator("#badges")).toContainText("Vector store");
  await expect(page.locator("#badges")).toContainText("OPENROUTER_API_KEY missing");
  await expect(page.locator("#toolList .tool.rag")).toHaveText(["search_knowledge_base"]);
  await expect(page.locator("#toolList .tool.mcp", { hasText: "calculate" })).toBeVisible();
  await expect(page.locator("#toolList .tool")).toHaveText([
    "search_knowledge_base", "get_current_time", "calculate", "get_weather",
  ]);
  await expect(page.locator(".chips .chip")).toHaveCount(4);
  await expect(page.locator("#wave")).toBeVisible();
});

test("typed message renders user bubble, answer and tool trace", async ({ page }) => {
  let sent;
  await page.route("**/api/chat", async (route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({ json: BOT_REPLY });
  });
  await page.goto("/");
  await page.fill("#textIn", "What are the Plus limits?");
  await page.press("#textIn", "Enter");

  await expect(page.locator(".msg.user .bubble")).toHaveText("What are the Plus limits?");
  await expect(page.locator(".msg.bot .bubble")).toHaveText(BOT_REPLY.answer);
  await expect(page.locator(".msg.bot .trace .step")).toHaveCount(2);
  await expect(page.locator(".msg.bot .trace .tag.rag")).toHaveText("RAG");
  await expect(page.locator(".msg.bot .meta")).toContainText("1.2s");
  expect(sent.text).toBe("What are the Plus limits?");
  expect(typeof sent.session_id).toBe("string");
});

test("example chip sends its question", async ({ page }) => {
  await page.route("**/api/chat", (route) => route.fulfill({ json: { answer: "ok", steps: [], latency_ms: 10 } }));
  await page.goto("/");
  await page.locator(".chips .chip").first().click();
  await expect(page.locator(".msg.user .bubble")).toHaveCount(1);
  await expect(page.locator(".msg.bot .bubble")).toHaveText("ok");
});

test("server error message from the error envelope is shown to the user", async ({ page }) => {
  await page.route("**/api/chat", (route) =>
    route.fulfill({
      status: 502,
      json: { error: { code: "llm_unavailable", message: "The assistant is temporarily unavailable.", request_id: "r1" } },
    }),
  );
  await page.goto("/");
  await page.fill("#textIn", "hello");
  await page.press("#textIn", "Enter");
  await expect(page.locator(".msg.bot .bubble.err")).toContainText("The assistant is temporarily unavailable.");
});

test("real server validation error is shown (no stub)", async ({ page }) => {
  // The test server has no API key, so the real backend answers with its 503 envelope.
  await page.goto("/");
  await page.fill("#textIn", "hello");
  await page.press("#textIn", "Enter");
  await expect(page.locator(".msg.bot .bubble.err")).toContainText("not configured");
});

test("clear resets the conversation and calls /api/reset", async ({ page }) => {
  await page.route("**/api/chat", (route) => route.fulfill({ json: { answer: "hi", steps: [], latency_ms: 5 } }));
  const reset = page.waitForRequest((r) => r.url().endsWith("/api/reset") && r.method() === "POST");
  await page.goto("/");
  await page.fill("#textIn", "hello");
  await page.press("#textIn", "Enter");
  await expect(page.locator(".msg.bot .bubble")).toHaveText("hi");
  await page.click("#clearBtn");
  await reset;
  await expect(page.locator(".msg")).toHaveCount(0);
  await expect(page.locator(".chips .chip")).toHaveCount(4);
});
