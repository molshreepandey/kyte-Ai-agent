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

test.beforeEach(async ({ page }) => {
  // Keep tests silent and deterministic: no real speech synthesis.
  await page.addInitScript(() => {
    // @ts-ignore
    window.speechSynthesis && (window.speechSynthesis.speak = (u) => u.onend && u.onend());
  });
});

test("page loads with status badges, tools and example chips", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("AI Voice Agent");
  await expect(page.locator("h1")).toHaveText("AI Voice Agent");
  await expect(page.locator("#badges")).toContainText("Vector store");
  await expect(page.locator("#badges")).toContainText("OPENROUTER_API_KEY missing");
  await expect(page.locator("#toolList .tool.rag")).toHaveText(["search_knowledge_base"]);
  await expect(page.locator("#toolList .tool.mcp", { hasText: "calculate" })).toBeVisible();
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

test("server error is shown to the user", async ({ page }) => {
  await page.route("**/api/chat", (route) =>
    route.fulfill({ status: 502, json: { detail: "LLM request failed" } }),
  );
  await page.goto("/");
  await page.fill("#textIn", "hello");
  await page.press("#textIn", "Enter");
  await expect(page.locator(".msg.bot .bubble.err")).toContainText("LLM request failed");
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
