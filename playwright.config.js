// Browser smoke tests for the web UI. The real FastAPI server is started; LLM-backed
// endpoints are stubbed per test with page.route(), so no API key or network is needed.
const { defineConfig, devices } = require("@playwright/test");

const PORT = Number(process.env.E2E_PORT || 8799);
const PYTHON = process.env.PYTHON || (process.platform === "win32" ? ".venv\\Scripts\\python" : ".venv/bin/python");

module.exports = defineConfig({
  testDir: "tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `${PYTHON} -m app.main`,
    url: `http://127.0.0.1:${PORT}/api/status`,
    reuseExistingServer: false,
    timeout: 60_000,
    // An explicitly empty key wins over any developer .env, keeping the run offline.
    env: { HOST: "127.0.0.1", PORT: String(PORT), OPENROUTER_API_KEY: "" },
  },
});
