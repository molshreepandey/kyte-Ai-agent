/* AI Voice Agent - browser client.
   Mic -> VAD -> 16 kHz WAV -> /api/stt -> /api/chat -> speechSynthesis (TTS). */
(() => {
  const $ = (id) => document.getElementById(id);
  const SESSION = "s-" + Math.random().toString(36).slice(2, 10);

  const EXAMPLES = [
    { tag: "RAG", text: "What are the transfer limits on the Plus plan?" },
    { tag: "MCP", text: "What's the weather in Mumbai right now?" },
    { tag: "RAG+MCP", text: "If I send 2,500 dollars abroad on the Premium plan, how much is the fee?" },
    { tag: "RAG+MCP", text: "What time is it in New York, and when does your phone support close?" },
  ];

  const els = {
    orb: $("orb"), callBtn: $("callBtn"), muteBtn: $("muteBtn"), clearBtn: $("clearBtn"),
    hint: $("hint"), label: $("stateLabel"), timer: $("timer"), feed: $("feed"),
    empty: $("empty"), chips: $("chips"), form: $("composer"), input: $("textIn"),
    wave: $("wave"), toast: $("toast"), badges: $("badges"), tools: $("toolList"),
  };

  let state = "idle";           // idle | listening | thinking | speaking
  let inCall = false;
  let voiceOn = true;
  let busy = false;
  let callStart = 0, timerId = null;

  const LABELS = {
    idle: "Ready to call", listening: "Listening…", thinking: "Thinking…", speaking: "Speaking…",
  };
  const HINTS = {
    idle: "Press the call button and just talk. I'll answer out loud.",
    listening: "Go ahead, I'm listening. I'll respond when you pause.",
    thinking: "Transcribing and deciding which tools to use…",
    speaking: "Tap the orb to interrupt.",
  };
  function setState(s) {
    state = s;
    document.body.dataset.state = s;
    els.label.textContent = inCall || s !== "idle" ? LABELS[s] : "Ready to call";
    els.hint.textContent = HINTS[s];
  }

  /* ------------------------------------------------------------ helpers */
  let toastTimer;
  function toast(msg) {
    els.toast.textContent = msg;
    els.toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove("show"), 5000);
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function scrollDown() { els.feed.scrollTop = els.feed.scrollHeight; }
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      // Server errors use {"error": {"code", "message", "request_id"}}.
      let detail = r.statusText || "Request failed";
      try {
        const body = await r.json();
        detail = (body.error && body.error.message) || body.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return r.json();
  }

  /* ------------------------------------------------------------- status */
  async function loadStatus() {
    try {
      const s = await api("/api/status");
      els.badges.innerHTML = "";
      const badge = (k, v, warn) => {
        const b = el("span", "badge" + (warn ? " warn" : ""));
        b.append(k + " ", Object.assign(el("b"), { textContent: v }));
        els.badges.append(b);
      };
      sttMode = (s.stt_mode === "browser" && SR) ? "browser" : "openrouter";
      if (s.stt_mode === "browser" && !SR) toast("This browser has no speech recognition. Use Chrome or Edge, or set STT_MODE=openrouter.");
      if (!s.ready) badge("⚠", "OPENROUTER_API_KEY missing", true);
      badge("LLM", s.llm_model);
      badge("STT", s.stt_model);
      badge("Vector store", `${s.chunks} chunks · ${s.embedder}`);
      els.tools.innerHTML = "";
      s.tools.forEach((t) => els.tools.append(el("span", "tool " + t.kind, t.name)));
    } catch (e) { toast("Cannot reach server: " + e.message); }
  }

  /* ------------------------------------------------------------ messages */
  function hideEmpty() { if (els.empty) { els.empty.remove(); els.empty = null; } }
  function addUser(text) {
    hideEmpty();
    const m = el("div", "msg user");
    m.append(el("div", "bubble", text));
    els.feed.append(m); scrollDown();
  }
  function addThinking() {
    const m = el("div", "msg bot");
    const b = el("div", "bubble");
    const t = el("span", "typing"); t.append(el("i"), el("i"), el("i"));
    b.append(t); m.append(b);
    els.feed.append(m); scrollDown();
    return m;
  }
  function renderBot(m, data) {
    m.innerHTML = "";
    if (data.steps && data.steps.length) {
      const trace = el("div", "trace");
      data.steps.forEach((s) => {
        const d = el("details", "step " + s.kind);
        const sum = el("summary");
        sum.append(el("span", "tag " + s.kind, s.kind === "rag" ? "RAG" : "MCP"),
          el("code", null, s.tool),
          el("span", "args", JSON.stringify(s.args)));
        d.append(sum, el("pre", null, s.output));
        trace.append(d);
      });
      m.append(trace);
    }
    m.append(el("div", "bubble", data.answer));
    const meta = el("div", "meta", `${(data.latency_ms / 1000).toFixed(1)}s`);
    const replay = el("button", null, "▶ replay");
    replay.onclick = () => speak(data.answer, false);
    meta.append(replay);
    m.append(meta);
    scrollDown();
  }
  function renderError(m, msg) {
    m.innerHTML = "";
    m.append(el("div", "bubble err", "⚠ " + msg));
    scrollDown();
  }

  /* ----------------------------------------------------------------- TTS */
  let voice = null;
  function pickVoice() {
    const vs = speechSynthesis.getVoices();
    if (!vs.length) return;
    const en = vs.filter((v) => /^en[-_]/i.test(v.lang));
    voice = en.find((v) => /natural|online/i.test(v.name) && /en[-_](IN|US|GB)/i.test(v.lang))
      || en.find((v) => /google/i.test(v.name)) || en.find((v) => /en[-_]IN/i.test(v.lang))
      || en.find((v) => /en[-_]US/i.test(v.lang)) || en[0] || vs[0];
  }
  if ("speechSynthesis" in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }

  function speak(text, thenListen = true) {
    return new Promise((resolve) => {
      if (!voiceOn || !("speechSynthesis" in window)) return resolve();
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      if (voice) u.voice = voice;
      u.rate = 1.03;
      const done = () => { if (state === "speaking") setState("idle"); resolve(); };
      u.onend = done; u.onerror = done;
      if (thenListen) setState("speaking");
      speechSynthesis.speak(u);
    });
  }
  function stopSpeaking() { if ("speechSynthesis" in window) speechSynthesis.cancel(); }

  /* ------------------------------------------------- audio capture + VAD */
  let stream = null, actx = null, analyser = null, recorder = null, chunks = [];
  let listenTok = 0;
  let sttMode = "browser";
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recog = null;

  /** Free STT using the browser's Web Speech API. Resolves with text ("" if nothing heard). */
  function listenBrowser() {
    return new Promise((resolve, reject) => {
      const tok = ++listenTok;
      let text = "", settled = false;
      const finish = (v, err) => {
        if (settled) return; settled = true;
        err ? reject(err) : resolve(tok === listenTok ? v : "");
      };
      recog = new SR();
      recog.lang = "en-US";
      recog.interimResults = false;
      recog.continuous = false;
      recog.onresult = (e) => { text = Array.from(e.results).map((r) => r[0].transcript).join(" "); };
      recog.onerror = (e) => {
        if (e.error === "no-speech" || e.error === "aborted") finish("");
        else finish("", new Error("Speech recognition error: " + e.error));
      };
      recog.onend = () => finish(text.trim());
      try { recog.start(); } catch (e) { finish("", e); }
    });
  }

  async function ensureMic() {
    if (stream && stream.active) return;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === "suspended") await actx.resume();
    analyser = actx.createAnalyser();
    analyser.fftSize = 1024;
    actx.createMediaStreamSource(stream).connect(analyser);
  }

  /** Records one utterance, stopping on silence. Resolves with a Blob or null. */
  function recordUtterance() {
    return new Promise((resolve) => {
      const tok = ++listenTok;
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onstop = () => resolve(spoke && tok === listenTok ? new Blob(chunks, { type: recorder.mimeType }) : null);

      const buf = new Float32Array(analyser.fftSize);
      const t0 = performance.now();
      let spoke = false, lastVoice = t0, floor = 0.01;
      recorder.start(100);

      (function tick() {
        if (tok !== listenTok || recorder.state === "inactive") return;
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        const rms = Math.sqrt(sum / buf.length);
        const now = performance.now();
        if (!spoke) floor = Math.min(0.03, floor * 0.98 + rms * 0.02);   // adapt to room noise
        const thr = Math.max(0.02, floor * 2.5);
        if (rms > thr) { spoke = true; lastVoice = now; }
        const silentFor = now - lastVoice;
        if ((spoke && silentFor > 1300) || now - t0 > 20000 || (!spoke && now - t0 > 9000)) {
          recorder.stop();
          return;
        }
        requestAnimationFrame(tick);
      })();
    });
  }

  async function blobToWav16k(blob) {
    const arr = await blob.arrayBuffer();
    const decoded = await actx.decodeAudioData(arr);
    const len = Math.ceil(decoded.duration * 16000);
    const off = new OfflineAudioContext(1, len, 16000);
    const src = off.createBufferSource();
    src.buffer = decoded; src.connect(off.destination); src.start();
    const pcm = (await off.startRendering()).getChannelData(0);
    const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const w = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    w(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVEfmt ");
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
    out.setUint32(24, 16000, true); out.setUint32(28, 32000, true);
    out.setUint16(32, 2, true); out.setUint16(34, 16, true); w(36, "data");
    out.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      out.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([out], { type: "audio/wav" });
  }

  /* ------------------------------------------------------- call lifecycle */
  async function listenLoop() {
    while (inCall) {
      setState("listening");
      let text = "";
      if (sttMode === "browser") {
        try { text = await listenBrowser(); } catch (e) { toast(e.message); endCall(); return; }
        if (!inCall) break;
        if (!text) continue;                     // silence: keep listening
        setState("thinking");
      } else {
        let blob;
        try { blob = await recordUtterance(); } catch (e) { toast("Recording error: " + e.message); break; }
        if (!inCall) break;
        if (!blob) continue;
        setState("thinking");
        try {
          const wav = await blobToWav16k(blob);
          const fd = new FormData(); fd.append("audio", wav, "speech.wav");
          text = (await api("/api/stt", { method: "POST", body: fd })).text.trim();
        } catch (e) { toast(e.message); endCall(); return; }
      }
      if (!inCall) break;
      if (!text) continue;
      await ask(text);
    }
  }

  async function ask(text) {
    if (busy) return;
    busy = true;
    addUser(text);
    setState("thinking");
    const slot = addThinking();
    try {
      const data = await api("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION, text }),
      });
      renderBot(slot, data);
      busy = false;
      await speak(data.answer);
    } catch (e) {
      renderError(slot, e.message);
      toast(e.message);
    } finally {
      busy = false;
      if (!inCall) setState("idle");
    }
  }

  async function startCall() {
    if (inCall) return;
    try { await ensureMic(); }
    catch (e) { return toast("Microphone unavailable: " + (e.message || e.name) + " - you can still type below."); }
    if (sttMode === "browser" && !SR) return toast("Speech recognition needs Chrome or Edge.");
    inCall = true;
    document.body.classList.add("in-call");
    els.callBtn.querySelector("span").textContent = "End call";
    callStart = Date.now();
    timerId = setInterval(() => {
      const s = Math.floor((Date.now() - callStart) / 1000);
      els.timer.textContent = String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
    }, 500);
    listenLoop();
  }

  function endCall() {
    inCall = false; listenTok++;
    if (recorder && recorder.state !== "inactive") recorder.stop();
    if (recog) { try { recog.abort(); } catch (_) {} }
    stopSpeaking();
    clearInterval(timerId);
    document.body.classList.remove("in-call");
    els.callBtn.querySelector("span").textContent = "Start call";
    els.timer.textContent = "00:00";
    setState("idle");
  }

  const toggleCall = () => (inCall ? endCall() : startCall());
  els.callBtn.onclick = toggleCall;
  els.orb.onclick = () => {
    if (!inCall) return startCall();
    if (state === "speaking") { stopSpeaking(); setState("listening"); return; }  // barge-in
    if (state === "listening" && sttMode === "browser" && recog) recog.stop();                 // finish early
    else if (state === "listening" && recorder && recorder.state === "recording") recorder.stop();
  };

  /* ---------------------------------------------------------- controls */
  els.muteBtn.onclick = () => {
    voiceOn = !voiceOn;
    els.muteBtn.textContent = voiceOn ? "🔊 Voice on" : "🔇 Voice off";
    if (!voiceOn) stopSpeaking();
  };
  els.clearBtn.onclick = async () => {
    stopSpeaking();
    els.feed.innerHTML = "";
    els.feed.append(buildEmpty());
    await api("/api/reset", { method: "POST", body: new URLSearchParams({ session_id: SESSION }) }).catch(() => {});
  };
  els.form.onsubmit = (e) => {
    e.preventDefault();
    const t = els.input.value.trim();
    if (!t || busy) return;
    els.input.value = "";
    stopSpeaking();
    ask(t);
  };

  function buildEmpty() {
    const wrap = el("div", "empty");
    wrap.id = "empty";
    wrap.append(el("div", "empty-icon", "🎙️"), el("h3", null, "Ask me anything"),
      el("p", null, "Try a question that needs the knowledge base, a live system tool, or both."));
    const chips = el("div", "chips");
    EXAMPLES.forEach((x) => {
      const c = el("button", "chip");
      const cls = x.tag === "RAG" ? "rag" : "mcp";
      c.append(el("span", "tag " + cls, x.tag), x.text);
      c.onclick = () => { if (!busy) { stopSpeaking(); ask(x.text); } };
      chips.append(c);
    });
    wrap.append(chips);
    els.empty = wrap;
    return wrap;
  }
  els.empty.replaceWith(buildEmpty());

  /* ------------------------------------------------------------ waveform */
  const ctx = els.wave.getContext("2d");
  const bars = 48;
  const heights = new Array(bars).fill(2);
  const data = new Uint8Array(512);
  function draw() {
    const W = els.wave.width, H = els.wave.height;
    ctx.clearRect(0, 0, W, H);
    const color = { listening: "#4ade80", thinking: "#fbbf24", speaking: "#38d9c4" }[state] || "#6b7090";
    if (state === "listening" && analyser) analyser.getByteFrequencyData(data);
    const step = W / bars, now = performance.now() / 1000;
    for (let i = 0; i < bars; i++) {
      let target = 2;
      if (state === "listening" && analyser) target = (data[Math.floor(i * 0.6) + 2] / 255) * H * 0.95;
      else if (state === "speaking") target = (0.25 + 0.75 * Math.abs(Math.sin(now * 5 + i * 0.7) * Math.sin(now * 2.3 + i * 0.3))) * H * 0.8;
      else if (state === "thinking") target = (0.1 + 0.2 * (Math.sin(now * 4 - i * 0.35) + 1)) * H * 0.5;
      heights[i] += (Math.max(2, target) - heights[i]) * 0.3;
      const h = heights[i], x = i * step + step * 0.2;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.35 + 0.65 * (h / H);
      ctx.beginPath();
      ctx.roundRect(x, (H - h) / 2, step * 0.6, h, 3);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(draw);
  }

  setState("idle");
  loadStatus();
  draw();
})();
