const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");
const dist = process.env.DS_UI_DIST;
assert.ok(dist, "Set DS_UI_DIST to the built page");
const tick = () => new Promise((resolve) => setImmediate(resolve));
for (const all of [false, true])
  test(
    "failed previews recover without trapping the mixer " +
      (all ? "all fail" : "partial failure"),
    async () => {
      const dom = new JSDOM(
          fs.readFileSync(path.join(dist, "index.html"), "utf8"),
          {
            url: "http://localhost/#0123456789abcdef",
            runScripts: "outside-only",
          },
        ),
        w = dom.window,
        counts = {};
      let failing = true;
      let releasePreviews;
      const pendingPreviews = new Promise((resolve) => {
        releasePreviews = resolve;
      });
      try {
        w.SITE = {};
        w.HTMLElement.prototype.scrollIntoView = function () {};
        w.HTMLCanvasElement.prototype.getContext = () => ({
          clearRect() {},
          drawImage() {},
          fillRect() {},
        });
        w.requestAnimationFrame = () => 1;
        w.cancelAnimationFrame = () => {};
        w.AudioContext = class {
          constructor() {
            this.currentTime = 0;
            this.destination = {};
          }
          async resume() {}
          createGain() {
            return {
              connect() {},
              gain: { setValueAtTime() {}, setTargetAtTime() {} },
            };
          }
          createBufferSource() {
            return { connect() {}, start() {}, stop() {} };
          }
          async decodeAudioData() {
            return {
              duration: 9,
              getChannelData: () => new Float32Array(1000),
            };
          }
        };
        const stems = ["kick", "snare", "perc", "hh"].map((stem) => ({
          stem,
          label: stem,
          group: "kit",
          name: stem + ".wav",
          preview: stem + ".mp3",
        }));
        w.fetch = async (url) => {
          counts[url] = (counts[url] || 0) + 1;
          if (url.includes("/preview/")) {
            await pendingPreviews;
            return {
              ok: !(failing && (all || url.endsWith("/hh.mp3"))),
              arrayBuffer: async () => new ArrayBuffer(1),
            };
          }
          return {
            ok: true,
            json: async () =>
              url === "/api/limits"
                ? { left: 3, per_day: 3, max_minutes: 8, max_mb: 30 }
                : {
                    status: "done",
                    name: "song.wav",
                    duration: 9,
                    expires: Date.now() / 1000 + 100,
                    stems,
                    shots: [],
                    loops: [],
                  },
          };
        };
        w.eval(fs.readFileSync(path.join(dist, "app.js"), "utf8"));
        await tick();
        await tick();
        const time = w.document.getElementById("time");
        assert.equal(
          time.textContent,
          "Loading…",
          "pending previews must not be reported as unavailable",
        );
        releasePreviews();
        await tick();
        await tick();
        assert.ok(
          !time.textContent.includes("Loading"),
          "settled failures must not leave permanent Loading",
        );
        if (all) assert.match(time.textContent, /unavailable/);
        else assert.equal(time.textContent, "0:00 / 0:09");
        failing = false;
        const hh = [...w.document.querySelectorAll("#tracks .trk")].at(-1),
          mute = hh.querySelector("button");
        mute.click();
        mute.click();
        await tick();
        await tick();
        assert.equal(
          counts["/api/jobs/0123456789abcdef/preview/hh.mp3"],
          2,
          "mute/unmute must retry the missing file",
        );
        assert.equal(time.textContent, "0:00 / 0:09");
        assert.equal(hh.querySelector("canvas").title, "");
        w.document.getElementById("play").click();
        await tick();
        assert.match(w.document.getElementById("play").textContent, /Pause/);
      } finally {
        releasePreviews();
        await tick();
        await tick();
        w.close();
      }
    },
  );
