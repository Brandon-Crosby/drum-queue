/* Playback adjustments only; downloaded stems are never processed here. */
(() => {
  const ranges = Object.freeze({ levelDb: [-24, 6], lowDb: [-12, 12], midsDb: [-6, 6], highDb: [-12, 12] });
  const key = 'ds_preview_sound_v2';
  const bounded = (value, fallback, limits) => typeof value === 'number' && Number.isFinite(value) ? Math.max(limits[0], Math.min(limits[1], value)) : fallback;
  function settings(value = {}) {
    if (!value || typeof value !== 'object') value = {};
    const tracks = {};
    for (const [stem, saved] of Object.entries(value.tracks || {}).slice(0, 32)) {
      if (!/^[a-z][a-z0-9_-]{0,39}$/.test(stem) || !saved || typeof saved !== 'object') continue;
      const clean = { enabled: typeof saved.enabled === 'boolean' ? saved.enabled : true };
      for (const [name, limits] of Object.entries(ranges)) if (typeof saved[name] === 'number' && Number.isFinite(saved[name])) clean[name] = bounded(saved[name], 0, limits);
      tracks[stem] = clean;
    }
    return { includeBass: typeof value.includeBass === 'boolean' ? value.includeBass : false, tracks };
  }
  function trackSettings(track, value = settings()) {
    const saved = value.tracks?.[track.stem] || {};
    return {
      enabled: typeof saved.enabled === 'boolean' ? saved.enabled : true,
      levelDb: bounded(saved.levelDb, track.stem === 'bass' ? -4 : 0, ranges.levelDb),
      lowDb: bounded(saved.lowDb, 0, ranges.lowDb),
      midsDb: bounded(saved.midsDb, track.group === 'kit' ? 1.5 : 0, ranges.midsDb),
      highDb: bounded(saved.highDb, 0, ranges.highDb),
    };
  }
  function load() { try { return settings(JSON.parse(localStorage.getItem(key))); } catch (_) { return settings(); } }
  function save(value) { try { localStorage.setItem(key, JSON.stringify(settings(value))); } catch (_) {} }
  function createTrack(context, track, destination, value) {
    const input = context.createGain(), low = context.createBiquadFilter(), mids = context.createBiquadFilter(), high = context.createBiquadFilter(), level = context.createGain();
    low.type = 'lowshelf'; low.frequency.value = 180;
    mids.type = 'peaking'; mids.frequency.value = 1800; mids.Q.value = .8;
    high.type = 'highshelf'; high.frequency.value = 6000;
    input.connect(low); low.connect(mids); mids.connect(high); high.connect(level); level.connect(destination);
    function update(value, immediate = false) {
      const s = trackSettings(track, value);
      for (const [parameter, target] of [[low.gain, s.enabled ? s.lowDb : 0], [mids.gain, s.enabled ? s.midsDb : 0], [high.gain, s.enabled ? s.highDb : 0], [level.gain, 10 ** (s.levelDb / 20)]]) {
        if (immediate) parameter.setValueAtTime(target, context.currentTime);
        else parameter.setTargetAtTime(target, context.currentTime, .015);
      }
    }
    update(value, true);
    return { input, update, disconnect() { for (const node of [input, low, mids, high, level]) node.disconnect(); } };
  }
  window.DrumPreviewSound = Object.freeze({ ranges, settings, trackSettings, load, save, createTrack });
})();
