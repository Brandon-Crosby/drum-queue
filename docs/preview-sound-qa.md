# Preview sound controls

The mixer starts with bass included at -4 dB and a broad +1.5 dB drum-mid lift (1.8 kHz, Q 0.8). These are starting settings based on the approved local audition, not a claim that every song needs the same EQ. Per-stem EQ panels are collapsed by default. Each contains Level, Low, Mid and High dials, an EQ bypass and per-stem Reset; Include bass, Reset balance and existing mute/solo controls remain available. The Low and High shelves are at 180 Hz and 6 kHz. EQ bypass leaves the stem level active. Settings are saved in this browser origin.

Sound changes apply only in the playback graph. Individual WAVs, ZIPs and one-shots retain their original content. Kit-only mode excludes bass. A split without a bass stem disables the bass controls. Only kit tracks start with the approved mid lift; other tracks start with flat EQ and can be adjusted individually. Settings are independent per stem.

Mac playback reads the original local WAV stems so even older jobs avoid their old mono previews. Hosted previews are encoded at 44.1 kHz, stereo when present, 320 kbps MP3; WAV previews use PCM 24-bit. Mono inputs remain mono. Existing hosted jobs keep their old encoded previews until a new split is generated. Preview files and decoded stereo buffers are larger than the former 22.05 kHz mono versions; long-job memory/performance remains to be measured.

## Verification

- All 13 Python audio/service regression tests pass, including synthetic 16 kHz stereo content surviving preview encoding, mono input, unchanged source bytes and 320 kbps output.
- `tests/preview_sound_browser.html` renders through the product's actual Web Audio graph in OfflineAudioContext. Browser checks verify starting settings, invalid saved values, bounded dials, independent stem settings, measured bass/mid gains, low/high shelves, EQ bypass, unaffected vocal defaults and preserved stereo polarity.
- UI checks passed in Chrome: dial keyboard increments/labels, snare changes leaving kick settings intact, per-stem EQ bypass disabling only the tone dials, Include bass linked to bass mute, Kit only, reset, and settings persistence after reload. Mouse dragging passed for all four snare dials, including the upper Mid limit; kick settings stayed independent. Live playback advanced, pause held its position, resume and Stop worked, and playback returned to the start at the end of the clip. The rendered desktop layout was visually inspected.
- The actual Zone Six source component builder passes: current Mac patch applies, dependency hashes remain unchanged, local HTML builds without hosted ads/bot checks, and JavaScript syntax checks pass. The UI preview uses this packaged Mac frontend.
- Original stem downloads remain routed to their existing file/ZIP endpoints. Playback sound settings are never sent to those endpoints. The local QA fixture reads private test audio from disk and is not committed or uploaded.

Live UI playback timing is verified; audible output at the Mac speakers was not independently measured. The user's approved WAV audition and offline browser rendering establish the tonal reference; they do not replace that live UI acceptance. No installed runtime or hosted deployment was replaced during this preview.
