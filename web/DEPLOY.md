# Putting the website online

```
visitor ──> Cloudflare Pages (the site + ads)  ──/api/*──>  Modal (FastAPI + L4 GPU)
             web/dist, web/functions/api               web/backend/modal_app.py
```

- **Cloudflare Pages** serves the pages for free, and its small proxy function forwards `/api/*`
  to Modal with the visitor's real IP and a shared secret.
- **Modal** runs the API and the GPU. You pay per second of GPU only while a song is splitting;
  idle costs nothing. Modal's Starter plan includes $30/month of free usage.

Expect about an hour, most of it waiting on sign-ups and the first build.

## 1. Modal (the GPU backend)

1. Sign up at [modal.com](https://modal.com).
2. On your Mac, from the repo folder:
   ```bash
   python3 -m pip install modal
   python3 -m modal setup          # opens the browser to log in
   ```
3. Make two random secrets and store them in Modal (keep the PROXY_SECRET handy for step 3):
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # run twice
   python3 -m modal secret create drumsplit-secrets PROXY_SECRET=<first> IP_SALT=<second>
   ```
4. Deploy:
   ```bash
   python3 -m modal deploy web/backend/modal_app.py
   ```
   The first deploy builds the GPU image and downloads both models (several minutes).
   It prints a URL like `https://<you>--drumsplit-api.modal.run`. That's your **MODAL_API_URL**.
   Visiting it directly should say `Forbidden`, which is correct: only your site may call it.

Knobs at the top of `modal_app.py`: `MIX_MODEL` (`htdemucs` is ~4x cheaper than `htdemucs_ft`),
`MAX_GPUS`, `MAX_MINUTES`, and in `PUBLIC_ENV`: `LIMIT_PER_IP` (splits per visitor per day) and
`DAILY_CAP` (splits per day for the whole site, which is your bill's ceiling). Redeploy after changes.

## 2. Domain

Buy one (Cloudflare Registrar sells at cost, about $10/yr for a .com). Put the name, domain and a
contact address in `web/site.json`. For the contact address, Cloudflare **Email Routing** can
forward `hello@yourdomain` to your Gmail for free.

## 3. Cloudflare Pages (the website)

1. Cloudflare dashboard > **Workers & Pages** > Create > **Pages** > Connect to Git >
   pick `drum-queue` (give Cloudflare's GitHub app access to it).
2. Build settings:
   - Root directory: `web`
   - Build command: `python3 build_site.py`
   - Build output directory: `dist`
3. **Settings > Variables and secrets** (Production), add:
   - `MODAL_API_URL` = the URL from step 1.4
   - `PROXY_SECRET` = the same value you gave Modal (mark it encrypted)
4. **Custom domains**: add your domain.
5. Open the site and split a song. Every push to `main` redeploys the site automatically.

## 4. Bot check (recommended)

Cloudflare dashboard > **Turnstile** > Add site (your domain, "Managed").
- Site key: add `TURNSTILE_SITEKEY` as a Pages variable (or put it in `site.json`), redeploy.
- Secret key: `python3 -m modal secret create drumsplit-secrets --force PROXY_SECRET=... IP_SALT=... TURNSTILE_SECRET=<secret>`
  then `python3 -m modal deploy web/backend/modal_app.py`.

## 5. Ads (Google AdSense)

1. Once the site is live on your domain, apply at [adsense.google.com](https://adsense.google.com).
2. Put your publisher ID (`ca-pub-...`) in `ADSENSE_CLIENT` (Pages variable or `site.json`) and
   redeploy. That adds the ad code to every page and creates `ads.txt`, which AdSense checks.
3. In AdSense: turn on **Auto ads** for the site, and under **Privacy & messaging** set up the
   consent message for European visitors (required to show ads there).
4. Optional: make a display ad unit and put its slot ID in `ADSENSE_WAIT_SLOT` to show an ad
   while songs split (when people are looking at the page and waiting).

Approval can take days to a few weeks. The guide pages are there because AdSense often turns
down sites that are only a tool. Adding more genuinely useful guides helps.

## Running it on your Mac first

```bash
.venv/bin/pip install fastapi uvicorn python-multipart mutagen
.venv/bin/python web/backend/local_dev.py      # http://127.0.0.1:8780
```
Same site and API, using your Mac instead of a cloud GPU.

## Costs at a glance

| | Cost |
|---|---|
| Cloudflare Pages + proxy + Turnstile | free |
| Domain | ~$10/yr |
| Modal GPU (L4) | ~$0.00022/s while splitting, about 1-2 cents per song with `htdemucs_ft`; $30/month free credit |
| Modal storage (24 h of stems) | small |

`DAILY_CAP` × cost per song is the most a day can cost you. At the default 300 that's a few dollars.
