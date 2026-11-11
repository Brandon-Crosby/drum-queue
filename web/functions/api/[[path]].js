// Cloudflare Pages Function: forwards /api/* to the Modal backend.
// Cloudflare sets CF-Connecting-IP itself (visitors can't fake it), so the daily limit is per real
// visitor, and PROXY_SECRET means the Modal URL refuses anyone who skips this proxy.
// Set MODAL_API_URL and PROXY_SECRET in Pages > Settings > Environment variables.
export async function onRequest({ request, env }) {
  if (!env.MODAL_API_URL || !env.PROXY_SECRET) {
    return Response.json({ error: 'Site is not configured yet.' }, { status: 503 });
  }
  const url = new URL(request.url);
  const target = env.MODAL_API_URL.replace(/\/$/, '') + url.pathname + url.search;

  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.set('x-proxy-secret', env.PROXY_SECRET);
  headers.set('x-real-ip', request.headers.get('cf-connecting-ip') || '');
  // Visitor location from Cloudflare's own lookup (country + state/region code), for daily totals.
  const cf = request.cf || {};
  headers.set('x-country', cf.country || request.headers.get('cf-ipcountry') || '');
  headers.set('x-region', cf.regionCode || '');

  const init = { method: request.method, headers, redirect: 'manual' };
  if (!['GET', 'HEAD'].includes(request.method)) {
    init.body = request.body;            // streamed, so a 30 MB upload never sits in memory
    init.duplex = 'half';
  }
  const resp = await fetch(target, init);
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
}
