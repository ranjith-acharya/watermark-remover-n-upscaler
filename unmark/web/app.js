const $ = (id) => document.getElementById(id);
const show = (el, on = true) => el.classList.toggle('hidden', !on);

let ENV = null;
let source = null;
let jobId = null;
let poller = null;

// --------------------------------------------------------------------------
// environment
// --------------------------------------------------------------------------

async function loadEnv() {
  ENV = await (await fetch('/api/env')).json();

  const cuda = ENV.torch.available && ENV.torch.cuda;
  const pills = [
    ['ffmpeg', ENV.ffmpeg],
    [ENV.default_encoder, true],
    [cuda ? ENV.torch.device : 'no CUDA', cuda],
  ];
  $('env').innerHTML = pills
    .map(([label, on]) => `<span class="pill ${on ? 'on' : 'off'}">${label}</span>`)
    .join('');

  $('model').innerHTML = Object.entries(ENV.models)
    .map(([k, label]) => `<option value="${k}">${label}</option>`).join('');

  $('encoder').innerHTML = ['<option value="auto">Auto</option>'].concat(
    ENV.encoders.map((e) => `<option value="${e.name}" ${e.usable ? '' : 'disabled'}>` +
      `${e.name}${e.usable ? '' : ' - unavailable'}</option>`)).join('');

  const dead = ENV.encoders.filter((e) => !e.usable).map((e) => e.name);
  $('encoderNote').textContent = dead.length
    ? `${dead.join(', ')} cannot start here (GPU driver too old for this ffmpeg build).`
    : 'Hardware encoding available.';

  if (ENV.torch.available && ENV.torch.cuda) {
    $('engineNote').textContent = `Auto will use LaMa on the ${ENV.torch.device}.`;
  }
  if (!ENV.torch.available) {
    $('engineNote').textContent =
      'Auto falls back to Telea (CPU). Run install_ai.bat for LaMa, which is much '
      + 'better over structure.';
    $('modeNote').textContent = 'Real-ESRGAN needs PyTorch. Run install_ai.bat.';
  } else if (!ENV.torch.cuda) {
    $('engineNote').textContent = 'PyTorch found, but no CUDA - AI tiers run on CPU (slow).';
  }
}

// --------------------------------------------------------------------------
// source loading
// --------------------------------------------------------------------------

async function post(url, body) {
  const res = await fetch(url, { method: 'POST', body });
  const data = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) throw new Error(data.detail || 'request failed');
  return data;
}

function setBusy(el, text) {
  el.dataset.label = el.dataset.label || el.textContent;
  el.disabled = true;
  el.textContent = text;
}
function clearBusy(el) {
  el.disabled = false;
  if (el.dataset.label) el.textContent = el.dataset.label;
}

async function useSource(promise, button) {
  clearError();
  if (button) setBusy(button, 'Scanning...');
  try {
    source = await promise;
    renderSource();
    renderDetection();
    await refreshPreview();
    show($('step-detect'));
    show($('step-options'));
    show($('step-result'), false);
  } catch (err) {
    showError(err.message);
  } finally {
    if (button) clearBusy(button);
  }
}

function renderSource() {
  const i = source.info;
  const mb = (i.width * i.height) / 1e6;
  $('sourceInfo').innerHTML = `
    <dl class="kv">
      <dt>File</dt><dd class="mono">${source.path}</dd>
      <dt>Video</dt><dd>${i.width} x ${i.height} &middot; ${i.fps.toFixed(2)} fps &middot;
          ${i.duration.toFixed(1)}s &middot; ${i.n_frames} frames &middot; ${i.codec}
          ${i.has_audio ? '+ audio' : '(no audio)'}</dd>
    </dl>`;
  show($('sourceInfo'));
  updateTargetNote();
}

function renderOutro() {
  const o = source.outro;
  const box = $('outroInfo');
  show(box, !!o);
  if (!o) return;
  box.className = 'notice';
  box.innerHTML = `<b>End card found</b> &mdash; ${o.seconds}s from ${o.start_time}s
    (${Math.round(o.confidence * 100)}% confidence).
    <span class="muted">${o.reason}</span>`;
}

function renderDetection() {
  renderOutro();
  const r = source.regions[0];
  if (!r) {
    $('detectInfo').innerHTML =
      `<span class="badge warn">No watermark found</span>
       <div class="notice warn">Nothing static and overlaid was detected, so removal
       will be skipped rather than guessing a position and damaging good footage.
       ${source.outro ? 'The end card above can still be trimmed.' : ''}
       You can still upscale.</div>`;
    return;
  }
  const detected = r.source === 'detected';
  const badge = detected
    ? `<span class="badge ok">Detected &middot; ${Math.round(r.confidence * 100)}% confidence</span>`
    : '<span class="badge warn">Nothing detected - using the requested Flow preset</span>';
  const extra = source.regions.length > 1
    ? ` <span class="muted">(+${source.regions.length - 1} more region)</span>` : '';
  $('detectInfo').innerHTML = `${badge}${extra}
    <dl class="kv" style="margin-top:12px">
      <dt>Position</dt><dd class="mono">x ${r.x}, y ${r.y} &middot; ${r.w} x ${r.h} px</dd>
    </dl>`;
}

async function refreshPreview() {
  if (!source) return;
  const zoom = $('zoom').checked ? 1 : 0;
  const engine = $('engine').value;  // 'auto' is resolved server-side
  $('previewNote').textContent = 'rendering...';
  $('previewImg').src =
    `/api/preview/${source.id}?engine=${engine}&zoom=${zoom}&t=${Date.now()}`;
  $('previewImg').onload = () => {
    $('previewNote').textContent = zoom
      ? 'Nearest-neighbour zoom around the watermark.'
      : 'Full frame.';
  };
}

// --------------------------------------------------------------------------
// options
// --------------------------------------------------------------------------

function updateTargetNote() {
  if (!source) return;
  const short = { off: null, '720p': 720, '1080p': 1080, '1440p': 1440, '4k': 2160 }[$('target').value];
  const i = source.info;
  if (!short) {
    $('targetNote').textContent = `Stays ${i.width} x ${i.height}.`;
  } else {
    const f = short / Math.min(i.width, i.height);
    const even = (n) => Math.max(2, Math.round(n) - (Math.round(n) % 2));
    $('targetNote').textContent =
      `${i.width} x ${i.height} to ${even(i.width * f)} x ${even(i.height * f)} (${f.toFixed(2)}x).`;
  }
  const ai = $('upscaleMode').value === 'ai' && $('target').value !== 'off';
  $('modeNote').textContent = ai
    ? `Roughly ${Math.round(i.n_frames * 0.6 / 60)}-${Math.round(i.n_frames * 2 / 60)} min on this GPU.`
    : 'Resampled by ffmpeg; near-instant.';

  const parts = [];
  if ($('target').value !== 'off') parts.push($('target').value.toUpperCase());
  $('planNote').textContent = parts.length ? `Output: ${parts.join(', ')}` : '';
}

// --------------------------------------------------------------------------
// processing
// --------------------------------------------------------------------------

async function startJob() {
  clearError();
  const body = new FormData();
  body.set('source_id', source.id);
  body.set('remove', 'true');
  body.set('engine', $('engine').value);
  body.set('target', $('target').value);
  body.set('upscale_mode', $('upscaleMode').value);
  body.set('model', $('model').value);
  body.set('encoder', $('encoder').value);
  body.set('quality', $('quality').value);
  body.set('trim_outro', $('trimOutro').checked ? 'true' : 'false');

  try {
    const job = await post('/api/process', body);
    jobId = job.id;
    $('run').disabled = true;
    show($('cancel'));
    show($('progressWrap'));
    show($('step-result'), false);
    poller = setInterval(pollJob, 400);
  } catch (err) {
    showError(err.message);
  }
}

async function pollJob() {
  const job = await (await fetch(`/api/job/${jobId}`)).json();
  $('bar').style.width = `${Math.round(job.fraction * 100)}%`;
  $('stage').textContent = job.message || job.stage;
  $('pct').textContent = job.fraction ? `${Math.round(job.fraction * 100)}%` : '';
  if (!job.done) return;

  clearInterval(poller);
  $('run').disabled = false;
  show($('cancel'), false);
  if (job.error) { showError(job.error); return; }
  if (job.cancelled) { $('stage').textContent = 'Cancelled.'; return; }
  renderResult(job.result);
}

function renderResult(res) {
  const p = res.plan;
  const matte = res.matte && res.matte.note ? res.matte.note : 'exact recovery not needed';
  $('resultInfo').innerHTML = `
    <dl class="kv">
      <dt>Saved to</dt><dd class="mono">${res.output}</dd>
      <dt>Resolution</dt><dd>${p.out_w} x ${p.out_h}${p.mode === 'ai' ? ' (Real-ESRGAN)' : ''}</dd>
      <dt>Removal</dt><dd>${res.engine === 'none' ? 'skipped - nothing detected'
          : `${res.engine} &mdash; <span class="muted">${matte}</span>`}</dd>
      ${res.outro ? `<dt>End card</dt><dd>trimmed ${res.outro.seconds}s from
          ${res.outro.start_time}s</dd>` : ''}
      <dt>Encoder</dt><dd>${res.encoder}</dd>
      <dt>Time</dt><dd>${res.seconds}s for ${res.frames} frames (${res.fps} fps)</dd>
    </dl>`;
  $('resultVideo').src = `/api/video/${jobId}?t=${Date.now()}`;
  show($('step-result'));
  $('step-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// --------------------------------------------------------------------------
// errors
// --------------------------------------------------------------------------

function showError(msg) {
  clearError();
  const box = document.createElement('div');
  box.className = 'err-box';
  box.id = 'errBox';
  box.textContent = msg;
  $('step-options').classList.contains('hidden')
    ? $('step-source').appendChild(box)
    : $('step-options').appendChild(box);
}
function clearError() { const e = $('errBox'); if (e) e.remove(); }

// --------------------------------------------------------------------------
// wiring
// --------------------------------------------------------------------------

$('browse').onclick = () => $('file').click();
$('file').onchange = () => {
  const f = $('file').files[0];
  if (!f) return;
  const body = new FormData();
  body.set('file', f);
  useSource(post('/api/upload', body), $('browse'));
};

$('openPath').onclick = () => {
  const p = $('path').value.trim();
  if (!p) return;
  const body = new FormData();
  body.set('path', p);
  useSource(post('/api/open', body), $('openPath'));
};
$('path').onkeydown = (e) => { if (e.key === 'Enter') $('openPath').click(); };

const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add('hover');
}));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove('hover');
}));
drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const body = new FormData();
  body.set('file', f);
  useSource(post('/api/upload', body));
});

$('zoom').onchange = refreshPreview;
$('engine').onchange = refreshPreview;
$('target').onchange = updateTargetNote;
$('upscaleMode').onchange = updateTargetNote;
$('quality').oninput = () => { $('qualityVal').textContent = $('quality').value; };
$('run').onclick = startJob;
$('cancel').onclick = () => fetch(`/api/job/${jobId}/cancel`, { method: 'POST' });

loadEnv();
