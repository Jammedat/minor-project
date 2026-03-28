/**
 * attendance.js — Live video attendance marking
 *
 * Flow:
 *  1. Start camera (getUserMedia)
 *  2. Every N seconds, capture frame → send to /api/attendance/frame
 *  3. Backend returns match result
 *  4. If new match → add to marked list, show overlay, show toast
 *  5. Track already-marked IDs to avoid duplicate toasts
 */

'use strict';

let stream        = null;
let scanInterval  = null;
let cameraRunning = false;
let markedToday   = new Set();
let frameCount    = 0;
let lastStatus    = '';

const video       = document.getElementById('video');
const snapCanvas  = document.getElementById('snap');
const ctx         = snapCanvas ? snapCanvas.getContext('2d') : null;
const cameraBtn   = document.getElementById('cameraBtn');
const scanStatus  = document.getElementById('scanStatus');
const frameInfo   = document.getElementById('frameInfo');
const markedList  = document.getElementById('markedList');
const matchOverlay = document.getElementById('matchOverlay');
const matchBox    = document.getElementById('matchBox');
const matchName   = document.getElementById('matchName');
const matchRoll   = document.getElementById('matchRoll');
const matchConf   = document.getElementById('matchConf');
const intervalSel = document.getElementById('intervalSel');
const countPresent = document.getElementById('countPresent');
const countAbsent  = document.getElementById('countAbsent');
const markedBadge  = document.getElementById('markedBadge');

// Collect already-marked roll numbers from the DOM
document.querySelectorAll('.marked-item').forEach(el => {
  const codeEl = el.querySelector('code');
  if (codeEl) markedToday.add(codeEl.textContent.trim());
});

// ── Camera control ─────────────────────────────────────────────────────────────

async function toggleCamera() {
  if (cameraRunning) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    setStatus('Starting…', 'secondary');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    video.srcObject = stream;
    await video.play();

    cameraRunning = true;
    cameraBtn.innerHTML = '<i class="bi bi-stop-fill me-1"></i>Stop';
    cameraBtn.className = 'btn btn-sm btn-danger';
    setStatus('Scanning…', 'success');

    startScanLoop();

  } catch (err) {
    console.error('Camera error:', err);
    setStatus('Camera error', 'danger');
    showToast('Could not access camera. Please allow camera permission.', 'danger', 6000);
  }
}

function stopCamera() {
  clearInterval(scanInterval);
  scanInterval  = null;
  cameraRunning = false;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  cameraBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Start';
  cameraBtn.className = 'btn btn-sm btn-success';
  setStatus('Stopped', 'secondary');
  hideOverlay();

  if (frameInfo) frameInfo.textContent = 'Camera stopped.';
}

function startScanLoop() {
  clearInterval(scanInterval);
  const interval = parseInt(intervalSel ? intervalSel.value : 2000);
  scanInterval = setInterval(scanFrame, interval);

  if (intervalSel) {
    intervalSel.addEventListener('change', () => {
      if (cameraRunning) {
        clearInterval(scanInterval);
        scanInterval = setInterval(scanFrame, parseInt(intervalSel.value));
      }
    });
  }
}

// ── Frame capture & recognition ────────────────────────────────────────────────

async function scanFrame() {
  if (!cameraRunning || video.readyState < 2) return;

  // Capture frame
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;

  snapCanvas.width  = w;
  snapCanvas.height = h;

  // Mirror the image back to normal orientation before sending to backend
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  const imageB64 = snapCanvas.toDataURL('image/jpeg', 0.75);

  // Determine recognition mode
  const modeEl = document.querySelector('input[name="mode"]:checked');
  const mode   = modeEl ? modeEl.value : 'face';

  frameCount++;
  if (frameInfo) frameInfo.textContent = `Frame #${frameCount} · Mode: ${mode}`;

  try {
    const result = await fetch('/api/attendance/frame', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: imageB64, mode }),
    }).then(r => r.json());

    handleResult(result);

  } catch (err) {
    console.warn('Scan error:', err);
  }
}

// ── Result handling ────────────────────────────────────────────────────────────

function handleResult(result) {
  if (!result.detected) {
    // No face/iris in frame — show subtle status
    setStatus('Scanning…', 'success');
    hideOverlay();
    return;
  }

  if (!result.matched) {
    setStatus('No match', 'warning');
    showOverlay(result.name || 'Unknown', '', `${result.confidence}%`, 'no-match');
    return;
  }

  // Matched!
  setStatus(`Matched: ${result.name}`, 'success');
  showOverlay(result.name, result.roll_no, `${result.confidence}% confidence`,
              result.already ? 'already' : 'matched');

  if (!result.already && result.new_mark) {
    // New mark — add to list
    addToMarkedList(result);
    showToast(`✓ ${result.name} marked present`, 'success', 3500);
    markedToday.add(result.roll_no);
    updateCounts(1);
  }
  // Already marked — no toast spam
}

function showOverlay(name, roll, conf, cls) {
  if (!matchOverlay) return;
  matchOverlay.classList.remove('d-none');
  matchBox.className = `match-box ${cls}`;
  matchName.textContent = name;
  matchRoll.textContent = roll ? `Roll: ${roll}` : '';
  matchConf.textContent = conf;

  // Auto-hide overlay after 3 seconds
  clearTimeout(matchOverlay._timeout);
  matchOverlay._timeout = setTimeout(hideOverlay, 3000);
}

function hideOverlay() {
  if (matchOverlay) matchOverlay.classList.add('d-none');
}

function addToMarkedList(result) {
  // Remove empty state message if present
  const emptyMsg = document.getElementById('emptyMsg');
  if (emptyMsg) emptyMsg.remove();

  const now = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  const mode = document.querySelector('input[name="mode"]:checked')?.value || 'face';

  const el = document.createElement('div');
  el.className = 'marked-item new-mark';
  el.id = `marked-${result.roll_no}`;
  el.innerHTML = `
    <div class="marked-avatar">${result.name[0].toUpperCase()}</div>
    <div class="marked-info">
      <div class="marked-name">${result.name}</div>
      <div class="marked-meta">
        <code>${result.roll_no}</code> &nbsp;·&nbsp;
        ${now} &nbsp;·&nbsp;
        <span class="badge bg-${mode === 'face' ? 'primary' : 'purple'} badge-sm">${mode}</span>
      </div>
    </div>
  `;

  // Prepend to list
  markedList.insertBefore(el, markedList.firstChild);

  // Remove new-mark highlight after animation
  setTimeout(() => el.classList.remove('new-mark'), 600);
}

function updateCounts(delta) {
  if (!countPresent || !countAbsent || !markedBadge) return;
  const p = parseInt(countPresent.textContent) + delta;
  const a = parseInt(countAbsent.textContent)  - delta;
  countPresent.textContent = p;
  countAbsent.textContent  = Math.max(0, a);
  markedBadge.textContent  = p;
}

// ── Status helper ──────────────────────────────────────────────────────────────

function setStatus(text, type) {
  if (!scanStatus || lastStatus === text) return;
  lastStatus = text;
  scanStatus.className = `badge bg-${type}`;
  scanStatus.textContent = text;
}

// ── Expose toggleCamera globally ───────────────────────────────────────────────
window.toggleCamera = toggleCamera;
