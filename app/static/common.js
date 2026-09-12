// Shared across ingest.html and search.html — same origin, same session cookie.
const API_BASE = "";

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Airtable field values aren't always plain strings/numbers -- rollups,
// barcodes, and a few other field types come back as nested objects
// (e.g. {text: "..."} for a barcode, {specialValue: "NaN"} for an empty
// rollup). Naively stringifying those renders "[object Object]" -- this
// pulls out something actually readable instead.
function formatFieldValue(v) {
  if (v === null || v === undefined) return "";
  if (Array.isArray(v)) return v.map(formatFieldValue).join(", ");
  if (typeof v === "object") {
    if ("text" in v) return String(v.text);
    if ("specialValue" in v) return String(v.specialValue);
    if ("value" in v) return String(v.value);
    if ("name" in v) return String(v.name);
    return JSON.stringify(v);
  }
  return String(v);
}

function renderFieldsPreview(fields) {
  const entries = Object.entries(fields || {});
  if (!entries.length) return "";
  return `<div class="fields-preview">${entries.map(([k, v]) =>
    `<div><b>${escapeHtml(k)}:</b> ${escapeHtml(formatFieldValue(v))}</div>`
  ).join("")}</div>`;
}

// ---------- Index status bar ----------
// Text search runs on an in-process FAISS index -- no deployed/billed
// infra to start or stop (see CHANGELOG.md). This just shows the live
// document count as a sanity check that ingests are actually landing in it.
const costMeter = document.getElementById("cost-meter");

async function refreshIndexStatus() {
  if (!costMeter) return;
  try {
    const res = await fetch(`${API_BASE}/admin/faiss-index/status`);
    const data = await res.json();
    costMeter.innerHTML = `<b>${data.text_documents_indexed}</b> text-indexed &middot; <b>${data.image_documents_indexed}</b> image-indexed`;
  } catch (err) {
    costMeter.textContent = `Status unavailable: ${err.message}`;
  }
}

refreshIndexStatus();
setInterval(refreshIndexStatus, 30000);

// ---------- Logout ----------
const logoutBtn = document.getElementById("logout-btn");
if (logoutBtn) {
  logoutBtn.addEventListener("click", async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, { method: "POST" });
    } catch (err) {
      // Ignore -- redirecting to /login either way is the right outcome.
    }
    window.location.href = "/login";
  });
}
