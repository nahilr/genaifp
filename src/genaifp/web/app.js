const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#image-file');
const dropZone = document.querySelector('#drop-zone');
const analyzeButton = document.querySelector('#analyze-button');
const analyzeLabel = document.querySelector('#analyze-label');
const uploadPrompt = document.querySelector('#upload-prompt');
const previewState = document.querySelector('#preview-state');
const imagePreview = document.querySelector('#image-preview');
const fileTitle = document.querySelector('#file-title');
const fileSubtitle = document.querySelector('#file-subtitle');
const setupNote = document.querySelector('#setup-note');
const setupMessage = document.querySelector('#setup-message');
const pageStatus = document.querySelector('#page-status');
const emptyResult = document.querySelector('#empty-result');
const resultContent = document.querySelector('#result-content');
const resultCaption = document.querySelector('#result-caption');

const allowedTypes = new Set(['image/png', 'image/jpeg', 'image/webp']);
const maxBytes = 20 * 1024 * 1024;
const maxPixels = 40_000_000;
let modelReady = false;
let selectedFile = null;
let previewUrl = null;

function report(message) {
  pageStatus.textContent = message;
}

function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.onload = null;
  imagePreview.onerror = null;
  imagePreview.removeAttribute('src');
  previewState.hidden = true;
  uploadPrompt.hidden = false;
  dropZone.classList.remove('has-preview');
}

function resetResult(caption = 'Choose an image to see its result') {
  resultContent.hidden = true;
  emptyResult.hidden = false;
  resultCaption.textContent = caption;
}

function rejectFile(message) {
  selectedFile = null;
  fileInput.value = '';
  clearPreview();
  analyzeButton.disabled = !modelReady;
  resetResult();
  report(message);
}

function setFile(file) {
  if (!file) {
    selectedFile = null;
    clearPreview();
    analyzeButton.disabled = !modelReady;
    resetResult();
    report('Image selection cleared.');
    return;
  }
  if (!allowedTypes.has(file.type)) {
    rejectFile('Choose a PNG, JPEG, or WebP image.');
    return;
  }
  if (file.size > maxBytes) {
    rejectFile('This image is larger than 20 MB. Choose a smaller file.');
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  const currentPreview = previewUrl;
  imagePreview.onload = () => {
    if (previewUrl !== currentPreview) return;
    if (imagePreview.naturalWidth * imagePreview.naturalHeight > maxPixels) {
      rejectFile('This image is larger than 40 megapixels. Choose a smaller image.');
      return;
    }
    fileSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB · ready to analyze`;
  };
  imagePreview.onerror = () => {
    if (previewUrl === currentPreview) rejectFile('This file could not be decoded as a supported image.');
  };
  imagePreview.src = currentPreview;
  uploadPrompt.hidden = true;
  previewState.hidden = false;
  dropZone.classList.add('has-preview');
  fileTitle.textContent = file.name;
  fileSubtitle.textContent = 'Loading preview…';
  analyzeButton.disabled = !modelReady;
  resetResult();
  report(modelReady ? 'Image selected. Ready to analyze.' : 'Image selected. Waiting for the model.');
}

function showSetup(message) {
  modelReady = false;
  setupNote.hidden = false;
  setupMessage.textContent = message;
  analyzeButton.disabled = true;
}

function addText(parent, tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  parent.append(node);
  return node;
}

function renderResult(result) {
  emptyResult.hidden = true;
  resultContent.hidden = false;
  resultCaption.textContent = 'Experimental model output';
  document.querySelector('#prediction-name').textContent = result.predicted_engine || 'Unclassified';
  const verdict = document.querySelector('#verdict');
  verdict.classList.toggle('is-uncertain', result.status === 'uncertain');
  document.querySelector('#verdict-label').textContent = result.status === 'uncertain'
    ? 'Review recommended'
    : 'Closest supported match';
  const score = Number(result.confidence || 0);
  document.querySelector('#prediction-status').textContent = result.status === 'uncertain'
    ? `Uncertain result · top model score ${Math.round(score * 100)}%`
    : `Top model score ${Math.round(score * 100)}%`;

  const list = document.querySelector('#score-list');
  list.replaceChildren();
  const entries = Object.entries(result.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
  for (const [name, value] of entries) {
    const probability = Number(value);
    const row = document.createElement('li');
    row.className = 'score-row';
    addText(row, 'span', 'score-name', name.replaceAll('_', ' '));
    const track = document.createElement('span');
    track.className = 'score-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.className = 'score-fill';
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    track.append(fill);
    row.append(track);
    addText(row, 'span', 'score-value', `${(probability * 100).toFixed(1)}%`);
    list.append(row);
  }

  const provenance = document.querySelector('#provenance');
  provenance.replaceChildren();
  const source = result.model_provenance || {};
  const line = document.createElement('span');
  const label = document.createElement('strong');
  label.textContent = 'Validation source: ';
  line.append(label, document.createTextNode(`${source.dataset_id || 'Unknown dataset'} · ${source.held_out_test_samples ?? '—'} held-out images`));
  provenance.append(line);
  if (source.grouping_method) {
    addText(provenance, 'span', '', 'Validation groups keep shared prompts together; exact image duplicates are removed.');
  }
  if (Array.isArray(source.grouping_limitations) && source.grouping_limitations.length) {
    addText(provenance, 'span', '', source.grouping_limitations.join(' '));
  }
}

fileInput.addEventListener('change', () => setFile(fileInput.files?.[0]));

for (const eventName of ['dragenter', 'dragover']) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add('is-dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove('is-dragging');
  });
}
dropZone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  try {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
  } catch {
    // The local File object still works if assigning to the picker is unsupported.
  }
  setFile(file);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!modelReady || !selectedFile) return;
  const submittedFile = selectedFile;
  const payload = new FormData();
  payload.append('file', submittedFile, submittedFile.name);
  analyzeButton.disabled = true;
  analyzeLabel.textContent = 'Analyzing…';
  form.setAttribute('aria-busy', 'true');
  report('Analyzing image.');
  resultCaption.textContent = 'Analyzing selected image…';
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: payload });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || 'Analysis could not be completed.');
    if (selectedFile !== submittedFile) return;
    renderResult(body);
    report('Analysis complete. The uploaded image was discarded after processing.');
  } catch (error) {
    if (selectedFile === submittedFile) {
      resetResult();
      report(error instanceof Error ? error.message : 'Analysis could not be completed.');
    }
  } finally {
    form.removeAttribute('aria-busy');
    analyzeLabel.textContent = 'Analyze image';
    analyzeButton.disabled = !modelReady || !selectedFile;
  }
});

async function checkReadiness() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    const health = await response.json();
    if (!health.ready) {
      showSetup(health.message || 'Train a model from evaluated real data to enable analysis.');
      report('Analysis is unavailable until an evaluated real-data model is installed.');
      return;
    }
    modelReady = true;
    setupNote.hidden = true;
    analyzeButton.disabled = !selectedFile;
    report(selectedFile ? 'Evaluated model ready. Image is ready to analyze.' : 'Evaluated model ready.');
  } catch {
    showSetup('The local analysis service could not be reached. Restart it and reload this page.');
  }
}

checkReadiness();
