const form = document.getElementById('prediction-form');
const healthResult = document.getElementById('health-result');
const yieldResult = document.getElementById('yield-result');
const scenarioResult = document.getElementById('scenario-result');
const messageBox = document.getElementById('message');
const simulateBtn = document.getElementById('simulate-btn');

function setMessage(text, kind = 'success') {
  messageBox.textContent = text;
  messageBox.className = `message ${kind}`;
}

function getFormData() {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = Number.isFinite(Number(value)) ? Number(value) : value;
  }
  return payload;
}

async function callApi(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCookie('csrftoken') || ''
    },
    body: JSON.stringify(payload)
  });

  const rawText = await response.text();
  let data = {};

  if (rawText) {
    try {
      data = JSON.parse(rawText);
    } catch (error) {
      throw new Error(`Server returned an unexpected response: ${rawText.slice(0, 120)}`);
    }
  }

  if (!response.ok || data.success === false) {
    throw new Error(data.error || 'Request failed');
  }
  return data;
}

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return '';
}

function updateCards(health, yieldValue, scenarioText = '--') {
  healthResult.textContent = health || '--';
  yieldResult.textContent = yieldValue !== undefined && yieldValue !== null ? `${yieldValue.toFixed(3)} tons` : '--';
  scenarioResult.textContent = scenarioText;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setMessage('Running prediction...', 'success');

  try {
    const payload = getFormData();
    const data = await callApi('/api/predict/', payload);
    updateCards(data.crop_health, data.predicted_yield, 'Baseline');
    setMessage('Prediction ready.', 'success');
  } catch (error) {
    updateCards('--', null, '--');
    setMessage(error.message, 'error');
  }
});

simulateBtn.addEventListener('click', async () => {
  setMessage('Comparing what-if scenario...', 'success');

  try {
    const baseline = getFormData();
    const modified = { ...baseline, Irrigation_Hours: Number(baseline.Irrigation_Hours) + 2 };

    const data = await callApi('/api/simulate/', { baseline, modified });
    const delta = data.yield_change;
    updateCards(data.modified.crop_health, data.modified.predicted_yield, `${delta >= 0 ? '+' : ''}${delta.toFixed(3)} vs baseline`);
    setMessage('What-if comparison ready. This is model-based and not a physical simulation.', 'success');
  } catch (error) {
    updateCards('--', null, '--');
    setMessage(error.message, 'error');
  }
});
