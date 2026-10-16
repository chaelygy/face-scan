/**
 * ============================================
 *  🧠 FaceID Pro - Client-Side Application
 *  Face Detection & Recognition Engine
 * ============================================
 */

// ============================================
//  Configuration
// ============================================
const CONFIG = {
  modelUrl: '/models',
  descriptorsUrl: '/api/descriptors',
  datasetInfoUrl: '/api/dataset-info',
  defaultThreshold: 0.6,
  webcamInterval: 200, // ms between webcam detections
  detectionOptions: {
    minConfidence: 0.5
  }
};

// ============================================
//  State
// ============================================
let state = {
  isModelLoaded: false,
  faceMatcher: null,
  labeledDescriptors: [],
  threshold: CONFIG.defaultThreshold,
  webcamStream: null,
  webcamDetectionInterval: null,
  metadata: null
};

// ============================================
//  DOM Elements
// ============================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
  loadingOverlay: $('#loadingOverlay'),
  loadingText: $('#loadingText'),
  loadingBar: $('#loadingBar'),
  statusIndicator: $('#statusIndicator'),
  statusText: $('#statusText'),
  statPersons: $('#statPersons'),
  statImages: $('#statImages'),
  uploadArea: $('#uploadArea'),
  fileInput: $('#fileInput'),
  thresholdSlider: $('#thresholdSlider'),
  thresholdValue: $('#thresholdValue'),
  canvasContainer: $('#canvasContainer'),
  overlayCanvas: $('#overlayCanvas'),
  previewImage: $('#previewImage'),
  emptyState: $('#emptyState'),
  resultsList: $('#resultsList'),
  processingState: $('#processingState'),
  btnClear: $('#btnClear'),
  btnStartWebcam: $('#btnStartWebcam'),
  btnStopWebcam: $('#btnStopWebcam'),
  webcamVideo: $('#webcamVideo'),
  webcamCanvas: $('#webcamCanvas'),
  webcamPlaceholder: $('#webcamPlaceholder'),
  webcamResults: $('#webcamResults'),
  toastContainer: $('#toastContainer'),
  // Dataset tab
  dsPersonCount: $('#dsPersonCount'),
  dsImageCount: $('#dsImageCount'),
  dsDescCount: $('#dsDescCount'),
  dsTrainDate: $('#dsTrainDate'),
  datasetPersons: $('#datasetPersons'),
  // Training tab
  btnRefreshDataset: $('#btnRefreshDataset'),
  btnStartTraining: $('#btnStartTraining'),
  trainingStatusBox: $('#trainingStatusBox'),
  trainingCurrentPerson: $('#trainingCurrentPerson'),
  trainingProgressText: $('#trainingProgressText'),
  trainingProgressBar: $('#trainingProgressBar'),
  trainingPreviewThumb: $('#trainingPreviewThumb'),
  trainingLiveLog: $('#trainingLiveLog'),
  trainPersonsList: $('#trainPersonsList')
};

// ============================================
//  Initialization
// ============================================
document.addEventListener('DOMContentLoaded', init);

async function init() {
  setupEventListeners();
  await loadAIModels();
  await loadDescriptors();
  await loadDatasetInfo();
  hideLoading();
}

// ============================================
//  Model Loading
// ============================================
async function loadAIModels() {
  try {
    updateLoading('Loading Face Detection Model...', 10);
    await faceapi.nets.ssdMobilenetv1.loadFromUri(CONFIG.modelUrl);
    
    updateLoading('Loading Face Landmark Model...', 40);
    await faceapi.nets.faceLandmark68Net.loadFromUri(CONFIG.modelUrl);
    
    updateLoading('Loading Face Recognition Model...', 70);
    await faceapi.nets.faceRecognitionNet.loadFromUri(CONFIG.modelUrl);
    
    updateLoading('AI Models loaded successfully!', 90);
    state.isModelLoaded = true;
    
  } catch (error) {
    console.error('Failed to load AI models:', error);
    updateLoading('Failed to load AI models!', 0);
    showToast('Gagal memuat AI model. Pastikan server berjalan.', 'error');
    updateStatus('error', 'Error');
  }
}

// ============================================
//  Descriptors Loading
// ============================================
async function loadDescriptors() {
  try {
    updateLoading('Loading training data...', 95);
    
    const response = await fetch(CONFIG.descriptorsUrl);
    
    if (!response.ok) {
      if (response.status === 404) {
        showToast('Belum ada data training. Jalankan "npm run train" terlebih dahulu.', 'warning');
        updateStatus('error', 'No Data');
        return;
      }
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    state.metadata = data.metadata || null;
    
    const descriptorsData = data.descriptors || data;
    
    if (!Array.isArray(descriptorsData) || descriptorsData.length === 0) {
      showToast('Data training kosong. Tambahkan foto dan jalankan training.', 'warning');
      updateStatus('error', 'Empty Data');
      return;
    }
    
    // Convert to face-api.js LabeledFaceDescriptors
    state.labeledDescriptors = descriptorsData.map(item => {
      const descriptors = item.descriptors.map(d => new Float32Array(d));
      return new faceapi.LabeledFaceDescriptors(item.label, descriptors);
    });
    
    // Create FaceMatcher
    state.faceMatcher = new faceapi.FaceMatcher(
      state.labeledDescriptors, 
      state.threshold
    );
    
    // Update stats
    const totalDescs = descriptorsData.reduce((sum, item) => sum + item.descriptors.length, 0);
    DOM.statPersons.textContent = descriptorsData.length;
    DOM.dsPersonCount.textContent = descriptorsData.length;
    DOM.dsDescCount.textContent = totalDescs;
    
    // Update metadata display
    if (state.metadata) {
      DOM.statImages.textContent = state.metadata.totalImages || 0;
      DOM.dsImageCount.textContent = state.metadata.totalImages || 0;
      if (state.metadata.createdAt) {
        const date = new Date(state.metadata.createdAt);
        DOM.dsTrainDate.textContent = date.toLocaleDateString('id-ID', {
          day: 'numeric', month: 'short', year: 'numeric'
        });
      }
    }
    
    // Update dataset persons list
    updateDatasetPersonsList(descriptorsData);
    
    updateStatus('ready', 'Ready');
    showToast(`${descriptorsData.length} orang berhasil dimuat ke dalam sistem.`, 'success');
    
  } catch (error) {
    console.error('Failed to load descriptors:', error);
    showToast('Gagal memuat data training.', 'error');
    updateStatus('error', 'Error');
  }
}

async function loadDatasetInfo() {
  try {
    const response = await fetch(CONFIG.datasetInfoUrl);
    if (!response.ok) return;
    
    const data = await response.json();
    if (data.totalImages !== undefined) {
      DOM.statImages.textContent = data.totalImages;
      DOM.dsImageCount.textContent = data.totalImages;
    }
    
    // Render classes/folders in Training Tab
    if (DOM.trainPersonsList) {
      DOM.trainPersonsList.innerHTML = '';
      if (!data.persons || data.persons.length === 0) {
        DOM.trainPersonsList.innerHTML = `
          <div style="grid-column: 1/-1; padding: 25px; text-align:center; color: var(--text-muted); background: var(--bg-glass); border-radius: 12px; border: 1px dashed var(--border-primary);">
            <i class="ph ph-folder-dashed" style="font-size: 2.2rem; margin-bottom: 8px; display: block; color: var(--accent-primary);"></i>
            <strong style="color: var(--text-primary); display:block; margin-bottom:4px;">Folder dataset masih kosong</strong>
            Buka folder <code>dataset/</code> di laptop Anda, buat folder nama teman (misal: <code>dataset/Budi/</code>), lalu masukkan foto-foto mereka.
          </div>
        `;
      } else {
        const colors = [
          'linear-gradient(135deg, #6366f1, #8b5cf6)',
          'linear-gradient(135deg, #8b5cf6, #d946ef)',
          'linear-gradient(135deg, #ec4899, #f43f5e)',
          'linear-gradient(135deg, #f59e0b, #ef4444)',
          'linear-gradient(135deg, #10b981, #3b82f6)'
        ];
        data.persons.forEach((p, idx) => {
          const card = document.createElement('div');
          card.className = 'person-card';
          const thumbHtml = p.images && p.images.length > 0 
            ? `<img src="${p.images[0]}" style="width:40px; height:40px; border-radius:8px; object-fit:cover;">`
            : `<div class="person-avatar" style="background: ${colors[idx % colors.length]}">${p.name.substring(0,2).toUpperCase()}</div>`;
          
          card.innerHTML = `
            ${thumbHtml}
            <div class="person-info">
              <div class="person-name">${p.name}</div>
              <div class="person-count">${p.imageCount} foto ditemukan</div>
            </div>
          `;
          DOM.trainPersonsList.appendChild(card);
        });
      }
    }
    
    return data;
  } catch (e) {
    console.warn('Dataset info load error:', e);
  }
}

// ============================================
//  In-Browser Training Engine
// ============================================
async function startDatasetTraining() {
  if (!state.isModelLoaded) {
    showToast('AI Model belum siap. Tunggu hingga loading selesai.', 'warning');
    return;
  }

  DOM.btnStartTraining.disabled = true;
  DOM.btnStartTraining.innerHTML = '<i class="ph ph-spinner"></i> Sedang Melatih...';
  DOM.trainingStatusBox.style.display = 'block';

  try {
    const datasetData = await loadDatasetInfo();
    if (!datasetData || !datasetData.persons || datasetData.persons.length === 0) {
      showToast('Tidak ada folder person di dataset/!', 'error');
      DOM.trainingLiveLog.innerHTML = '<span style="color:var(--danger)">Folder dataset kosong. Buat folder nama teman terlebih dahulu.</span>';
      DOM.btnStartTraining.disabled = false;
      DOM.btnStartTraining.innerHTML = '<i class="ph ph-play"></i> Mulai Latih Dataset';
      return;
    }

    const totalImages = datasetData.totalImages;
    if (totalImages === 0) {
      showToast('Tidak ada foto yang ditemukan di dalam folder dataset.', 'warning');
      DOM.btnStartTraining.disabled = false;
      DOM.btnStartTraining.innerHTML = '<i class="ph ph-play"></i> Mulai Latih Dataset';
      return;
    }

    let processedCount = 0;
    let detectedCount = 0;
    const labeledDescriptors = [];
    const startTime = Date.now();

    for (const person of datasetData.persons) {
      DOM.trainingCurrentPerson.textContent = `Memproses: ${person.name}`;
      const descriptors = [];

      for (const imgUrl of person.images) {
        processedCount++;
        const pct = Math.round((processedCount / totalImages) * 100);
        DOM.trainingProgressBar.style.width = `${pct}%`;
        DOM.trainingProgressText.textContent = `${pct}%`;

        // Update preview image
        DOM.trainingPreviewThumb.innerHTML = `<img src="${imgUrl}" style="width:100%; height:100%; object-fit:cover;">`;
        DOM.trainingLiveLog.textContent = `Menganalisis wajah: ${person.name} (${processedCount}/${totalImages})...`;

        try {
          const img = await new Promise((resolve, reject) => {
            const i = new Image();
            i.onload = () => resolve(i);
            i.onerror = () => reject(new Error('Gagal load gambar'));
            i.src = imgUrl;
          });

          // Detect face with ResNet-34 descriptor
          const detection = await faceapi
            .detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detection) {
            descriptors.push(Array.from(detection.descriptor));
            detectedCount++;
            const score = Math.round(detection.detection.score * 100);
            DOM.trainingLiveLog.innerHTML = `✅ Wajah <strong>${person.name}</strong> terdeteksi (${score}% confidence)`;
          } else {
            DOM.trainingLiveLog.innerHTML = `⚠️ <span style="color:var(--warning)">Wajah tidak terdeteksi pada ${imgUrl}</span>`;
          }
        } catch (err) {
          console.error(err);
          DOM.trainingLiveLog.innerHTML = `❌ Error pada file: ${imgUrl}`;
        }

        // Small yield for UI refresh
        await new Promise(r => setTimeout(r, 60));
      }

      if (descriptors.length > 0) {
        labeledDescriptors.push({
          label: person.name,
          descriptors: descriptors
        });
      }
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    const detectionRate = Math.round((detectedCount / totalImages) * 100);

    const outputData = {
      metadata: {
        createdAt: new Date().toISOString(),
        totalPersons: labeledDescriptors.length,
        totalImages: totalImages,
        totalDetected: detectedCount,
        detectionRate: `${detectionRate}%`,
        trainingDuration: `${duration}s`,
        modelVersion: 'SSD MobileNet v1 + ResNet-34'
      },
      descriptors: labeledDescriptors
    };

    // Save to server
    DOM.trainingLiveLog.textContent = 'Menyimpan model ke labeled_descriptors.json...';
    const saveRes = await fetch('/api/save-descriptors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(outputData)
    });

    if (saveRes.ok) {
      await loadDescriptors();
      showToast(`🎉 Training Selesai! ${detectedCount} wajah berhasil dipelajari (${detectionRate}% akurasi deteksi).`, 'success');
      DOM.trainingLiveLog.innerHTML = `<strong>🎉 Selesai!</strong> Berhasil melatih ${labeledDescriptors.length} orang (${detectedCount}/${totalImages} wajah) dalam ${duration} detik. Siap digunakan!`;
    } else {
      showToast('Gagal menyimpan file hasil training.', 'error');
    }

  } catch (error) {
    console.error('Training error:', error);
    showToast('Terjadi kesalahan selama training.', 'error');
  } finally {
    DOM.btnStartTraining.disabled = false;
    DOM.btnStartTraining.innerHTML = '<i class="ph ph-play"></i> Latih Ulang Dataset';
  }
}

// ============================================
//  Event Listeners
// ============================================
function setupEventListeners() {
  // Tab navigation
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  
  // File upload
  DOM.uploadArea.addEventListener('click', () => DOM.fileInput.click());
  DOM.fileInput.addEventListener('change', handleFileSelect);
  
  // Drag & drop
  DOM.uploadArea.addEventListener('dragover', handleDragOver);
  DOM.uploadArea.addEventListener('dragleave', handleDragLeave);
  DOM.uploadArea.addEventListener('drop', handleDrop);
  
  // Paste from clipboard
  document.addEventListener('paste', handlePaste);
  
  // Threshold slider
  DOM.thresholdSlider.addEventListener('input', handleThresholdChange);
  
  // Clear button
  DOM.btnClear.addEventListener('click', clearResults);
  
  // Webcam
  DOM.btnStartWebcam.addEventListener('click', startWebcam);
  DOM.btnStopWebcam.addEventListener('click', stopWebcam);

  // Training
  if (DOM.btnRefreshDataset) {
    DOM.btnRefreshDataset.addEventListener('click', async () => {
      showToast('Memindai folder dataset...', 'info');
      await loadDatasetInfo();
      showToast('Daftar folder dataset berhasil diperbarui.', 'success');
    });
  }
  if (DOM.btnStartTraining) {
    DOM.btnStartTraining.addEventListener('click', startDatasetTraining);
  }
}

// ============================================
//  Tab Navigation
// ============================================
function switchTab(tabName) {
  // Update buttons
  $$('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  
  // Update content
  $$('.tab-content').forEach(content => {
    content.classList.remove('active');
  });
  
  const targetTab = $(`#${tabName}Tab`);
  if (targetTab) {
    targetTab.classList.add('active');
  }
  
  // Stop webcam if switching away
  if (tabName !== 'webcam' && state.webcamStream) {
    stopWebcam();
  }
}

// ============================================
//  File Upload Handlers
// ============================================
function handleDragOver(e) {
  e.preventDefault();
  DOM.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
  e.preventDefault();
  DOM.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  DOM.uploadArea.classList.remove('drag-over');
  
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    processFile(files[0]);
  }
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files.length > 0) {
    processFile(files[0]);
  }
  // Reset input so the same file can be selected again
  e.target.value = '';
}

function handlePaste(e) {
  const items = e.clipboardData?.items;
  if (!items) return;
  
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) {
        processFile(file);
        showToast('Foto dari clipboard berhasil ditempel!', 'info');
      }
      break;
    }
  }
}

// ============================================
//  Face Recognition Processing
// ============================================
async function processFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('File harus berupa gambar (JPG, PNG, WebP)', 'error');
    return;
  }
  
  if (file.size > 10 * 1024 * 1024) {
    showToast('Ukuran file terlalu besar (max 10MB)', 'error');
    return;
  }
  
  if (!state.isModelLoaded) {
    showToast('AI model belum selesai dimuat. Tunggu sebentar...', 'warning');
    return;
  }
  
  if (!state.faceMatcher) {
    showToast('Belum ada data training. Jalankan "npm run train" terlebih dahulu.', 'warning');
    return;
  }
  
  // Show processing state
  showProcessing();
  
  try {
    // Load image
    const img = await loadImageFromFile(file);
    
    // Display preview
    DOM.previewImage.src = img.src;
    DOM.previewImage.onload = async () => {
      // Setup canvas
      const canvas = DOM.overlayCanvas;
      canvas.width = DOM.previewImage.naturalWidth;
      canvas.height = DOM.previewImage.naturalHeight;
      
      // Detect faces
      const detections = await faceapi
        .detectAllFaces(DOM.previewImage, new faceapi.SsdMobilenetv1Options(CONFIG.detectionOptions))
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      if (detections.length === 0) {
        hideProcessing();
        showToast('Tidak ada wajah terdeteksi dalam foto ini.', 'warning');
        DOM.canvasContainer.style.display = 'block';
        DOM.emptyState.style.display = 'none';
        return;
      }
      
      // Match faces
      const results = detections.map(detection => {
        const match = state.faceMatcher.findBestMatch(detection.descriptor);
        return {
          detection: detection,
          match: match
        };
      });
      
      // Draw results
      drawDetections(canvas, results);
      displayResults(results);
      
      hideProcessing();
      DOM.canvasContainer.style.display = 'block';
      DOM.emptyState.style.display = 'none';
      DOM.resultsList.style.display = 'flex';
      DOM.btnClear.style.display = 'flex';
      
      const recognized = results.filter(r => r.match.label !== 'unknown').length;
      showToast(
        `${detections.length} wajah terdeteksi, ${recognized} dikenali.`, 
        recognized > 0 ? 'success' : 'info'
      );
    };
    
  } catch (error) {
    console.error('Processing error:', error);
    hideProcessing();
    showToast('Terjadi kesalahan saat memproses foto.', 'error');
  }
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ============================================
//  Drawing & Display
// ============================================
function drawDetections(canvas, results) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  results.forEach((result, index) => {
    const box = result.detection.detection.box;
    const label = result.match.label;
    const distance = result.match.distance;
    const confidence = Math.max(0, ((1 - distance) * 100)).toFixed(1);
    const isKnown = label !== 'unknown';
    
    // Colors
    const color = isKnown ? '#10b981' : '#ef4444';
    const bgColor = isKnown ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
    
    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    
    // Draw corner accents
    const cornerLength = Math.min(box.width, box.height) * 0.15;
    ctx.lineWidth = 4;
    
    // Top-left
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + cornerLength);
    ctx.lineTo(box.x, box.y);
    ctx.lineTo(box.x + cornerLength, box.y);
    ctx.stroke();
    
    // Top-right
    ctx.beginPath();
    ctx.moveTo(box.x + box.width - cornerLength, box.y);
    ctx.lineTo(box.x + box.width, box.y);
    ctx.lineTo(box.x + box.width, box.y + cornerLength);
    ctx.stroke();
    
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + box.height - cornerLength);
    ctx.lineTo(box.x, box.y + box.height);
    ctx.lineTo(box.x + cornerLength, box.y + box.height);
    ctx.stroke();
    
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(box.x + box.width - cornerLength, box.y + box.height);
    ctx.lineTo(box.x + box.width, box.y + box.height);
    ctx.lineTo(box.x + box.width, box.y + box.height - cornerLength);
    ctx.stroke();
    
    // Draw semi-transparent fill
    ctx.fillStyle = bgColor;
    ctx.fillRect(box.x, box.y, box.width, box.height);
    
    // Draw subtle landmark points if available
    if (result.detection && result.detection.landmarks) {
      const landmarks = result.detection.landmarks;
      ctx.fillStyle = 'rgba(99, 241, 218, 0.75)';
      const positions = landmarks.positions || [];
      positions.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw label background
    const labelText = isKnown ? `${label} (${confidence}%)` : `Tidak Dikenali`;
    const fontSize = Math.max(13, Math.min(20, box.width * 0.065));
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    const textMetrics = ctx.measureText(labelText);
    const labelHeight = fontSize + 12;
    const labelPadding = 10;
    const labelWidth = textMetrics.width + labelPadding * 2;
    
    // Label position (above box, or inside box if too close to top)
    const labelX = Math.max(0, box.x);
    const labelY = (box.y - labelHeight - 6 < 0) ? (box.y + 6) : (box.y - labelHeight - 6);
    
    // Draw label bg
    ctx.fillStyle = color;
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(labelX, labelY, labelWidth, labelHeight, 6);
      ctx.fill();
    } else {
      ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    }
    
    // Draw label text
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, labelX + labelPadding, labelY + labelHeight / 2);
  });
}

function displayResults(results) {
  DOM.resultsList.innerHTML = '';
  
  results.forEach((result, index) => {
    const label = result.match.label;
    const distance = result.match.distance;
    const confidence = Math.max(0, ((1 - distance) * 100)).toFixed(1);
    const isKnown = label !== 'unknown';
    
    let confidenceClass = 'low';
    if (confidence >= 80) confidenceClass = 'high';
    else if (confidence >= 60) confidenceClass = 'medium';
    
    const initials = isKnown ? label.substring(0, 2).toUpperCase() : '?';
    
    const item = document.createElement('div');
    item.className = 'result-item';
    item.innerHTML = `
      <div class="result-avatar ${isKnown ? '' : 'unknown'}">
        ${initials}
      </div>
      <div class="result-info">
        <div class="result-name ${isKnown ? '' : 'unknown'}">
          ${isKnown ? label : 'Tidak Dikenali'}
        </div>
        <div class="result-distance">
          Distance: ${distance.toFixed(4)}
        </div>
        <div class="confidence-bar">
          <div class="confidence-bar-fill ${confidenceClass}" 
               style="width: 0%"
               data-target-width="${confidence}%"></div>
        </div>
      </div>
      <div class="result-confidence">
        <div class="confidence-value ${confidenceClass}">${confidence}%</div>
        <div class="confidence-label">Confidence</div>
      </div>
    `;
    
    DOM.resultsList.appendChild(item);
    
    // Animate confidence bar
    requestAnimationFrame(() => {
      setTimeout(() => {
        const bar = item.querySelector('.confidence-bar-fill');
        bar.style.width = bar.dataset.targetWidth;
      }, 100 + index * 100);
    });
  });
}

function clearResults() {
  DOM.canvasContainer.style.display = 'none';
  DOM.resultsList.style.display = 'none';
  DOM.resultsList.innerHTML = '';
  DOM.emptyState.style.display = 'block';
  DOM.btnClear.style.display = 'none';
  DOM.previewImage.src = '';
  
  const ctx = DOM.overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.overlayCanvas.width, DOM.overlayCanvas.height);
}

// ============================================
//  Webcam
// ============================================
async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Akses kamera tidak diizinkan di protokol ini. Pastikan Anda membuka web lewat http://localhost:5000 (bukan file:///...)', 'error');
    return;
  }

  if (!state.isModelLoaded) {
    showToast('AI model belum selesai dimuat. Tunggu sebentar.', 'warning');
    return;
  }
  
  if (!state.faceMatcher) {
    showToast('Info: Dataset belum dilatih. Anda bisa melatih foto di tab "Latih Model". Wajah akan terdeteksi sebagai "Tidak Dikenali".', 'info');
  }
  
  try {
    try {
      state.webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
    } catch (constraintError) {
      console.warn('Initial webcam constraint failed, falling back to basic video:', constraintError);
      state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    }
    
    DOM.webcamVideo.srcObject = state.webcamStream;
    DOM.webcamVideo.classList.add('active');
    DOM.webcamPlaceholder.style.display = 'none';
    DOM.btnStartWebcam.style.display = 'none';
    DOM.btnStopWebcam.style.display = 'inline-flex';
    
    showToast('Webcam aktif! Mendeteksi wajah...', 'success');
    
    // Wait for video to be ready
    DOM.webcamVideo.addEventListener('playing', () => {
      // Setup canvas
      DOM.webcamCanvas.width = DOM.webcamVideo.videoWidth;
      DOM.webcamCanvas.height = DOM.webcamVideo.videoHeight;
      
      // Start detection loop
      state.webcamDetectionInterval = setInterval(detectWebcamFaces, CONFIG.webcamInterval);
    }, { once: true });
    
  } catch (error) {
    console.error('Webcam error:', error);
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      showToast('Izin kamera ditolak! Klik ikon gembok/kamera di sebelah kiri alamat URL browser, lalu ubah Izin Kamera menjadi "Izinkan" (Allow) dan refresh.', 'error');
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      showToast('Kamera sedang dipakai aplikasi lain (Zoom, Teams, OBS, atau tab browser lain). Tutup aplikasi tersebut.', 'error');
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      showToast('Webcam tidak ditemukan. Pastikan kamera terpasang dan switch privasi fisik kamera di laptop dalam posisi terbuka.', 'error');
    } else {
      showToast(`Gagal mengakses webcam: ${error.message || error.name}. Pastikan izin kamera aktif.`, 'error');
    }
  }
}

function stopWebcam() {
  if (state.webcamDetectionInterval) {
    clearInterval(state.webcamDetectionInterval);
    state.webcamDetectionInterval = null;
  }
  
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
  
  DOM.webcamVideo.srcObject = null;
  DOM.webcamVideo.classList.remove('active');
  DOM.webcamPlaceholder.style.display = 'flex';
  DOM.btnStartWebcam.style.display = 'inline-flex';
  DOM.btnStopWebcam.style.display = 'none';
  DOM.webcamResults.innerHTML = '';
  
  const ctx = DOM.webcamCanvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.webcamCanvas.width, DOM.webcamCanvas.height);
  
  showToast('Webcam dimatikan.', 'info');
}

async function detectWebcamFaces() {
  if (!DOM.webcamVideo.videoWidth) return;
  
  try {
    const detections = await faceapi
      .detectAllFaces(DOM.webcamVideo, new faceapi.SsdMobilenetv1Options(CONFIG.detectionOptions))
      .withFaceLandmarks()
      .withFaceDescriptors();
    
    const canvas = DOM.webcamCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (detections.length === 0) {
      DOM.webcamResults.innerHTML = '<div class="result-item"><div class="result-info"><div class="result-name unknown">Tidak ada wajah terdeteksi</div></div></div>';
      return;
    }
    
    // Resize detections to match display size
    const displaySize = { 
      width: DOM.webcamVideo.videoWidth, 
      height: DOM.webcamVideo.videoHeight 
    };
    
    const results = detections.map(detection => {
      const match = state.faceMatcher 
        ? state.faceMatcher.findBestMatch(detection.descriptor)
        : { label: 'unknown', distance: 1.0 };
      return { detection, match };
    });
    
    // Draw on canvas
    drawDetections(canvas, results);
    
    // Update results sidebar
    DOM.webcamResults.innerHTML = results.map(result => {
      const label = result.match.label;
      const distance = result.match.distance;
      const confidence = Math.max(0, ((1 - distance) * 100)).toFixed(1);
      const isKnown = label !== 'unknown';
      const initials = isKnown ? label.substring(0, 2).toUpperCase() : '?';
      
      let confidenceClass = 'low';
      if (confidence >= 80) confidenceClass = 'high';
      else if (confidence >= 60) confidenceClass = 'medium';
      
      return `
        <div class="result-item">
          <div class="result-avatar ${isKnown ? '' : 'unknown'}">${initials}</div>
          <div class="result-info">
            <div class="result-name ${isKnown ? '' : 'unknown'}">
              ${isKnown ? label : 'Tidak Dikenali'}
            </div>
          </div>
          <div class="result-confidence">
            <div class="confidence-value ${confidenceClass}">${confidence}%</div>
          </div>
        </div>
      `;
    }).join('');
    
  } catch (error) {
    // Silently fail on detection errors during webcam
  }
}

// ============================================
//  Threshold Control
// ============================================
function handleThresholdChange(e) {
  state.threshold = parseFloat(e.target.value);
  DOM.thresholdValue.textContent = state.threshold.toFixed(2);
  
  // Recreate FaceMatcher with new threshold
  if (state.labeledDescriptors.length > 0) {
    state.faceMatcher = new faceapi.FaceMatcher(
      state.labeledDescriptors, 
      state.threshold
    );
  }
}

// ============================================
//  Dataset Display
// ============================================
function updateDatasetPersonsList(descriptorsData) {
  DOM.datasetPersons.innerHTML = '';
  
  const colors = [
    'linear-gradient(135deg, #6366f1, #8b5cf6)',
    'linear-gradient(135deg, #8b5cf6, #d946ef)',
    'linear-gradient(135deg, #ec4899, #f43f5e)',
    'linear-gradient(135deg, #f59e0b, #ef4444)',
    'linear-gradient(135deg, #10b981, #3b82f6)',
    'linear-gradient(135deg, #06b6d4, #6366f1)',
    'linear-gradient(135deg, #14b8a6, #10b981)',
    'linear-gradient(135deg, #f97316, #eab308)',
  ];
  
  descriptorsData.forEach((person, index) => {
    const card = document.createElement('div');
    card.className = 'person-card';
    card.innerHTML = `
      <div class="person-avatar" style="background: ${colors[index % colors.length]}">
        ${person.label.substring(0, 2).toUpperCase()}
      </div>
      <div class="person-info">
        <div class="person-name">${person.label}</div>
        <div class="person-count">${person.descriptors.length} descriptor(s)</div>
      </div>
    `;
    DOM.datasetPersons.appendChild(card);
  });
}

// ============================================
//  UI Helpers
// ============================================
function updateLoading(text, progress) {
  if (DOM.loadingText) DOM.loadingText.textContent = text;
  if (DOM.loadingBar) DOM.loadingBar.style.width = progress + '%';
}

function hideLoading() {
  updateLoading('Selesai!', 100);
  setTimeout(() => {
    DOM.loadingOverlay.classList.add('hidden');
  }, 500);
}

function showProcessing() {
  DOM.emptyState.style.display = 'none';
  DOM.canvasContainer.style.display = 'none';
  DOM.resultsList.style.display = 'none';
  DOM.processingState.style.display = 'block';
}

function hideProcessing() {
  DOM.processingState.style.display = 'none';
}

function updateStatus(status, text) {
  DOM.statusIndicator.className = 'nav-stat status-indicator ' + status;
  DOM.statusText.textContent = text;
}

function showToast(message, type = 'info') {
  const icons = {
    success: 'ph ph-check-circle',
    error: 'ph ph-x-circle',
    warning: 'ph ph-warning',
    info: 'ph ph-info'
  };
  
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="toast-icon ${icons[type] || icons.info}"></i>
    <span class="toast-message">${message}</span>
  `;
  
  DOM.toastContainer.appendChild(toast);
  
  // Auto-remove after 4 seconds
  setTimeout(() => {
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
