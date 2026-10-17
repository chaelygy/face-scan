/**
 * ============================================
 *  🌐 Face Recognition Web Server (Express)
 *  Serves the website, model files, and APIs
 * ============================================
 */

const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// JSON and Body Parser (large limit for face descriptors and image uploads)
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// Directories
const BASE_DIR = __dirname;
const DATASET_DIR = path.join(BASE_DIR, 'dataset');
const DESCRIPTORS_FILE = path.join(BASE_DIR, 'labeled_descriptors.json');
const DATASET_INFO_FILE = path.join(BASE_DIR, 'dataset_info.json');

// Ensure dataset directory exists
if (!fs.existsSync(DATASET_DIR)) {
  fs.mkdirSync(DATASET_DIR, { recursive: true });
}

// Serve explicit directories
const libDir = fs.existsSync(path.join(BASE_DIR, 'lib'))
  ? path.join(BASE_DIR, 'lib')
  : path.join(BASE_DIR, 'node_modules', '@vladmandic', 'face-api', 'dist');
app.use('/lib', express.static(libDir));

const modelsDir = fs.existsSync(path.join(BASE_DIR, 'models'))
  ? path.join(BASE_DIR, 'models')
  : path.join(BASE_DIR, 'node_modules', '@vladmandic', 'face-api', 'model');
app.use('/models', express.static(modelsDir));

app.use('/dataset', express.static(DATASET_DIR));

// Serve static website files from root
app.use(express.static(BASE_DIR));

// ============================================
//  API Endpoints
// ============================================

// API: Get labeled descriptors
app.get('/api/descriptors', (req, res) => {
  if (fs.existsSync(DESCRIPTORS_FILE)) {
    try {
      const data = JSON.parse(fs.readFileSync(DESCRIPTORS_FILE, 'utf8'));
      return res.json(data);
    } catch (err) {
      console.error('Error reading descriptors:', err);
      return res.status(500).json({ error: 'Failed to read descriptors file' });
    }
  }
  res.json({
    metadata: { totalPersons: 0, totalImages: 0 },
    descriptors: []
  });
});

// API: Save labeled descriptors (POST)
app.post('/api/save-descriptors', (req, res) => {
  try {
    const data = req.body;
    if (!data) {
      return res.status(400).json({ success: false, error: 'No data provided' });
    }
    fs.writeFileSync(DESCRIPTORS_FILE, JSON.stringify(data, null, 2), 'utf8');
    console.log(`✅ Saved ${data.descriptors ? data.descriptors.length : 0} face descriptors to labeled_descriptors.json`);
    res.json({ success: true, message: 'Model dataset berhasil disimpan!' });
  } catch (error) {
    console.error('Error saving descriptors:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// API: Get dataset info
app.get('/api/dataset-info', (req, res) => {
  if (!fs.existsSync(DATASET_DIR)) {
    return res.json({ persons: [], totalImages: 0 });
  }

  const validExts = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']);
  try {
    const names = fs.readdirSync(DATASET_DIR).filter(d => {
      const p = path.join(DATASET_DIR, d);
      return fs.statSync(p).isDirectory();
    }).sort();

    const persons = [];
    let totalImages = 0;

    for (const name of names) {
      const personDir = path.join(DATASET_DIR, name);
      const files = fs.readdirSync(personDir).filter(f => {
        const ext = path.extname(f).toLowerCase();
        return validExts.has(ext);
      }).sort();

      const images = files.map(f => `/dataset/${encodeURIComponent(name)}/${encodeURIComponent(f)}`);
      persons.append ? null : persons.push({
        name,
        imageCount: images.length,
        images
      });
      totalImages += images.length;
    }

    const result = { persons, totalImages };

    // Also update dataset_info.json cache
    try {
      fs.writeFileSync(DATASET_INFO_FILE, JSON.stringify(result, null, 2), 'utf8');
    } catch (e) {
      // Ignore cache write errors
    }

    res.json(result);
  } catch (error) {
    console.error('Error scanning dataset directory:', error);
    // Fallback to static dataset_info.json if exists
    if (fs.existsSync(DATASET_INFO_FILE)) {
      const data = JSON.parse(fs.readFileSync(DATASET_INFO_FILE, 'utf8'));
      return res.json(data);
    }
    res.status(500).json({ error: error.message, persons: [], totalImages: 0 });
  }
});

// API: Upload image(s) to dataset folder
app.post('/api/upload-images', (req, res) => {
  try {
    const { personName, images } = req.body;
    if (!personName || !images || !Array.isArray(images) || images.length === 0) {
      return res.status(400).json({ success: false, error: 'Data personName atau images tidak valid' });
    }

    // Sanitize person name
    const safeName = personName.trim().replace(/[<>:"/\\|?*]/g, '_');
    const personDir = path.join(DATASET_DIR, safeName);
    if (!fs.existsSync(personDir)) {
      fs.mkdirSync(personDir, { recursive: true });
    }

    let savedCount = 0;
    const timestamp = Date.now();

    images.forEach((imgData, index) => {
      // imgData should be base64 data url, e.g. "data:image/jpeg;base64,..."
      let base64 = imgData;
      let ext = '.jpg';
      const match = imgData.match(/^data:image\/([a-zA-Z0-9+]+);base64,(.+)$/);
      if (match) {
        ext = match[1] === 'jpeg' ? '.jpg' : `.${match[1]}`;
        base64 = match[2];
      }
      const buffer = Buffer.from(base64, 'base64');
      const filename = `photo_${timestamp}_${index + 1}${ext}`;
      fs.writeFileSync(path.join(personDir, filename), buffer);
      savedCount++;
    });

    console.log(`✅ Uploaded ${savedCount} image(s) for ${safeName}`);
    res.json({ success: true, count: savedCount, personName: safeName });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log('');
  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║     🌐 FaceID Pro - Web Server Running             ║');
  console.log('╚════════════════════════════════════════════════════╝');
  console.log(`🚀 Server berjalan di: http://localhost:${PORT}`);
  console.log('');
});
