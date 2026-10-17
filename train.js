/**
 * ============================================
 *  🧠 Face Recognition Training System
 *  Teachable Machine Style - by Antigravity
 * ============================================
 * 
 * Reads photos from dataset/ folders and generates
 * face descriptors (embeddings) for recognition.
 * 
 * Usage:
 *   1. Create folders in dataset/ for each person
 *   2. Add 3-10 clear face photos per person
 *   3. Run: npm run train
 * 
 * Output: labeled_descriptors.json
 */

const path = require('path');
const fs = require('fs');

// Load TensorFlow.js Node backend for GPU acceleration
let tf;
try {
  tf = require('@tensorflow/tfjs-node');
} catch (e) {
  console.log('⚠️  TensorFlow.js Node backend not available, falling back to CPU');
  tf = require('@tensorflow/tfjs');
}

const faceapi = require('@vladmandic/face-api');

// Configuration
const DATASET_DIR = path.join(__dirname, 'dataset');
const OUTPUT_FILE = path.join(__dirname, 'labeled_descriptors.json');
const MODEL_DIR = fs.existsSync(path.join(__dirname, 'models'))
  ? path.join(__dirname, 'models')
  : path.join(__dirname, 'node_modules', '@vladmandic', 'face-api', 'model');
const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'];

// Detection options for high accuracy
const DETECTION_OPTIONS = new faceapi.SsdMobilenetv1Options({
  minConfidence: 0.5,
  maxResults: 1
});

/**
 * Load face-api.js models from disk
 */
async function loadModels() {
  console.log('\n📦 Loading AI models...');
  
  if (!fs.existsSync(MODEL_DIR)) {
    console.error('❌ Model directory not found!');
    console.error('   Run "npm install" first to download models.');
    process.exit(1);
  }
  
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_DIR);
  console.log('   ✅ SSD MobileNet v1 (Face Detection)');
  
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_DIR);
  console.log('   ✅ Face Landmark 68 Net (Facial Landmarks)');
  
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_DIR);
  console.log('   ✅ Face Recognition Net (128-D Embeddings)');
  
  console.log('   📊 Model accuracy: ~99.38% on LFW benchmark\n');
}

/**
 * Get image files from a directory
 */
function getImageFiles(dir) {
  return fs.readdirSync(dir).filter(file => {
    const ext = path.extname(file).toLowerCase();
    return SUPPORTED_EXTENSIONS.includes(ext);
  });
}

/**
 * Load an image and return as tensor
 */
async function loadImage(imagePath) {
  const buffer = fs.readFileSync(imagePath);
  if (tf && tf.node && typeof tf.node.decodeImage === 'function') {
    return tf.node.decodeImage(buffer, 3);
  }
  if (faceapi && faceapi.tf && faceapi.tf.node && typeof faceapi.tf.node.decodeImage === 'function') {
    return faceapi.tf.node.decodeImage(buffer, 3);
  }
  throw new Error('TensorFlow Node backend tidak tersedia. Silakan gunakan tombol training langsung di web browser!');
}

/**
 * Process all images for a person and return face descriptors
 */
async function getDescriptorsForPerson(personName, personDir) {
  const imageFiles = getImageFiles(personDir);
  
  if (imageFiles.length === 0) {
    console.log(`   ⚠️  No image files found`);
    return { descriptors: [], stats: { total: 0, detected: 0, failed: 0 } };
  }
  
  console.log(`   📷 Found ${imageFiles.length} image(s)`);
  const descriptors = [];
  let detected = 0;
  let failed = 0;
  
  for (const imgFile of imageFiles) {
    const imgPath = path.join(personDir, imgFile);
    try {
      const tensor = await loadImage(imgPath);
      
      const detection = await faceapi
        .detectSingleFace(tensor, DETECTION_OPTIONS)
        .withFaceLandmarks()
        .withFaceDescriptor();
      
      tf.dispose(tensor);
      
      if (detection) {
        const raw = Array.from(detection.descriptor);
        const norm = Math.sqrt(raw.reduce((s, x) => s + x * x, 0)) || 1.0;
        const normalized = raw.map(x => Number((x / norm).toFixed(6)));
        descriptors.push(normalized);
        detected++;
        const confidence = (detection.detection.score * 100).toFixed(1);
        console.log(`   ✅ ${imgFile} (confidence: ${confidence}%)`);
      } else {
        failed++;
        console.log(`   ❌ ${imgFile} (no face detected)`);
      }
    } catch (err) {
      failed++;
      console.log(`   ❌ ${imgFile} (error: ${err.message})`);
    }
  }
  
  return { 
    descriptors, 
    stats: { total: imageFiles.length, detected, failed } 
  };
}

/**
 * Main training function
 */
async function main() {
  const startTime = Date.now();
  
  console.log('');
  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║     🧠 FACE RECOGNITION TRAINING SYSTEM           ║');
  console.log('║     Teachable Machine Style                        ║');
  console.log('╚════════════════════════════════════════════════════╝');
  
  // Check dataset directory exists
  if (!fs.existsSync(DATASET_DIR)) {
    fs.mkdirSync(DATASET_DIR, { recursive: true });
    console.log('\n❌ Dataset directory was empty! Created "dataset/" folder.');
    console.log('');
    console.log('📁 Please organize your photos like this:');
    console.log('');
    console.log('   dataset/');
    console.log('   ├── Budi/');
    console.log('   │   ├── foto1.jpg');
    console.log('   │   ├── foto2.jpg');
    console.log('   │   └── foto3.jpg');
    console.log('   ├── Andi/');
    console.log('   │   ├── foto1.jpg');
    console.log('   │   └── foto2.jpg');
    console.log('   └── Sari/');
    console.log('       ├── foto1.jpg');
    console.log('       └── foto2.jpg');
    console.log('');
    console.log('💡 Tips for best accuracy (>95%):');
    console.log('   • Use 3-10 photos per person');
    console.log('   • Clear, well-lit photos');
    console.log('   • Face should be clearly visible');
    console.log('   • Include different angles and expressions');
    console.log('   • Minimum image size: 200x200 pixels');
    console.log('');
    return;
  }
  
  // Get person directories
  const persons = fs.readdirSync(DATASET_DIR).filter(d => {
    const fullPath = path.join(DATASET_DIR, d);
    return fs.statSync(fullPath).isDirectory();
  });
  
  if (persons.length === 0) {
    console.log('\n❌ No person folders found in dataset/');
    console.log('   Create subfolders with person names and add their photos inside.');
    console.log('   Example: dataset/Budi/foto1.jpg');
    return;
  }
  
  console.log(`\n👥 Found ${persons.length} person(s): ${persons.join(', ')}\n`);
  console.log('─'.repeat(50));
  
  // Load AI models
  await loadModels();
  
  // Process each person
  const labeledDescriptors = [];
  let totalImages = 0;
  let totalDetected = 0;
  let totalFailed = 0;
  
  for (let i = 0; i < persons.length; i++) {
    const person = persons[i];
    const personDir = path.join(DATASET_DIR, person);
    
    console.log(`\n👤 [${i + 1}/${persons.length}] Processing: ${person}`);
    console.log('─'.repeat(40));
    
    const { descriptors, stats } = await getDescriptorsForPerson(person, personDir);
    
    totalImages += stats.total;
    totalDetected += stats.detected;
    totalFailed += stats.failed;
    
    if (descriptors.length > 0) {
      labeledDescriptors.push({
        label: person,
        descriptors: descriptors
      });
      console.log(`   📊 ${person}: ${stats.detected}/${stats.total} faces detected`);
    } else {
      console.log(`   ⚠️  ${person}: No faces detected! This person won't be recognized.`);
    }
  }
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  
  // Save results
  console.log('\n' + '═'.repeat(50));
  
  if (labeledDescriptors.length > 0) {
    // Create output with metadata
    const output = {
      metadata: {
        createdAt: new Date().toISOString(),
        totalPersons: labeledDescriptors.length,
        totalImages: totalImages,
        totalDetected: totalDetected,
        detectionRate: ((totalDetected / totalImages) * 100).toFixed(1) + '%',
        trainingDuration: elapsed + 's',
        modelVersion: 'SSD MobileNet v1 + ResNet-34'
      },
      descriptors: labeledDescriptors
    };
    
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
    
    console.log('');
    console.log('🎉 TRAINING COMPLETE!');
    console.log('');
    console.log('📊 Training Summary:');
    console.log(`   ├── Persons trained  : ${labeledDescriptors.length}`);
    console.log(`   ├── Total images     : ${totalImages}`);
    console.log(`   ├── Faces detected   : ${totalDetected}`);
    console.log(`   ├── Detection rate   : ${((totalDetected / totalImages) * 100).toFixed(1)}%`);
    console.log(`   ├── Failed           : ${totalFailed}`);
    console.log(`   └── Duration         : ${elapsed}s`);
    console.log('');
    console.log(`📄 Output: ${OUTPUT_FILE}`);
    console.log('');
    console.log('═'.repeat(50));
    console.log('');
    console.log('🚀 Next step: Run "npm start" to launch the website!');
    console.log('   Then open http://localhost:3000 in your browser.');
    console.log('');
    
    // Print accuracy tips
    if (totalDetected / totalImages < 0.8) {
      console.log('⚠️  Detection rate is below 80%. Tips to improve:');
      console.log('   • Use clearer, well-lit photos');
      console.log('   • Make sure faces are clearly visible');
      console.log('   • Avoid photos with multiple faces');
      console.log('   • Use higher resolution images');
      console.log('');
    }
    
    // Print per-person summary
    console.log('📋 Per-person Summary:');
    for (const ld of labeledDescriptors) {
      const count = ld.descriptors.length;
      const quality = count >= 5 ? '🟢 Excellent' : count >= 3 ? '🟡 Good' : '🔴 Low';
      console.log(`   ${ld.label}: ${count} descriptor(s) - ${quality}`);
    }
    console.log('');
    console.log('💡 For best accuracy, aim for 5+ photos per person.');
    console.log('');
  } else {
    console.log('');
    console.log('❌ TRAINING FAILED - No faces were detected!');
    console.log('');
    console.log('📋 Troubleshooting:');
    console.log('   1. Make sure photos clearly show faces');
    console.log('   2. Use good lighting');
    console.log('   3. Face should be front-facing or slight angle');
    console.log('   4. Image should be at least 200x200 pixels');
    console.log('   5. Supported formats: JPG, PNG, WebP, BMP');
    console.log('');
  }
}

// Run training
main().catch(err => {
  console.error('\n❌ Fatal Error:', err.message);
  console.error(err.stack);
  process.exit(1);
});
