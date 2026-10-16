/**
 * ============================================
 *  🌐 Face Recognition Web Server
 *  Serves the website and model files
 * ============================================
 */

const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static website files
app.use(express.static(__dirname));

// Serve face-api.js library
const libDir = fs.existsSync(path.join(__dirname, 'lib'))
  ? path.join(__dirname, 'lib')
  : path.join(__dirname, 'node_modules', '@vladmandic', 'face-api', 'dist');
app.use('/lib', express.static(libDir));

// Serve face-api.js model files
const modelsDir = fs.existsSync(path.join(__dirname, 'models'))
  ? path.join(__dirname, 'models')
  : path.join(__dirname, 'node_modules', '@vladmandic', 'face-api', 'model');
app.use('/models', express.static(modelsDir));

// API: Get labeled descriptors
app.get('/api/descriptors', (req, res) => {
  const descriptorsPath = path.join(__dirname, 'labeled_descriptors.json');
  
  if (fs.existsSync(descriptorsPath)) {
    const data = JSON.parse(fs.readFileSync(descriptorsPath, 'utf8'));
    res.json(data);
  } else {
    res.status(404).json({
      error: 'No training data found',
      message: 'Run "npm run train" first to generate face descriptors.'
    });
  }
});

// API: Get dataset info
app.get('/api/dataset-info', (req, res) => {
  const datasetDir = path.join(__dirname, 'dataset');
  
  if (!fs.existsSync(datasetDir)) {
    return res.json({ persons: [], totalImages: 0 });
  }
  
  const persons = fs.readdirSync(datasetDir)
    .filter(d => fs.statSync(path.join(datasetDir, d)).isDirectory())
    .map(name => {
      const personDir = path.join(datasetDir, name);
      const images = fs.readdirSync(personDir).filter(f => 
        /\.(jpg|jpeg|png|webp|bmp|gif)$/i.test(f)
      );
      return { name, imageCount: images.length };
    });
  
  const totalImages = persons.reduce((sum, p) => sum + p.imageCount, 0);
  
  res.json({ persons, totalImages });
});

// Start server
app.listen(PORT, () => {
  console.log('');
  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║     🌐 Face Recognition Web Server                ║');
  console.log('╚════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`🚀 Server running at: http://localhost:${PORT}`);
  console.log('');
  
  // Check if training data exists
  const descriptorsPath = path.join(__dirname, 'labeled_descriptors.json');
  if (fs.existsSync(descriptorsPath)) {
    const data = JSON.parse(fs.readFileSync(descriptorsPath, 'utf8'));
    const count = data.descriptors ? data.descriptors.length : 0;
    console.log(`✅ Training data loaded: ${count} person(s)`);
  } else {
    console.log('⚠️  No training data found!');
    console.log('   Run "npm run train" to generate face descriptors first.');
  }
  
  console.log('');
  console.log('📋 Available routes:');
  console.log(`   • Website:     http://localhost:${PORT}`);
  console.log(`   • Models:      http://localhost:${PORT}/models/`);
  console.log(`   • Descriptors: http://localhost:${PORT}/api/descriptors`);
  console.log(`   • Dataset:     http://localhost:${PORT}/api/dataset-info`);
  console.log('');
  console.log('Press Ctrl+C to stop the server.');
  console.log('');
});
