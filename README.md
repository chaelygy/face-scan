# 🧠 FaceID Pro - Sistem Pengenalan Wajah AI (Teachable Machine Style)

Sistem pengenalan wajah cerdas dengan akurasi tinggi (>95%) berbasis deep learning (SSD MobileNet v1 + Face Landmarks 68 + ResNet-34 128-D Embeddings). 

Dibuat khusus agar Anda cukup **memasukkan foto teman ke folder nama masing-masing**, lalu melatih (train) model secara otomatis hanya dengan satu klik seperti di Google Teachable Machine!

---

## 🚀 Cara Menjalankan Website (Super Mudah)

### Opsi 1: Klik Dua Kali File `start.bat`
Cukup klik dua kali file **[start.bat](file:///d:/New%20folder/start.bat)** di folder ini.
Server akan otomatis berjalan dan membuka browser ke:
👉 **`http://localhost:5000`**

---

## 📁 Cara Menambahkan Foto Teman Sekelas (Dataset)

1. Buka folder **[dataset/](file:///d:/New%20folder/dataset/)**
2. Buat folder baru dengan nama teman Anda, contoh:
   ```text
   dataset/
   ├── Andi/
   │   ├── andi_1.jpg
   │   ├── andi_2.jpg
   │   └── andi_3.jpg
   ├── Siti/
   │   ├── siti_1.jpg
   │   └── siti_2.jpg
   ├── Budi/
   │   ├── budi_1.jpg
   │   └── budi_2.jpg
   └── [Nama_Teman_Lain]/
       ├── foto_1.jpg
       └── foto_2.jpg
   ```
3. Masukkan **2 - 10 foto** untuk setiap orang.

---

## 🎯 Cara Melatih (Training) Model:

1. Buka website di **`http://localhost:5000`**
2. Klik tab **"🧠 Latih Model (Training)"**
3. Anda akan melihat daftar semua folder teman yang baru saja Anda masukkan.
4. Klik tombol **"🚀 Mulai Latih Dataset"**
5. Sistem akan membaca setiap foto, mendeteksi wajah, mengambil 128-D face embedding, dan menyimpannya secara otomatis ke `labeled_descriptors.json`.
6. Selesai! Model langsung aktif saat itu juga tanpa perlu restart server!

---

## 🔍 Cara Menggunakan Fitur Pengenalan Wajah:

1. **Tab "Kenali Wajah" (Upload Foto):**
   - Drag & drop atau pilih foto teman yang ingin dikenali.
   - Kotak bounding box hijau akan muncul di wajah dengan nama teman dan persentase tingkat kemiripan (Confidence %).
   - Anda juga dapat mengatur slider **Sensitivitas Pengenalan** sesuai kebutuhan.

2. **Tab "Webcam Live":**
   - Klik tombol **"Mulai Webcam"**
   - Kamera akan aktif dan mengenali wajah secara real-time langsung di layar!

---

## 🏆 Kunci Akurasi di Atas 95%:
- **Pencahayaan Jelas:** Pastikan foto wajah tidak terlalu gelap atau blur.
- **Variasi Sudut:** 1 foto lurus ke depan, 1 sedikit senyum, 1 sedikit menyamping.
- **Resolusi Bagus:** Foto minimal 200x200 pixel.
- **Wajah Terlihat:** Jangan tertutup masker atau kacamata hitam pekat saat training.
