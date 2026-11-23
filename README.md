# Implementasi Object Detection YOLO11n pada Raspberry Pi 5 dengan Indikator LED GPIO

Repositori ini berisi kode program dan dokumentasi praktikum **Linux Embedded** untuk mendeteksi objek secara *real-time* menggunakan model **YOLO11n** (*pretrained* dataset **MS COCO** 80 kelas) pada **Raspberry Pi 5**, terintegrasi dengan pengendalian indikator **LED berbasis GPIO**.

---

## 📋 Deskripsi Sistem

Sistem mengambil aliran citra (*video stream*) dari kamera USB atau IP Webcam smartphone, memproses setiap *frame* menggunakan model `yolo11n.pt`, lalu menyaring objek target (`bottle`) berdasarkan nilai *confidence* tertentu untuk mengendalikan dua indikator LED:

* **Objek target terdeteksi (`bottle` dengan `confidence >= 0.50`):**
  * 🟢 **LED Hijau (GPIO 17):** `ON`
  * 🔴 **LED Merah (GPIO 27):** `OFF`
* **Objek target tidak terdeteksi atau `confidence < 0.50`:**
  * 🟢 **LED Hijau (GPIO 17):** `OFF`
  * 🔴 **LED Merah (GPIO 27):** `ON`
* **Program dihentikan (`q` atau `Ctrl+C`):**
  * 🟢 **LED Hijau & 🔴 LED Merah:** `OFF`

---

## 🛠️ Kebutuhan Perangkat Keras & Rangkaian GPIO

| Komponen | Pin Raspberry Pi 5 (BCM) | Pin Fisik | Keterangan |
| :--- | :---: | :---: | :--- |
| **LED Hijau** | `GPIO 17` | Pin 11 | Anoda (`+`) seri dengan resistor 220–330 Ω |
| **LED Merah** | `GPIO 27` | Pin 13 | Anoda (`+`) seri dengan resistor 220–330 Ω |
| **Ground (`GND`)** | `GND` | Pin 6 | Terhubung ke katoda (`-`) kedua LED |
| **Kamera** | Port USB / Wi-Fi | - | Kamera USB (`/dev/video0`) atau IP Webcam |

---

## 📁 Struktur Direktori

```text
praktikum_yolo/
├── yolo11n.pt          # Bobot model YOLO11 Nano (MS COCO 80 kelas)
├── test_camera.py      # Skrip pengujian awal kamera dan deteksi YOLO
├── yolo_led.py         # Skrip utama deteksi multi-thread + kontrol LED GPIO
├── requirements.txt    # Daftar dependensi pustaka Python
├── .gitignore          # Pengecualian folder venv_yoloCoco & cache
└── README.md           # Dokumentasi proyek
