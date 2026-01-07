# Models Directory

Folder ini berisi model-model yang sudah ditraining dan siap digunakan oleh aplikasi.

## Struktur Folder

```
models/
├── Decision Tree/          # Model Decision Tree
│   ├── model.pkl          # Model utama
│   ├── scaler.pkl         # Feature scaler
│   ├── feature_cols.pkl   # Daftar feature columns
│   ├── metadata.pkl       # Metadata model (best params, CV scores)
│   ├── metrics.pkl        # Metrics evaluasi
│   └── feature_importance.pkl  # Feature importance ranking
│
├── CNN/                   # Model CNN Multi-head
│   ├── fcn_multihead_final.keras  # Model utama (Keras format)
│   ├── scaler_X_cnn_eval.pkl      # Input scaler
│   ├── scaler_yreg_cnn_eval.pkl   # Target regression scaler
│   ├── y_map_max_cnn_eval.pkl     # Max values untuk heatmap normalization
│   ├── evaluation_metrics_cnn_fixed.pkl  # Metrics evaluasi
│   └── evaluation_metrics_cnn_fixed.json   # Metrics evaluasi (JSON)
│
└── GRU/                   # Model GRU Multi-output
    ├── multi_gru_model.keras              # Model utama (Keras format)
    ├── scaler_X_multi_conservative.pkl    # Feature scaler
    ├── scalers_y_multi_conservative.pkl   # Target scalers (per target)
    ├── feature_cols_multi_conservative.pkl # Feature columns
    └── target_cols_multi_conservative.pkl # Target columns
```

## Catatan

- Model disimpan di folder `models/` setelah training selesai
- Training artifacts lengkap tetap disimpan di folder `Training/` untuk referensi
- Folder `models/` berisi file-file minimal yang diperlukan aplikasi untuk melakukan prediksi

## Penggunaan

Aplikasi (`app.py`) akan memuat model dari folder ini untuk melakukan prediksi.
