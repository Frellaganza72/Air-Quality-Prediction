import { useEffect, useState } from "react";
import { getDashboard } from "../api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

const CHART_COLORS = ["#3b82f6", "#f59e0b", "#16a34a"];

type HeatmapCell = {
  value: number;
  district?: string;
  row?: number;
  col?: number;
  pollutant?: string;
};
type TrendPoint = { date: string; pm25: number };
type ISPUInfo = {
  overall: string;
  pm25: string;
  o3: string;
  co: string;
  color: string;
};

type Predictions = {
  decision_tree: { pm25: number; o3: number; co: number; ispu: string };
  lstm: { pm25: number; o3: number; co: number; ispu: string };
  cnn: { pm25: number; o3: number; co: number; ispu: string };
};

type TrendData = {
  data: TrendPoint[];
  period: string;
};

type HeatmapData = {
  data: HeatmapCell[];
  grid_size: string;
  pollutants: string[];
};

type AnomaliesData = {
  data: Array<{
    pollutant: string;
    description: string;
    datetime: string;
    value?: number;
    increase_percent?: number;
  }>;
  count: number;
  period: string;
};

type StatisticsData = {
  ispu_categories: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  total_days: number;
  total_hours: number;
  date_range: {
    start: string;
    end: string;
  } | null;
};

type DashboardResponse = {
  status: string;
  timestamp: string;
  primary_model: string;
  predictions: Predictions;
  trend: TrendData;
  heatmap: HeatmapData;
  anomalies: AnomaliesData;
  statistics: StatisticsData;
};

export default function Dashboard() {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>(() => {
    // Default to today's date in YYYY-MM-DD format
    const today = new Date();
    return today.toISOString().split('T')[0];
  });
  const [anoms, setAnoms] = useState<
    Array<{
      pollutant: string;
      description: string;
      datetime: string;
      value?: number;
      increase_percent?: number;
    }>
  >([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        // Pass selectedDate to getDashboard
        const res = await getDashboard(selectedDate);
        setData(res);
        // Use anomalies from dashboard response (7 latest from 30 days)
        setAnoms(res.anomalies?.data || []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [selectedDate]);  // Re-fetch when date changes

  if (loading)
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#666" }}>
        Loading...
      </div>
    );
  if (!data)
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#666" }}>
        No data available
      </div>
    );

  // Access new API structure: predictions.decision_tree, predictions.lstm, predictions.cnn
  const dt = (data as any).predictions?.decision_tree || { pm25: 0, o3: 0, co: 0, ispu: 'SEDANG' };
  const lstm = (data as any).predictions?.lstm || { pm25: 0, o3: 0, co: 0, ispu: 'SEDANG' };
  const cnn = (data as any).predictions?.cnn || { pm25: 0, o3: 0, co: 0, ispu: 'SEDANG' };
  
  // Use primary_model from API, fallback to cnn if not specified
  const primaryModelName = (data as any).primary_model || 'cnn';
  const primaryPredictions = primaryModelName === 'cnn' ? cnn : primaryModelName === 'lstm' ? lstm : dt;
  
  // Get ISPU category from predictions (new structure)
  const ispuPrimaryCategory = primaryPredictions.ispu || 'SEDANG';

  // CO is already in µg/m³ from API (no conversion needed)
  // Values range from ~150 to ~1600 µg/m³ based on training data
  const dtCoUg = dt.co;
  const lstmCoUg = lstm.co;
  const cnnCoUg = cnn.co;

  // Stats for pie chart from API (30 days back from today)
    const statsData: { name: string; value: number; color: string }[] = ((data as any).statistics?.ispu_categories || (data as any).ispu_statistics?.categories) || [
      { name: "Baik", value: 0, color: "#22c55e" },
      { name: "Sedang", value: 0, color: "#eab308" },
      { name: "Tidak Sehat", value: 0, color: "#f97316" },
      { name: "Sangat Tidak Sehat", value: 0, color: "#ef4444" },
    ];
  
  const ispuStats = ((data as any).statistics || (data as any).ispu_statistics) || {
    total_days: 0,
    total_hours: 0,
    date_range: null
  };

  // Mapping ISPU category to color
  const ispuColorMap: Record<string, string> = {
    "Baik": "#22c55e",
    "Sedang": "#eab308",
    "Tidak Sehat": "#f97316",
    "Sangat Tidak Sehat": "#ef4444",
    "Berbahaya": "#7f1d1d",
  };

  // Heatmap: only 50 boxes, layout 10 columns x 5 rows
  const heatmapCells = ((data as any).heatmap?.data || (data as any).heatmap || []).slice(0, 50);
  const heatmapCols = 10;
  
  // Trend data - handle both old and new API structures
  const trendData = ((data as any).trend?.data || (data as any).trend || []) as TrendPoint[];

  // Determine worst pollutant according to ISPU labels from decision tree model's ispu fields
  const severityOrder: Record<string, number> = {
    "Baik": 1,
    "Sedang": 2,
    "Tidak Sehat": 3,
    "Sangat Tidak Sehat": 4,
    "Berbahaya": 5,
  };
  function worstPollutantFromISPU(ispu: ISPUInfo) {
    const items: Array<{ pollutant: string; label: string }> = [
      { pollutant: "PM2.5", label: ispu.pm25 },
      { pollutant: "O₃", label: ispu.o3 },
      { pollutant: "CO", label: ispu.co },
    ];
    items.sort((a, b) => (severityOrder[b.label] || 0) - (severityOrder[a.label] || 0));
    return items[0];
  }
  
  // Create ISPU info object for primary model
  const ispuPrimary: ISPUInfo = {
    overall: ispuPrimaryCategory,
    pm25: dt.ispu,  // Use DT for individual pollutant status
    o3: dt.ispu,
    co: dt.ispu,
    color: ispuColorMap[ispuPrimaryCategory] || "#ef4444"
  };
  
  const worst = worstPollutantFromISPU(ispuPrimary);

  // Prepare separate chart data for each pollutant (to avoid scale issues)
  const pm25Data = [
    {
      name: "Decision Tree",
      value: Number(dt.pm25.toFixed(1)),
    },
    {
      name: "GRU",
      value: Number(lstm.pm25.toFixed(1)),
    },
    {
      name: "CNN",
      value: Number(cnn.pm25.toFixed(1)),
    },
  ];

  const o3Data = [
    {
      name: "Decision Tree",
      value: Number(dt.o3.toFixed(1)),
    },
    {
      name: "GRU",
      value: Number(lstm.o3.toFixed(1)),
    },
    {
      name: "CNN",
      value: Number(cnn.o3.toFixed(1)),
    },
  ];

  const coData = [
    {
      name: "Decision Tree",
      value: Number(dtCoUg.toFixed(1)),
    },
    {
      name: "GRU",
      value: Number(lstmCoUg.toFixed(1)),
    },
    {
      name: "CNN",
      value: Number(cnnCoUg.toFixed(1)),
    },
  ];

  // Format anomaly date without time (user requested no hour)
  const formatDateOnly = (s: string) => {
    try {
      const d = new Date(s);
      return `${d.getDate()}/${d.getMonth() + 1}/${d.getFullYear()}`;
    } catch {
      // fallback: strip time part if common format
      return s.split("T")[0] || s;
    }
  };

  return (
    <div
      style={{ background: "#f0f4f8", minHeight: "100vh", paddingBottom: 40 }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px" }}>
        {/* ISPU Reference Table */}
        <div className="card" style={{ marginBottom: 24, background: "#f8fafc" }}>
          <h3 style={{ fontSize: 16, marginBottom: 16, color: "#1e293b", fontWeight: 600 }}>
            📋 Panduan Klasifikasi ISPU (Indeks Standar Pencemar Udara)
          </h3>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 13,
                background: "#ffffff",
                borderRadius: "8px",
                overflow: "hidden",
              }}
            >
              <thead>
                <tr style={{ background: "#e2e8f0", borderBottom: "2px solid #cbd5e1" }}>
                  <th style={{ padding: "12px 14px", textAlign: "left", fontWeight: 600, color: "#1e293b" }}>
                    Kategori
                  </th>
                  <th style={{ padding: "12px 14px", textAlign: "center", fontWeight: 600, color: "#1e293b" }}>
                    Warna
                  </th>
                  <th style={{ padding: "12px 14px", textAlign: "center", fontWeight: 600, color: "#1e293b" }}>
                    PM2.5 (µg/m³)
                  </th>
                  <th style={{ padding: "12px 14px", textAlign: "center", fontWeight: 600, color: "#1e293b" }}>
                    O₃ (µg/m³)
                  </th>
                  <th style={{ padding: "12px 14px", textAlign: "center", fontWeight: 600, color: "#1e293b" }}>
                    CO (µg/m³)
                  </th>
                  <th style={{ padding: "12px 14px", textAlign: "left", fontWeight: 600, color: "#1e293b" }}>
                    Keterangan
                  </th>
                </tr>
              </thead>
              <tbody>
                {/* Baik */}
                <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#ffffff" }}>
                  <td style={{ padding: "12px 14px", fontWeight: 600, color: "#1e293b" }}>Baik</td>
                  <td style={{ padding: "12px 14px", textAlign: "center" }}>
                    <div
                      style={{
                        display: "inline-block",
                        width: "24px",
                        height: "24px",
                        background: "#22c55e",
                        borderRadius: "4px",
                        border: "1px solid #16a34a",
                      }}
                    />
                  </td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>0 - 15.5</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>0 - 120</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>0 - 4,000</td>
                  <td style={{ padding: "12px 14px", color: "#475569" }}>
                    ✅ Udara bersih, aman untuk aktivitas outdoor
                  </td>
                </tr>

                {/* Sedang */}
                <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#ffffff" }}>
                  <td style={{ padding: "12px 14px", fontWeight: 600, color: "#1e293b" }}>Sedang</td>
                  <td style={{ padding: "12px 14px", textAlign: "center" }}>
                    <div
                      style={{
                        display: "inline-block",
                        width: "24px",
                        height: "24px",
                        background: "#eab308",
                        borderRadius: "4px",
                        border: "1px solid #ca8a04",
                      }}
                    />
                  </td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>15.6 - 55.4</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>121 - 235</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>4,001 - 8,000</td>
                  <td style={{ padding: "12px 14px", color: "#475569" }}>
                    ⚠️ Orang sensitif perlu membatasi aktivitas outdoor
                  </td>
                </tr>

                {/* Tidak Sehat */}
                <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#ffffff" }}>
                  <td style={{ padding: "12px 14px", fontWeight: 600, color: "#1e293b" }}>Tidak Sehat</td>
                  <td style={{ padding: "12px 14px", textAlign: "center" }}>
                    <div
                      style={{
                        display: "inline-block",
                        width: "24px",
                        height: "24px",
                        background: "#f97316",
                        borderRadius: "4px",
                        border: "1px solid #c2410c",
                      }}
                    />
                  </td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>55.5 - 150.4</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>236 - 400</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>8,001 - 15,000</td>
                  <td style={{ padding: "12px 14px", color: "#475569" }}>
                    🚫 Disarankan tetap di dalam ruangan yang berAC/filter
                  </td>
                </tr>

                {/* Sangat Tidak Sehat */}
                <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#ffffff" }}>
                  <td style={{ padding: "12px 14px", fontWeight: 600, color: "#1e293b" }}>Sangat Tidak Sehat</td>
                  <td style={{ padding: "12px 14px", textAlign: "center" }}>
                    <div
                      style={{
                        display: "inline-block",
                        width: "24px",
                        height: "24px",
                        background: "#ef4444",
                        borderRadius: "4px",
                        border: "1px solid #b91c1c",
                      }}
                    />
                  </td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>150.5 - 250.4</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>401 - 800</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>15,001 - 30,000</td>
                  <td style={{ padding: "12px 14px", color: "#475569" }}>
                    🔴 Bahaya serius ke kesehatan, hindari aktivitas outdoor
                  </td>
                </tr>

                {/* Berbahaya */}
                <tr style={{ background: "#ffffff" }}>
                  <td style={{ padding: "12px 14px", fontWeight: 600, color: "#1e293b" }}>Berbahaya</td>
                  <td style={{ padding: "12px 14px", textAlign: "center" }}>
                    <div
                      style={{
                        display: "inline-block",
                        width: "24px",
                        height: "24px",
                        background: "#7f1d1d",
                        borderRadius: "4px",
                        border: "1px solid #430a0a",
                      }}
                    />
                  </td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>&gt; 250.5</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>&gt; 801</td>
                  <td style={{ padding: "12px 14px", textAlign: "center", color: "#475569" }}>&gt; 30,000</td>
                  <td style={{ padding: "12px 14px", color: "#475569" }}>
                    🚨 KRITIS! Harus isolasi diri di dalam ruangan
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: "#64748b", fontStyle: "italic" }}>
            💡 Catatan: Kategori ISPU ditentukan oleh polutan terburuk di antara PM2.5, O₃, dan CO.
          </div>
        </div>

        {/* Top header with prediction days and date picker */}
        <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", marginBottom: 20, gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <label style={{ fontSize: 14, fontWeight: 600, color: "#475569" }}>
              Pilih Tanggal:
            </label>
            <input 
              type="date" 
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={{
                padding: "8px 12px",
                border: "2px solid #cbd5e1",
                borderRadius: "6px",
                fontSize: "14px",
                fontWeight: 500,
                color: "#1e293b",
                cursor: "pointer"
              }}
            />
            <span style={{ color: "#475569", fontWeight: 600, minWidth: "200px" }}>
              Prediksi {new Date(selectedDate + 'T00:00:00').toLocaleDateString('id-ID', { day: '2-digit', month: 'long', year: 'numeric' })}
            </span>
          </div>
        </div>

        {/* Warning Banner */}
        {ispuPrimaryCategory !== "Baik" && ispuPrimaryCategory !== "Sedang" && (
          <div
            style={{
              background: "linear-gradient(135deg, #f97316 0%, #ea580c 100%)",
              color: "white",
              padding: "12px 16px",
              borderRadius: 12,
              marginBottom: 20,
              display: "flex",
              alignItems: "center",
              gap: 12,
              boxShadow: "0 4px 12px rgba(249, 115, 22, 0.3)",
            }}
          >
            <span style={{ fontSize: 20 }}>⚠️</span>
            <div>
              <strong>Peringatan!</strong> {worst.pollutant} terburuk menurut ISPU: <strong>{worst.label}</strong>
            </div>
          </div>
        )}

        {/* Main Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(350px, 1fr))",
            gap: 20,
            marginBottom: 24,
          }}
        >
          {/* Status Kualitas Udara */}
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <h3 style={{ color: "#3b82f6", fontSize: 16, fontWeight: 600, margin: 0 }}>
                Status Kualitas Udara
              </h3>
              <span style={{ fontSize: 12, color: "#666", fontStyle: "italic" }}>
                ({primaryModelName === 'cnn' ? 'CNN ⭐ Best' : primaryModelName === 'lstm' ? 'GRU' : 'Decision Tree'})
              </span>
            </div>

            {/* PM2.5 */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ color: "#666" }}>PM2.5</span>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <strong style={{ fontSize: 20, color: ispuColorMap[ispuPrimaryCategory] || "#ef4444" }}>{primaryPredictions.pm25.toFixed(1)}</strong>
                  <span style={{ fontSize: 14, color: "#666" }}>µg/m³</span>
                </div>
              </div>
            </div>

            {/* O3 */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ color: "#666" }}>O₃</span>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <strong style={{ fontSize: 20, color: ispuColorMap[ispuPrimaryCategory] || "#ef4444" }}>{primaryPredictions.o3.toFixed(1)}</strong>
                  <span style={{ fontSize: 14, color: "#666" }}>µg/m³</span>
                </div>
              </div>
            </div>

            {/* CO (already in µg/m³) */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ color: "#666" }}>CO</span>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <strong style={{ fontSize: 20, color: ispuColorMap[ispuPrimaryCategory] || "#ef4444" }}>{primaryPredictions.co.toFixed(1)}</strong>
                  <span style={{ fontSize: 14, color: "#666" }}>µg/m³</span>
                </div>
              </div>
            </div>
          </div>

          {/* Grafik Trend */}
          <div className="card">
            <h3 style={{ color: "#3b82f6", fontSize: 16, marginBottom: 12, fontWeight: 600 }}>
              Grafik Trend PM2.5
            </h3>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={trendData}>
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(val) => {
                    const d = new Date(val);
                    return `${d.getDate()}/${d.getMonth() + 1}`;
                  }}
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="pm25" fill="#60a5fa" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Heatmap (50 boxes) */}
          <div className="card">
            <h3 style={{ color: "#3b82f6", fontSize: 16, marginBottom: 12, fontWeight: 600 }}>
              Heatmap Polusi Kota Malang
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${heatmapCols}, 1fr)`, gap: 6 }}>
              {heatmapCells.map((cell: HeatmapCell, i: number) => {
                const val = cell?.value || 0;
                const pollutant = cell?.pollutant || "PM2.5";
                
                // Different thresholds based on pollutant type
                let breakpoints: { [key: string]: [number, number, number] } = {
                  "PM2.5": [15.5, 55.5, 150.5],
                  "O3": [60, 120, 235],
                  "CO": [2000, 4000, 10000]
                };
                
                const [threshold1, threshold2, threshold3] = breakpoints[pollutant] || breakpoints["PM2.5"];
                
                // Color based on pollutant-specific thresholds
                const color = val > threshold3 ? "#ef4444"     // Red - Sangat Tidak Sehat
                           : val > threshold2 ? "#f97316"     // Orange - Tidak Sehat
                           : val > threshold1 ? "#eab308"     // Yellow - Sedang
                           :                    "#22c55e";    // Green - Baik
                
                return (
                  <div
                    key={i}
                    title={`${cell.district || ""}\n${pollutant}: ${val.toFixed(1)} µg/m³`}
                    style={{ 
                      aspectRatio: "1", 
                      background: color, 
                      borderRadius: 4,
                      cursor: "pointer",
                      transition: "transform 0.2s ease"
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.transform = "scale(1.05)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.transform = "scale(1)";
                    }}
                  />
                );
              })}
            </div>
          </div>
        </div>

        {/* Perbandingan Prediksi: Tabel + Grafik */}
        <div style={{ marginBottom: 24 }}>
          <div className="card" style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: 18, marginBottom: 16, color: "#2d3748", fontWeight: 600 }}>
              Perbandingan Prediksi 3 Algoritma
            </h3>

            {/* Tabel Perbandingan */}
            <div style={{ overflowX: "auto", marginBottom: 20 }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 14,
                }}
              >
                <thead>
                  <tr style={{ background: "#f8fafc", borderBottom: "2px solid #e2e8f0" }}>
                    <th
                      style={{
                        padding: "12px 16px",
                        textAlign: "left",
                        fontWeight: 600,
                        color: "#1e293b",
                        fontSize: 14,
                      }}
                    >
                      Algoritma
                    </th>
                    <th
                      style={{
                        padding: "12px 16px",
                        textAlign: "center",
                        fontWeight: 600,
                        color: "#1e293b",
                        fontSize: 14,
                      }}
                    >
                      PM2.5 (µg/m³)
                    </th>
                    <th
                      style={{
                        padding: "12px 16px",
                        textAlign: "center",
                        fontWeight: 600,
                        color: "#1e293b",
                        fontSize: 14,
                      }}
                    >
                      O₃ (µg/m³)
                    </th>
                    <th
                      style={{
                        padding: "12px 16px",
                        textAlign: "center",
                        fontWeight: 600,
                        color: "#1e293b",
                        fontSize: 14,
                      }}
                    >
                      CO (µg/m³)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px 16px", fontWeight: 600, color: "#1e293b" }}>
                      Decision Tree
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {dt.pm25.toFixed(1)}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {dt.o3.toFixed(1)}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {dtCoUg.toFixed(1)}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0", background: primaryModelName === 'lstm' ? "#f0f9ff" : "transparent" }}>
                    <td style={{ padding: "12px 16px", fontWeight: 600, color: "#1e293b" }}>
                      GRU {primaryModelName === 'lstm' && <span style={{ color: "#22c55e", fontWeight: "bold" }}></span>}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {lstm.pm25.toFixed(1)}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {lstm.o3.toFixed(1)}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {lstmCoUg.toFixed(1)}
                    </td>
                  </tr>
                  <tr style={{ background: "#fef3c7", opacity: 0.9 }}>
                    <td style={{ padding: "12px 16px", fontWeight: 600, color: "#1e293b" }}>
                      CNN (Spatial) {primaryModelName === 'cnn' && <span style={{ color: "#22c55e", fontWeight: "bold" }}></span>}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {cnn.pm25.toFixed(1)}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {cnn.o3.toFixed(1)}
                    </td>
                    <td style={{ padding: "12px 16px", textAlign: "center", color: "#1e293b", fontWeight: 500 }}>
                      {cnnCoUg.toFixed(1)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Grafik Perbandingan - Dipecah per polutan */}
            <div style={{ marginTop: 20 }}>
              <h4 style={{ fontSize: 16, marginBottom: 16, color: "#475569", fontWeight: 600 }}>
                Grafik Perbandingan Prediksi
              </h4>
              
              {/* Grid untuk 3 grafik */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
                {/* Grafik PM2.5 */}
                <div style={{ background: "#ffffff", padding: "12px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#1e293b", marginBottom: 8, textAlign: "center" }}>
                    PM2.5 (µg/m³)
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={pm25Data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                      <YAxis tick={{ fontSize: 10 }} width={50} />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: "#ffffff",
                          border: "1px solid #e2e8f0",
                          borderRadius: "6px",
                          fontSize: "12px"
                        }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {pm25Data.map((_, index) => (
                          <Cell key={`pm25-cell-${index}`} fill={CHART_COLORS[index]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Grafik O3 */}
                <div style={{ background: "#ffffff", padding: "12px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#1e293b", marginBottom: 8, textAlign: "center" }}>
                    O₃ (µg/m³)
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={o3Data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                      <YAxis tick={{ fontSize: 10 }} width={50} />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: "#ffffff",
                          border: "1px solid #e2e8f0",
                          borderRadius: "6px",
                          fontSize: "12px"
                        }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {o3Data.map((_, index) => (
                          <Cell key={`o3-cell-${index}`} fill={CHART_COLORS[index]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Grafik CO */}
                <div style={{ background: "#ffffff", padding: "12px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#1e293b", marginBottom: 8, textAlign: "center" }}>
                    CO (µg/m³)
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={coData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                      <YAxis tick={{ fontSize: 10 }} width={50} />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: "#ffffff",
                          border: "1px solid #e2e8f0",
                          borderRadius: "6px",
                          fontSize: "12px"
                        }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {coData.map((_, index) => (
                          <Cell key={`co-cell-${index}`} fill={CHART_COLORS[index]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Statistik & Anomali Row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          {/* Statistik Kategori ISPU */}
          <div className="card">
            <h3 style={{ fontSize: 18, marginBottom: 12, color: "#2d3748", fontWeight: 600 }}>
              📊 Statistik Kategori ISPU (30 hari)
            </h3>
            {statsData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={statsData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {statsData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ marginTop: 12, fontSize: 14, color: "#666" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span>Total data:</span>
                    <strong>{ispuStats.total_hours} jam</strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span>Total hari:</span>
                    <strong>{ispuStats.total_days} hari</strong>
                  </div>
                  {ispuStats.date_range && (
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#94a3b8" }}>
                      <span>Rentang:</span>
                      <span>{ispuStats.date_range.start} - {ispuStats.date_range.end}</span>
                    </div>
                  )}
                  {statsData.find(cat => cat.name === "Baik") && (
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                      <span>Hari dengan kualitas Baik:</span>
                      <strong>
                        {statsData.find(cat => cat.name === "Baik")?.value || 0} hari (
                        {ispuStats.total_days > 0
                          ? ((statsData.find(cat => cat.name === "Baik")?.value || 0) / ispuStats.total_days * 100).toFixed(0)
                          : 0}%)
                      </strong>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div style={{ padding: 40, textAlign: "center", color: "#999" }}>
                Tidak ada data statistik tersedia
              </div>
            )}
          </div>

          {/* Deteksi Anomali dari Data Outliers */}
          <div className="card">
            <h3 style={{ fontSize: 18, marginBottom: 12, color: "#2d3748", fontWeight: 600 }}>
              ⚠️ Deteksi Anomali
            </h3>
            {anoms.length === 0 ? (
              <div style={{ textAlign: "center", color: "#999", padding: 40 }}>Tidak ada anomali terdeteksi</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {/* Show top 2 anomalies */}
                {anoms.slice(0, 2).map((a, i) => {
                  const pollutantColor = 
                    a.pollutant === "PM2.5" ? "#f97316" :
                    a.pollutant === "O3" ? "#8b5cf6" :
                    a.pollutant === "CO" ? "#06b6d4" : "#f97316";
                  
                  return (
                    <div key={i} style={{ 
                      background: "#fff7ed", 
                      padding: 14, 
                      borderRadius: 8, 
                      borderLeft: `4px solid ${pollutantColor}`,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center"
                    }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                          <span style={{ 
                            display: "inline-block",
                            background: pollutantColor,
                            color: "white",
                            padding: "4px 10px",
                            borderRadius: 12,
                            fontSize: 12,
                            fontWeight: 600
                          }}>
                            {a.pollutant}
                          </span>
                          <strong style={{ color: "#1e293b", fontSize: 14 }}>
                            {a.value || "N/A"} μg/m³
                          </strong>
                          {a.increase_percent !== undefined && (
                            <span style={{ color: "#dc2626", fontSize: 12, fontWeight: 600 }}>
                              ↑{a.increase_percent}%
                            </span>
                          )}
                        </div>
                        <div style={{ fontSize: 13, color: "#78350f", marginLeft: 0 }}>
                          {a.description}
                        </div>
                      </div>
                      <div style={{ 
                        fontSize: 12, 
                        color: "#9a3412", 
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        marginLeft: 12
                      }}>
                        {formatDateOnly(a.datetime)}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}