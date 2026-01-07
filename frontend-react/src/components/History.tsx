import { useEffect, useState } from "react";
import { getHistory } from "../api";

type Prediction = { pm25: number; o3: number; co: number };
type ISPUInfo = { overall: string; color: string };
type HistoryRecord = {
  date?: string;
  tanggal?: string;
  data_type?: string;
  measurements?: { pm25: number; o3: number; co: number };
  ispu?: { overall: string; color: string };
  decision_tree?: { predictions?: Prediction; ispu?: ISPUInfo } | Prediction;
};

export default function History() {
  const [data, setData] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await getHistory(20);
        const arr = (res && (res.records || res)) || [];
        setData(arr as HistoryRecord[]);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading)
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#666" }}>
        Loading...
      </div>
    );
  if (!data || data.length === 0)
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#666" }}>
        No history available
      </div>
    );

  return (
    <div
      style={{ background: "#f0f4f8", minHeight: "100vh", paddingBottom: 40 }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px" }}>
        <div
          style={{
            background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            color: "white",
            padding: "20px 24px",
            borderRadius: 12,
            marginBottom: 24,
            boxShadow: "0 4px 12px rgba(16, 185, 129, 0.3)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 16
          }}
        >
          <div>
            <h2 style={{ margin: 0, fontSize: 24, fontWeight: 600 }}>
              Riwayat Kualitas Udara
            </h2>
            <div style={{ marginTop: 8, fontSize: 15, opacity: 0.95 }}>
              Data historis kualitas udara aktual
            </div>
          </div>
          
          <button
            onClick={() => window.open("http://localhost:2000/api/history/export?limit=20", "_blank")}
            style={{
              background: "white",
              color: "#059669",
              border: "none",
              padding: "10px 20px",
              borderRadius: 8,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 8,
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
              transition: "transform 0.1s"
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = "scale(1.05)"}
            onMouseOut={(e) => e.currentTarget.style.transform = "scale(1)"}
          >
            <span>📥</span> Export Excel
          </button>
        </div>

        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 14,
              }}
            >
              <thead>
                <tr
                  style={{
                    background: "#f9fafb",
                    borderBottom: "2px solid #e5e7eb",
                  }}
                >
                  <th
                    style={{
                      padding: "14px 16px",
                      textAlign: "left",
                      fontWeight: 600,
                      color: "#374151",
                    }}
                  >
                    Tanggal
                  </th>
                  <th
                    style={{
                      padding: "14px 16px",
                      textAlign: "center",
                      fontWeight: 600,
                      color: "#374151",
                    }}
                  >
                    PM2.5 (μg/m³)
                  </th>
                  <th
                    style={{
                      padding: "14px 16px",
                      textAlign: "center",
                      fontWeight: 600,
                      color: "#374151",
                    }}
                  >
                    O₃ (μg/m³)
                  </th>
                  <th
                    style={{
                      padding: "14px 16px",
                      textAlign: "center",
                      fontWeight: 600,
                      color: "#374151",
                    }}
                  >
                    CO (mg/m³)
                  </th>
                  <th
                    style={{
                      padding: "14px 16px",
                      textAlign: "center",
                      fontWeight: 600,
                      color: "#374151",
                    }}
                  >
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.slice(0, 20).map((row: HistoryRecord, i: number) => {
                  // Handle new API structure (data_type: 'actual', measurements: {...})
                  const measurements = row.measurements || {};
                  const ispu_obj = row.ispu || { overall: "Baik", color: "#22c55e" };
                  
                  // Fallback to old structure if needed
                  const dt = row.decision_tree as unknown;
                  const isWithPred = (
                    obj: unknown
                  ): obj is { predictions: Prediction; ispu?: ISPUInfo } => {
                    return (
                      typeof obj === "object" &&
                      obj !== null &&
                      "predictions" in (obj as object)
                    );
                  };

                  const getVal = (key: keyof Prediction) => {
                    // Try new structure first
                    if (measurements) {
                      const v = (measurements as Record<string, unknown>)[key];
                      if (typeof v === "number") return v.toFixed(1);
                    }
                    
                    // Fallback to old structure
                    if (!dt) return "-";
                    if (isWithPred(dt)) {
                      const v = dt.predictions[key];
                      return typeof v === "number" ? v.toFixed(1) : "-";
                    }
                    if (
                      typeof dt === "object" &&
                      dt !== null &&
                      key in (dt as object)
                    ) {
                      const v = (dt as Record<string, unknown>)[key];
                      return typeof v === "number" ? v.toFixed(1) : "-";
                    }
                    return "-";
                  };

                  const getISPU = () => {
                    // Use new structure if available
                    if (ispu_obj && ispu_obj.overall) {
                      return ispu_obj;
                    }
                    
                    // Fallback to old structure
                    if (!dt || !isWithPred(dt) || !dt.ispu) {
                      return { overall: "Baik", color: "#22c55e" };
                    }
                    return dt.ispu;
                  };

                  const ispu = getISPU();

                  return (
                    <tr
                      key={i}
                      style={{
                        background: i % 2 === 0 ? "#ffffff" : "#f9fafb",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      <td style={{ padding: "12px 16px", color: "#4a5568" }}>
                        {row.date ?? row.tanggal}
                      </td>
                      <td
                        style={{
                          padding: "12px 16px",
                          textAlign: "center",
                          fontWeight: 600,
                          color: "#2d3748",
                        }}
                      >
                        {getVal("pm25")}
                      </td>
                      <td
                        style={{
                          padding: "12px 16px",
                          textAlign: "center",
                          fontWeight: 600,
                          color: "#2d3748",
                        }}
                      >
                        {getVal("o3")}
                      </td>
                      <td
                        style={{
                          padding: "12px 16px",
                          textAlign: "center",
                          fontWeight: 600,
                          color: "#2d3748",
                        }}
                      >
                        {getVal("co")}
                      </td>
                      <td style={{ padding: "12px 16px", textAlign: "center" }}>
                        <span
                          style={{
                            background: ispu.color,
                            color: "white",
                            padding: "4px 12px",
                            borderRadius: 16,
                            fontSize: 13,
                            fontWeight: 600,
                            display: "inline-block",
                          }}
                        >
                          {ispu.overall}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
