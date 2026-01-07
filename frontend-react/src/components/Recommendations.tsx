import { useEffect, useState } from "react";
import { getRecommendations } from "../api";

type RecommendationsResponse = {
  ispu_category: string;
  recommendations: Record<string, string[]>;
};

export default function Recommendations() {
  const [data, setData] = useState<RecommendationsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await getRecommendations();
        setData(res);
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
  if (!data)
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#666" }}>
        No recommendations available
      </div>
    );

  const recs = data.recommendations || {};
  const categoryIcons: Record<string, string> = {
    rumah_tangga: "🏠",
    transportasi: "🚗",
    kesehatan: "💊",
    perkantoran: "🏢",
    lingkungan: "🌿",
    komunitas: "👥",
  };
  const categoryLabels: Record<string, string> = {
    rumah_tangga: "Rumah Tangga",
    transportasi: "Transportasi",
    kesehatan: "Kesehatan",
    perkantoran: "Perkantoran",
    lingkungan: "Lingkungan",
    komunitas: "Komunitas",
  };

  return (
    <div
      style={{ background: "#f0f4f8", minHeight: "100vh", paddingBottom: 40 }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px" }}>
        <div
          style={{
            background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
            color: "white",
            padding: "20px 24px",
            borderRadius: 12,
            marginBottom: 24,
            boxShadow: "0 4px 12px rgba(59, 130, 246, 0.3)",
          }}
        >
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 600 }}>
            Rekomendasi Tindakan
          </h2>
          <div style={{ marginTop: 8, fontSize: 15, opacity: 0.95 }}>
            Kategori ISPU: <strong>{data.ispu_category}</strong>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(350px, 1fr))",
            gap: 20,
          }}
        >
          {Object.keys(recs).map((k, i) => (
            <div
              key={i}
              className="card"
              style={{
                border: "2px solid #e5e7eb",
                transition: "transform 0.2s, box-shadow 0.2s",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 16,
                }}
              >
                <span style={{ fontSize: 32 }}>{categoryIcons[k] || "📋"}</span>
                <h3
                  style={{
                    margin: 0,
                    fontSize: 18,
                    color: "#2d3748",
                    fontWeight: 600,
                  }}
                >
                  {categoryLabels[k] || k}
                </h3>
              </div>
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {recs[k].slice(0, 6).map((r: string, idx: number) => (
                  <li
                    key={idx}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                      marginBottom: 10,
                      fontSize: 14,
                      color: "#4a5568",
                      lineHeight: 1.6,
                    }}
                  >
                    <span
                      style={{
                        background: "#3b82f6",
                        color: "white",
                        borderRadius: "50%",
                        width: 20,
                        height: 20,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 11,
                        fontWeight: 600,
                        flexShrink: 0,
                        marginTop: 2,
                      }}
                    >
                      ✓
                    </span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
