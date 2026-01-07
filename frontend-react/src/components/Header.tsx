type Props = {
  tab: string;
  setTab: (t: string) => void;
};

export default function Header({ tab, setTab }: Props) {
  return (
    <header
      style={{
        background: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        color: "#fff",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      {/* Title Section */}
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "12px 20px",
          textAlign: "center",
        }}
      >
        <h1
          style={{
            margin: 0,
            fontSize: 28,
            fontWeight: 700,
            letterSpacing: "0.5px",
            textShadow: "0 2px 4px rgba(0,0,0,0.1)",
          }}
        >
          Prediksi Kualitas Udara
        </h1>
      </div>

      {/* Navigation Tabs */}
      <nav
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          background: "rgba(255,255,255,0.15)",
          display: "flex",
          justifyContent: "center",
          gap: 8,
          padding: "8px 20px",
        }}
      >
        <button
          onClick={() => setTab("prediksi")}
          style={{
            padding: "10px 28px",
            borderRadius: 25,
            border: "none",
            background: tab === "prediksi" ? "#fff" : "transparent",
            color: tab === "prediksi" ? "#4facfe" : "#fff",
            fontSize: 15,
            fontWeight: 600,
            cursor: "pointer",
            transition: "all 0.3s ease",
            boxShadow:
              tab === "prediksi" ? "0 2px 8px rgba(0,0,0,0.15)" : "none",
          }}
        >
          Prediksi
        </button>
        <button
          onClick={() => setTab("rekomendasi")}
          style={{
            padding: "10px 28px",
            borderRadius: 25,
            border: "none",
            background: tab === "rekomendasi" ? "#fff" : "transparent",
            color: tab === "rekomendasi" ? "#4facfe" : "#fff",
            fontSize: 15,
            fontWeight: 600,
            cursor: "pointer",
            transition: "all 0.3s ease",
            boxShadow:
              tab === "rekomendasi" ? "0 2px 8px rgba(0,0,0,0.15)" : "none",
          }}
        >
          Rekomendasi Tindakan
        </button>
        <button
          onClick={() => setTab("riwayat")}
          style={{
            padding: "10px 28px",
            borderRadius: 25,
            border: "none",
            background: tab === "riwayat" ? "#fff" : "transparent",
            color: tab === "riwayat" ? "#4facfe" : "#fff",
            fontSize: 15,
            fontWeight: 600,
            cursor: "pointer",
            transition: "all 0.3s ease",
            boxShadow:
              tab === "riwayat" ? "0 2px 8px rgba(0,0,0,0.15)" : "none",
          }}
        >
          Riwayat
        </button>
      </nav>
    </header>
  );
}
