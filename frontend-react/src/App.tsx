import { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import Dashboard from "./components/Dashboard";
import Recommendations from "./components/Recommendations";
import History from "./components/History";

function App() {
  const [tab, setTab] = useState("prediksi");

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <Header tab={tab} setTab={setTab} />
      <main style={{ flex: 1, overflow: "auto", background: "#f0f4f8" }}>
        {tab === "prediksi" && <Dashboard />}
        {tab === "rekomendasi" && <Recommendations />}
        {tab === "riwayat" && <History />}
      </main>
    </div>
  );
}

export default App;
