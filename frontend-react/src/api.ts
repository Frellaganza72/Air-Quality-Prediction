import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:2000",
  timeout: 10000,
});

export async function getDashboard(date?: string) {
  const params = date ? { date } : {};
  const res = await api.get("/api/dashboard", { params });
  return res.data;
}

export async function getRecommendations() {
  const res = await api.get("/api/recommendations");
  return res.data;
}

export async function getHistory(limit = 30) {
  const res = await api.get("/api/history", { params: { limit } });
  return res.data;
}

export async function getAnomalies(limit = 10) {
  const res = await api.get("/api/anomalies", { params: { limit } });
  return res.data;
}
