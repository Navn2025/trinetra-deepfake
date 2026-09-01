import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Home } from "./pages/home/Home";
import { DashboardLayout } from "./components/layout/DashboardLayout";
import { AnalyzePage } from "./pages/analyze/AnalyzePage";
import { HistoryPage } from "./pages/history/HistoryPage";
import { ReportsPage } from "./pages/reports/ReportsPage";
import { ContactsPage } from "./pages/contacts/ContactsPage";
import { FamilyPage } from "./pages/family/FamilyPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<DashboardLayout />}>
          <Route index element={<AnalyzePage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="contacts" element={<ContactsPage />} />
          <Route path="family" element={<FamilyPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
