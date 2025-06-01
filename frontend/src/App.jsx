// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import jesaTheme from './jesaTheme'; // Your custom theme
import Login from './components/Login';
import Signup from './components/SignUp';
import Dashboard from './components/Dashboard'; // Acts as the layout for /dashboard/*
import AdminPanel from './components/AdminPanel';
import UploadHistory from './components/UploadHistory';
import DashboardHome from './components/DashboardHome'; // For /dashboard main content
import { AuthProvider, useAuth } from './context/AuthContext';

// Import your chart page components
import GanttChartPage from './components/GanttChartPage';
import SCurvePage from './components/SCurvePage';
import TornadoChartPage from './components/TornadoChartPage';
import ScatterPlotPage from './components/ScatterPlotPage';
import CashFlowPage from './components/CashFlowPage';

const PrivateRoute = ({ children, adminOnly = false }) => {
  const { token, isAdmin } = useAuth();
  if (!token) {
    return <Navigate to="/login" />;
  }
  if (adminOnly && !isAdmin) {
    return <Navigate to="/dashboard" />;
  }
  return children;
};

function App() {
  return (
    <ThemeProvider theme={jesaTheme}>
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            
            <Route 
              path="/dashboard/*" // Parent route for all dashboard sections
              element={
                <PrivateRoute>
                  <Dashboard /> {/* Dashboard component provides sidebar and <Outlet/> */}
                </PrivateRoute>
              }
            >
              {/* Nested routes render inside Dashboard's <Outlet /> */}
              <Route index element={<DashboardHome />} /> 
              <Route path="upload-history" element={<UploadHistory />} />
              {/* Routes for your new chart pages */}
              <Route path="gantt-chart" element={<GanttChartPage />} />
              <Route path="s-curve" element={<SCurvePage />} />
              <Route path="tornado-chart" element={<TornadoChartPage />} />
              <Route path="scatter-plot" element={<ScatterPlotPage />} />
              <Route path="cash-flow" element={<CashFlowPage />} />
            </Route>

            <Route
              path="/admin"
              element={
                <PrivateRoute adminOnly={true}>
                  <AdminPanel />
                </PrivateRoute>
              }
            />
            <Route path="/" element={<Navigate to="/login" />} />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;