// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import jesaTheme from './jesaTheme'; // Your custom theme
import Login from './components/Login';
import Signup from './components/SignUp';
import Dashboard from './components/Dashboard'; // Acts as the layout for /dashboard/*
import AdminPanel from './components/AdminPanel';
import UploadHistory from './components/UploadHistory'; // Component for upload history
import DashboardHome from './components/DashboardHome'; // New component for /dashboard main content
import { AuthProvider, useAuth } from './context/AuthContext';

const PrivateRoute = ({ children, adminOnly = false }) => {
  const { token, isAdmin } = useAuth();
  if (!token) {
    return <Navigate to="/login" />;
  }
  if (adminOnly && !isAdmin) {
    return <Navigate to="/dashboard" />; // Or an unauthorized page
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
              <Route index element={<DashboardHome />} /> {/* Content for /dashboard itself */}
              <Route path="upload-history" element={<UploadHistory />} />
              {/* You would add other dashboard sub-page routes here:
                e.g., <Route path="gantt-chart" element={<GanttChartComponent />} />
                      <Route path="s-curve" element={<SCurveComponent />} /> 
              */}
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
            {/* Optional: Add a 404 Not Found Page */}
            {/* <Route path="*" element={<NotFoundPage />} /> */}
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;