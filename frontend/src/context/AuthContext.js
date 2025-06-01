import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [isAdmin, setIsAdmin] = useState(localStorage.getItem('isAdmin') === 'true');

  const login = async (email, password) => {
  try {
    const response = await axios.post('http://localhost:8000/api/auth/login', { email, password });
    setToken(response.data.access_token);
    setIsAdmin(response.data.is_admin); // Set from backend response
    localStorage.setItem('token', response.data.access_token);
    localStorage.setItem('isAdmin', String(response.data.is_admin)); // Store as string
    console.log("Login successful - isAdmin:", response.data.is_admin); // Debug
    // To make loginResponse available to Login.jsx as in my suggestion:
    return response.data; // <--- Add this return
  } catch (error) {
    throw error;
  }
};

  const logout = () => {
    setToken(null);
    setIsAdmin(false);
    localStorage.removeItem('token');
    localStorage.removeItem('isAdmin');
  };

  useEffect(() => {
    const checkAdminStatus = async () => {
      if (token) {
        try {
          const response = await axios.get('http://localhost:8000/api/admin/check', {
            headers: { Authorization: `Bearer ${token}` }
          });
          setIsAdmin(response.data.is_admin); // Update if backend confirms
          localStorage.setItem('isAdmin', response.data.is_admin);
          console.log("Checked admin status - isAdmin:", response.data.is_admin);
        } catch (error) {
          setIsAdmin(false);
          localStorage.setItem('isAdmin', 'false');
          console.log("Admin check failed:", error.message);
        }
      }
    };
    checkAdminStatus();
  }, [token]);

  return (
    <AuthContext.Provider value={{ token, isAdmin, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);