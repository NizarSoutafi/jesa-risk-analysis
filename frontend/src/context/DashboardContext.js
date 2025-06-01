// src/context/DashboardContext.js
import React, { createContext, useContext } from 'react';

const DashboardContext = createContext(null);

export const useDashboardContext = () => {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error('useDashboardContext must be used within a DashboardProvider');
  }
  return context;
};

export default DashboardContext;