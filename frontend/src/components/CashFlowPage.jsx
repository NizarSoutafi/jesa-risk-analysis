// src/components/CashFlowPage.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { Box, Typography, Paper, Alert, CircularProgress } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';
import { useDashboardContext } from '../context/DashboardContext'; // To get parsedData (TASK and TASKRSRC)
import axios from 'axios';
import { format, parseISO } from 'date-fns';

const CashFlowPage = () => {
  const { parsedData } = useDashboardContext(); // From XER upload (contains TASK and TASKRSRC)
  const [cashFlowData, setCashFlowData] = useState([]);
  const [totalProjectCost, setTotalProjectCost] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchCashFlow = async () => {
      // Ensure we have the necessary parts of parsedData
      if (!parsedData || !parsedData.TASK ) { 
        setError('Task data must be loaded first to generate cash flow. Please process an XER file.');
        setCashFlowData([]);
        setLoading(false);
        return;
      }

      setLoading(true);
      setError('');
      try {
        const token = localStorage.getItem('token');
        // Prepare payload for the backend
        const requestPayload = {
            tasks: parsedData.TASK,
            task_rsrcs: parsedData.TASKRSRC || [] // Send TASKRSRC if available
        };

        const response = await axios.post('http://localhost:8000/api/cash-flow-data', requestPayload, {
          headers: { Authorization: `Bearer ${token}` },
        });

        setCashFlowData(response.data.data || []);
        setTotalProjectCost(response.data.total_project_cost || 0);

        if ((response.data.data || []).length === 0 && (response.data.total_project_cost || 0) === 0) {
            setError('No cost data found or processed to generate cash flow. Ensure XER file contains cost information.');
        }

      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to fetch or process cash flow data.');
        setCashFlowData([]);
        setTotalProjectCost(0);
      } finally {
        setLoading(false);
      }
    };

    if (parsedData && (parsedData.TASK)) { // Check if essential data is present
        fetchCashFlow();
    } else if (!parsedData) { // Only show initial message if no parsedData at all
        setError('Please upload and process an XER file first.');
    }

  }, [parsedData]); // Re-fetch when parsedData changes


  if (!parsedData) { 
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Project Cash Flow Analysis</Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          Please upload and process an XER file to generate cash flow data.
        </Alert>
      </Box>
    );
  }

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
        <Typography sx={{ml: 2}}>Generating Cash Flow Data...</Typography>
      </Box>
    );
  }

  if (error && cashFlowData.length === 0) { // Show error only if no data is displayed
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Project Cash Flow Analysis</Typography>
        <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>
      </Box>
    );
  }
  
  if (cashFlowData.length === 0) {
     return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Project Cash Flow Analysis</Typography>
        <Alert severity="warning" sx={{ mt: 2 }}>
          No cash flow data available to display. This might be due to missing cost information in the XER file. Total calculated project cost from available data: ${totalProjectCost.toFixed(2)}.
        </Alert>
      </Box>
    );
  }
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Project Cash Flow (Cumulative Cost)
      </Typography>
      <Typography variant="h6" color="text.secondary" gutterBottom>
        Total Estimated Project Cost: ${totalProjectCost.toFixed(2)}
      </Typography>
      <Paper elevation={3} sx={{ p: {xs:1, sm:2}, mt: 2, height: '70vh', minHeight: '450px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={cashFlowData}
            margin={{ top: 5, right: 30, left: 40, bottom: 25 }} // Adjusted left margin for Y-axis label
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date"
              angle={-45}
              textAnchor="end" 
              height={70} 
              tickFormatter={(tick) => format(parseISO(tick), 'MMM dd, yy')} // Formatted date
              interval="preserveStartEnd" // Auto-adjust ticks, ensuring start and end are shown
            />
            <YAxis 
                label={{ value: 'Cumulative Cost ($)', angle: -90, position: 'insideLeft', offset: -10, style:{textAnchor: 'middle'} }}
                tickFormatter={(value) => `$${Number(value).toLocaleString(undefined, {maximumFractionDigits: 0})}`} // Format as currency
            />
            <Tooltip
                labelFormatter={(label) => format(parseISO(label), 'PP')} // Full date in tooltip
                formatter={(value, name) => [`$${Number(value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`, "Cumulative Cost"]}
            />
            <Legend verticalAlign="top" height={36}/>
            <Line type="monotone" dataKey="cumulative_cost" name="Cumulative Cost" stroke="#1d3b7e" strokeWidth={3} activeDot={{ r: 7 }} dot={{r:3, strokeWidth:1}} />
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};

export default CashFlowPage;