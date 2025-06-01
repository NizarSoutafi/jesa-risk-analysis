// src/components/ScatterPlotPage.jsx
import React, { useMemo } from 'react';
import { Box, Typography, Paper, Alert } from '@mui/material';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label
} from 'recharts';
import { useDashboardContext } from '../context/DashboardContext';

const ScatterPlotPage = () => {
  const { monteCarloResults } = useDashboardContext();

  const scatterData = useMemo(() => {
    if (!monteCarloResults || !monteCarloResults.all_simulated_durations || monteCarloResults.all_simulated_durations.length === 0) {
      return null;
    }

    return monteCarloResults.all_simulated_durations.map((duration, index) => ({
      simulationRun: index + 1,
      simulatedDurationDays: parseFloat(duration.toFixed(2)),
    }));
  }, [monteCarloResults]);

  if (!monteCarloResults || !monteCarloResults.all_simulated_durations) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Scatter Plot (Simulation Outcomes)</Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          Please run a Monte Carlo analysis first to see the Scatter Plot.
          This chart displays the distribution of simulated project durations.
        </Alert>
      </Box>
    );
  }

  if (!scatterData || scatterData.length === 0) {
     return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Scatter Plot (Simulation Outcomes)</Typography>
        <Alert severity="warning" sx={{ mt: 2 }}>
          No simulated duration data available from the Monte Carlo analysis.
        </Alert>
      </Box>
    );
  }
  
  // Determine domain for Y-axis (simulated duration)
  const durations = scatterData.map(d => d.simulatedDurationDays);
  const minYDuration = Math.min(...durations);
  const maxYDuration = Math.max(...durations);
  // Add some padding to the domain
  const yDomain = [
    Math.floor(minYDuration / 10) * 10, 
    Math.ceil(maxYDuration / 10) * 10
  ];


  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Scatter Plot of Simulated Project Durations
      </Typography>
      <Paper elevation={3} sx={{ p: {xs:1, sm:2}, mt: 2, height: '75vh', minHeight: '500px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{
              top: 20,
              right: 30,
              bottom: 30, // Increased for X-axis label
              left: 30,  // Increased for Y-axis label
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="number" 
              dataKey="simulationRun" 
              name="Simulation Run #"
              allowDecimals={false}
            >
              <Label value="Simulation Run #" offset={-20} position="insideBottom" />
            </XAxis>
            <YAxis 
              type="number" 
              dataKey="simulatedDurationDays" 
              name="Simulated Duration" 
              unit=" days"
              domain={yDomain}
            >
                <Label value="Simulated Project Duration (days)" angle={-90} position="insideLeft" style={{textAnchor: 'middle'}} />
            </YAxis>
            <Tooltip 
                cursor={{ strokeDasharray: '3 3' }} 
                formatter={(value, name) => {
                    if (name === "Simulated Duration") return [`${value} days`, name];
                    return [value, name];
                }}
            />
            <Legend verticalAlign="top" wrapperStyle={{paddingBottom: '10px'}}/>
            <Scatter 
                name="Simulated Durations" 
                data={scatterData} 
                fill="#1d3b7e" // JESA Blue
                shape="circle" // or "cross", "diamond", "square", "star", "triangle", "wye"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};

export default ScatterPlotPage;