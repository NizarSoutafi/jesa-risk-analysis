// src/components/TornadoChartPage.jsx
import React, { useMemo } from 'react';
import { Box, Typography, Paper, Alert } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, Cell,
  ReferenceLine // <<<<<<<<<<<<<<<<<<<< ADD ReferenceLine TO YOUR IMPORTS FROM RECHARTS
} from 'recharts';
import { useDashboardContext } from '../context/DashboardContext';
// Tooltip from MUI is not used in this specific component based on last version,
// but if you add MUI Tooltips, you would import it as MuiTooltip:
// import { Tooltip as MuiTooltip } from '@mui/material';


const TornadoChartPage = () => {
  const { monteCarloResults } = useDashboardContext(); 

  const tornadoData = useMemo(() => {
    if (!monteCarloResults || !monteCarloResults.task_sensitivities || monteCarloResults.task_sensitivities.length === 0) {
      return null;
    }

    const sortedSensitivities = [...monteCarloResults.task_sensitivities]
      .filter(item => typeof item.sensitivity === 'number' && item.task_name) 
      .sort((a, b) => Math.abs(b.sensitivity) - Math.abs(a.sensitivity))
      .slice(0, 15); 

    return sortedSensitivities.map(item => ({
      task_name: item.task_name,
      sensitivity: parseFloat(item.sensitivity.toFixed(4)), 
    }));
  }, [monteCarloResults]);

  if (!monteCarloResults || !monteCarloResults.task_sensitivities) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Tornado Chart (Task Sensitivity)</Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          Please run a Monte Carlo analysis first to see the Tornado Chart.
          This chart displays task sensitivities from the analysis results.
        </Alert>
      </Box>
    );
  }

  if (!tornadoData || tornadoData.length === 0) {
     return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Tornado Chart (Task Sensitivity)</Typography>
        <Alert severity="warning" sx={{ mt: 2 }}>
          No task sensitivity data available from the Monte Carlo analysis to generate the Tornado Chart.
        </Alert>
      </Box>
    );
  }

  const sensitivities = tornadoData.map(d => d.sensitivity);
  // Ensure sensitivities array is not empty before calling Math.min/max
  const minSensitivity = sensitivities.length > 0 ? Math.min(0, ...sensitivities) : 0; 
  const maxSensitivity = sensitivities.length > 0 ? Math.max(0, ...sensitivities) : 0;
  const absMax = Math.max(Math.abs(minSensitivity), Math.abs(maxSensitivity));
  // Ensure xDomain has a non-zero range if absMax is 0
  const xDomainRange = Math.ceil(absMax * 10)/10 || 0.1; 
  const xDomain = [-xDomainRange, xDomainRange]; 

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Tornado Chart 
        <Typography variant="subtitle1" component="span" sx={{ml:1, color: 'text.secondary'}}>
          (Task Sensitivity to Project Outcome)
        </Typography>
      </Typography>
      <Paper elevation={3} sx={{ p: {xs:1, sm:2}, mt: 2, height: '75vh', minHeight: '500px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={tornadoData}
            layout="vertical" 
            margin={{ top: 5, right: 40, left: 120, bottom: 20 }} 
            barCategoryGap="30%"
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
                type="number" 
                domain={xDomain} 
                label={{ value: "Sensitivity (e.g., Correlation Coefficient)", position: 'insideBottom', offset: -10, dy:10 }}
                tickFormatter={(value) => value.toFixed(2)} 
            />
            <YAxis 
                dataKey="task_name" 
                type="category" 
                width={180} 
                tick={{ fontSize: 9, width: 170, textAnchor: 'end' }} 
                interval={0} 
            />
            <RechartsTooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(4)}`, "Sensitivity"]}
                labelFormatter={(label) => `Task: ${label}`}
            />
            <Legend verticalAlign="top" wrapperStyle={{paddingBottom: '10px'}} payload={[
                { value: 'Positive Impact / Correlation', type: 'square', color: '#1d3b7e' }, // JESA Blue
                { value: 'Negative Impact / Correlation', type: 'square', color: '#c62828' }  // Example red
            ]}/>
            <ReferenceLine x={0} stroke="#666" strokeDasharray="2 2" /> 
            <Bar dataKey="sensitivity" name="Sensitivity" isAnimationActive={false}>
              {tornadoData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.sensitivity >= 0 ? "#1d3b7e" : "#c62828"} /> 
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};

export default TornadoChartPage;