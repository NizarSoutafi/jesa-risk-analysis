// src/components/SCurvePage.jsx
import React, { useMemo } from 'react';
import { Box, Typography, Paper, Alert } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useDashboardContext } from '../context/DashboardContext';
import { parseISO, format, eachDayOfInterval, differenceInDays } from 'date-fns';

const SCurvePage = () => {
  const { parsedData } = useDashboardContext();

  const sCurveData = useMemo(() => {
    if (!parsedData || !parsedData.TASK || parsedData.TASK.length === 0) {
      return null;
    }

    const tasks = parsedData.TASK.map(task => ({
      ...task,
      target_start_date: task.target_start_date ? parseISO(String(task.target_start_date)) : null,
      target_end_date: task.target_end_date ? parseISO(String(task.target_end_date)) : null,
      duration_hours: parseFloat(task.duration_hours) || 0,
    })).filter(task => 
        task.target_start_date instanceof Date && !isNaN(task.target_start_date) &&
        task.target_end_date instanceof Date && !isNaN(task.target_end_date) &&
        task.target_start_date < task.target_end_date && // Ensure start is before end
        task.duration_hours > 0
    );

    if (tasks.length === 0) {
      return [];
    }

    let minDate = tasks[0].target_start_date;
    let maxDate = tasks[0].target_end_date;

    tasks.forEach(task => {
      if (task.target_start_date < minDate) minDate = task.target_start_date;
      if (task.target_end_date > maxDate) maxDate = task.target_end_date;
    });
    
    if (!(minDate instanceof Date && !isNaN(minDate)) || !(maxDate instanceof Date && !isNaN(maxDate)) || minDate >= maxDate) {
        console.error("Invalid overall minDate or maxDate for S-Curve", {minDate, maxDate});
        return [];
    }

    const projectTimeline = eachDayOfInterval({ start: minDate, end: maxDate });
    let cumulativeDuration = 0;
    const dataPoints = [];

    projectTimeline.forEach(currentDate => {
      let dailyDurationIncrease = 0;
      tasks.forEach(task => {
        if (currentDate >= task.target_start_date && currentDate <= task.target_end_date) {
          const taskDurationDays = differenceInDays(task.target_end_date, task.target_start_date) + 1;
          if (taskDurationDays > 0) {
            dailyDurationIncrease += task.duration_hours / taskDurationDays;
          }
        }
      });
      cumulativeDuration += dailyDurationIncrease;
      dataPoints.push({
        date: format(currentDate, 'yyyy-MM-dd'),
        cumulativeWork: parseFloat((cumulativeDuration / 24).toFixed(2)),
      });
    });

    return dataPoints;
  }, [parsedData]);

  if (!parsedData || !parsedData.TASK || parsedData.TASK.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>S-Curve Analysis</Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          Please upload and process an XER file first to see the S-Curve.
        </Alert>
      </Box>
    );
  }

  // This is the block where the error likely was
  if (!sCurveData || sCurveData.length === 0) {
    return (
      <Box sx={{ p: 3 }}> {/* Ensure this Box is properly closed */}
        <Typography variant="h4" gutterBottom>S-Curve Analysis</Typography>
        <Alert severity="warning" sx={{ mt: 2 }}>
          Not enough valid task data (e.g., missing dates, zero durations, or inconsistent date ranges) to generate an S-Curve.
        </Alert> 
        {/* If there was </Alert>> </Paper> here, the fix is just </Alert> </Box> or wrap Alert in Paper if intended */}
      </Box> // Closing tag for the Box starting this conditional return
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        S-Curve Analysis (Cumulative Planned Work - Days)
      </Typography>
      <Paper elevation={3} sx={{ p: { xs: 1, sm: 2 }, mt: 2, height: '60vh', minHeight: '400px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={sCurveData}
            margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              angle={-45}
              textAnchor="end" 
              height={70} 
              tickFormatter={(tick) => format(parseISO(tick), 'MMM dd')}
              interval="preserveStartEnd"
            />
            <YAxis label={{ value: 'Cumulative Work (Person-Days)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
                labelFormatter={(label) => format(parseISO(label), 'PP')} 
                formatter={(value) => [`${Number(value).toFixed(2)} days`, "Cumulative Work"]}
            />
            <Legend verticalAlign="top" height={36}/>
            <Line type="monotone" dataKey="cumulativeWork" stroke="#1d3b7e" strokeWidth={2} activeDot={{ r: 6 }} dot={{r:2}} />
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};

export default SCurvePage;