// src/components/GanttChartPage.jsx
import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { Box, Typography, Paper, Alert, Button, ButtonGroup, IconButton, Tooltip as MuiTooltip } from '@mui/material';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, 
    Tooltip as RechartsTooltip, 
    Legend, ResponsiveContainer, Cell, LabelList 
} from 'recharts';
import { useDashboardContext } from '../context/DashboardContext';
import { ZoomIn as ZoomInIcon, ZoomOut as ZoomOutIcon, RestartAlt as ResetZoomIcon } from '@mui/icons-material';

// Custom Tick for Y-Axis to truncate long task names
const CustomYAxisTick = ({ x, y, payload }) => {
  const MAX_TASK_NAME_LENGTH = 30; // Adjust this length as needed
  const taskName = payload.value || '';
  const truncatedName = taskName.length > MAX_TASK_NAME_LENGTH
    ? `${taskName.substring(0, MAX_TASK_NAME_LENGTH)}...`
    : taskName;

  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={0} dy={4} textAnchor="end" fill="#666" fontSize={9}>
        {truncatedName}
      </text>
    </g>
  );
};

const GanttChartPage = () => {
  const { scheduleAnalysis } = useDashboardContext();

  const initialXDomain = useMemo(() => {
    if (!scheduleAnalysis || !scheduleAnalysis.tasks || scheduleAnalysis.tasks.length === 0) {
      return [0, 1000]; 
    }
    const tasksWithTiming = scheduleAnalysis.tasks.filter(task => 
        task.task_name && 
        typeof task.early_start === 'number' && 
        typeof task.duration_hours === 'number'
    );
    if (tasksWithTiming.length === 0) return [0, 1000];
    const maxFinish = Math.max(...tasksWithTiming.map(d => (parseFloat(d.early_start) || 0) + (parseFloat(d.duration_hours) || 0)), 0);
    return [0, Math.ceil(maxFinish / 100) * 100 || 1000];
  }, [scheduleAnalysis]);

  const [xDomain, setXDomain] = useState(initialXDomain);

  useEffect(() => {
    setXDomain(initialXDomain);
  }, [initialXDomain]);
  
  const ganttData = useMemo(() => {
    if (!scheduleAnalysis || !scheduleAnalysis.tasks || scheduleAnalysis.tasks.length === 0) {
      return null;
    }
    const sortedTasks = [...scheduleAnalysis.tasks]
      .filter(task => task.task_name && typeof task.early_start === 'number' && typeof task.duration_hours === 'number')
      .sort((a, b) => (parseFloat(a.early_start) || 0) - (parseFloat(b.early_start) || 0));

    return sortedTasks.map(task => {
      const earlyStartHours = parseFloat(task.early_start) || 0;
      const durationHours = parseFloat(task.duration_hours) || 0;
      const progressPercent = parseFloat(task.phys_complete_pct) || 0; // Assuming 0-100 scale
      
      const completedDurationHours = durationHours * (progressPercent / 100);
      const remainingDurationHours = durationHours - completedDurationHours;

      return {
        task_name: task.task_name || task.task_id,
        start_offset_hours: earlyStartHours,
        // For stacked bar: completed part, then remaining part
        completed_duration_hours: completedDurationHours,
        remaining_duration_hours: remainingDurationHours,
        actual_duration_hours: durationHours, // Keep for tooltip and total bar length
        is_critical: task.is_critical || false,
        early_finish_hours: parseFloat(task.early_finish) || (earlyStartHours + durationHours),
        progress_percent: progressPercent,
      };
    });
  }, [scheduleAnalysis]);

  const handleZoom = useCallback((factor) => {
    setXDomain(prevDomain => {
      const [min, max] = prevDomain;
      const range = max - min;
      const newRange = range * factor;
      const center = min + range / 2;
      let newMin = center - newRange / 2;
      let newMax = center + newRange / 2;
      if (factor > 1) {
        newMin = Math.max(initialXDomain[0], newMin);
        newMax = Math.min(initialXDomain[1], newMax);
        if ((newMax - newMin) >= (initialXDomain[1] - initialXDomain[0]) || (newMax - newMin) > 15000 ) {
            return initialXDomain;
        }
      } else {
          if (newRange < 50) return prevDomain; 
      }
      newMin = Math.max(initialXDomain[0], newMin);
      if (newMax > initialXDomain[1] && factor < 1) newMax = initialXDomain[1];
      if (newMin >= newMax) return initialXDomain; 
      return [newMin, newMax];
    });
  }, [initialXDomain]);

  const handleResetZoom = useCallback(() => {
    setXDomain(initialXDomain);
  }, [initialXDomain]);

  if (!scheduleAnalysis || !scheduleAnalysis.tasks || scheduleAnalysis.tasks.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Project Gantt Chart</Typography>
        <Alert severity="info" sx={{ mt: 2 }}>
          Please upload, process an XER file, and run a schedule check first.
          The Gantt chart uses data from the 'Schedule Health Check' results (which includes CPM data).
        </Alert>
      </Box>
    );
  }

  if (!ganttData || ganttData.length === 0) {
     return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Project Gantt Chart</Typography>
        <Alert severity="warning" sx={{ mt: 2 }}>
          No valid task data with timing information available to generate Gantt Chart. Ensure tasks have early start and duration.
        </Alert>
      </Box>
    );
  }
  
  const yAxisWidth = 220;

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          Project Gantt Chart 
          <Typography variant="subtitle1" component="span" sx={{ml:1, color: 'text.secondary'}}>
              (Timing in Project Hours)
          </Typography>
        </Typography>
        <ButtonGroup variant="outlined" size="small" aria-label="zoom controls">
          <MuiTooltip title="Zoom In">
            <IconButton onClick={() => handleZoom(0.7)}><ZoomInIcon /></IconButton>
          </MuiTooltip>
          <MuiTooltip title="Zoom Out">
            <IconButton onClick={() => handleZoom(1.5)}><ZoomOutIcon /></IconButton>
          </MuiTooltip>
           <MuiTooltip title="Reset Zoom">
            <IconButton onClick={handleResetZoom}><ResetZoomIcon /></IconButton>
          </MuiTooltip>
        </ButtonGroup>
      </Box>
      <Paper elevation={3} sx={{ p: {xs:1, sm:2}, mt: 2, height: '75vh', minHeight: '500px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={ganttData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: yAxisWidth - 20, bottom: 30 }} 
            barCategoryGap="35%" 
            barSize={15} // Slightly thicker bars for progress
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
                type="number" 
                domain={xDomain} 
                allowDataOverflow={true}
                label={{ value: "Project Time (Hours from Start)", position: 'insideBottom', offset: -5, dy:10 }}
                allowDecimals={false}
                tickCount={8} 
            />
            <YAxis 
                dataKey="task_name" 
                type="category" 
                width={yAxisWidth} 
                tick={<CustomYAxisTick />} 
                interval={0} 
            />
            <RechartsTooltip
              formatter={(value, name, props) => {
                if (name === 'start_offset_hours') return null; 
                const task = props.payload;
                if (name === 'Progress') {
                    return [`${task.progress_percent.toFixed(1)}%`, name];
                }
                return [
                    `${parseFloat(value).toFixed(1)} hrs (${name})`, 
                    `Critical: ${task.is_critical ? 'Yes' : 'No'}`,
                    `ES: ${task.start_offset_hours.toFixed(1)} hrs`,
                    `EF: ${(task.start_offset_hours + task.actual_duration_hours).toFixed(1)} hrs`,
                    `Progress: ${task.progress_percent.toFixed(1)}%`
                ];
              }}
              labelFormatter={(label, payload) => { 
                  if (payload && payload.length > 0) {
                      return payload[0].payload.task_name; 
                  }
                  return label;
              }}
            />
            <Legend verticalAlign="top" wrapperStyle={{paddingBottom: '10px'}} payload={[
                { value: 'Planned (Non-Critical)', type: 'square', color: '#1d3b7e' },
                { value: 'Planned (Critical)', type: 'square', color: '#b71c1c' }, // Darker Red for critical
                { value: 'Progress', type: 'square', color: '#4caf50' } // Green for progress
            ]}/>
            
            {/* Invisible bar for the start offset */}
            <Bar dataKey="start_offset_hours" stackId="taskTime" fill="transparent" isAnimationActive={false} />
            
            {/* Bar for completed duration (progress) */}
            <Bar dataKey="completed_duration_hours" stackId="taskTime" name="Progress" isAnimationActive={false}>
                {ganttData.map((entry, index) => (
                    <Cell key={`cell-progress-${index}`} fill={entry.is_critical ? "#8BC34A" : "#4caf50"} /> // Lighter/darker green for critical/non-critical progress
                ))}
            </Bar>

            {/* Bar for remaining duration */}
            <Bar dataKey="remaining_duration_hours" stackId="taskTime" name="Remaining" isAnimationActive={false}>
              {ganttData.map((entry, index) => (
                <Cell key={`cell-remaining-${index}`} fill={entry.is_critical ? "#d32f2f" : "#1d3b7e"} /> 
              ))}
            </Bar>

          </BarChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};

export default GanttChartPage;