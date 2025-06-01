// src/components/DashboardHome.jsx
import React from 'react';
import {
  Box, Typography, Card, CardContent, CardActions, Button, CircularProgress, Alert,
  List, ListItem, ListItemText, Paper, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow
} from '@mui/material';
import { UploadFile, CheckCircleOutline, WarningAmberOutlined, ErrorOutline } from '@mui/icons-material';
import { useDashboardContext } from '../context/DashboardContext'; // Import the consumer hook

const DashboardHome = () => {
  const {
    file,
    setFile, // This is actually handleFileChangeContext from Dashboard.jsx
    parsedData,
    scheduleAnalysis,
    monteCarloResults,
    error,
    successMessage,
    isLoading, // General loading state from context
    isProcessingFile, // Specific for file processing
    isAnalyzing,    // Specific for MC analysis
    setError,       // Allow DashboardHome to set global errors if needed
    setSuccessMessage, // Allow DashboardHome to set global success messages
    handleUpload,
    handleRunAnalysis
  } = useDashboardContext();

  // Local file change handler that then calls the context's handler
  const localHandleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (setFile && selectedFile) { // setFile here is handleFileChangeContext
        setFile(selectedFile); 
    } else if (selectedFile) {
        // Fallback if context didn't pass setFile directly, though it should
        console.warn("setFile from context is not available directly, consider passing a file handler");
    }
  };
  
  const localHandleDrop = (event) => {
    event.preventDefault(); event.stopPropagation();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.xer')) {
        if (setFile) setFile(droppedFile); // Use context's setFile (which is handleFileChangeContext)
        if (setError) setError('');
        if (setSuccessMessage) setSuccessMessage(`Dropped: "${droppedFile.name}". Click "Process Project File".`);
    } else {
      if (setError) setError('Invalid file type. Please drop a .xer file.');
    }
  };


  return (
    <Box>
      <Typography variant="h3" gutterBottom sx={{ mb: 3 }}>
        JESA Risk Analysis Dashboard
      </Typography>

      {(isLoading || isProcessingFile || isAnalyzing) && ( // More encompassing loading
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}>
          <CircularProgress color="primary" />
          <Typography sx={{ ml: 2 }}>
            {isProcessingFile ? "Processing File..." : isAnalyzing ? "Running Analysis..." : "Loading..."}
          </Typography>
        </Box>
      )}
      {error && <Alert severity="error" icon={<ErrorOutline />} sx={{ my: 2 }}>{error}</Alert>}
      {successMessage && <Alert severity="success" sx={{ my: 2 }}>{successMessage}</Alert>}

      {!isProcessingFile && !parsedData && !isLoading && (
        <Card sx={{mb:3, p:3, textAlign:'center', borderTop: `4px solid`, borderTopColor: 'primary.main'}}>
            <Typography variant="h5" gutterBottom>Get Started</Typography>
            <Typography color="text.secondary">
                Upload your XER project file to begin the risk analysis process. 
                Define custom risks (in the sidebar) and run simulations to understand potential project delays.
            </Typography>
        </Card>
      )}

      <Card sx={{ mb: 3, overflow: 'visible' }}>
        <CardContent sx={{ pb: 1 }}>
          <Typography variant="h4" gutterBottom>Project File Upload</Typography>
          <Box
            sx={{
              border: `2px dashed`, borderColor: 'divider', borderRadius: 1,
              p: { xs: 2, md: 3 }, textAlign: 'center',
              bgcolor: isProcessingFile ? 'action.disabledBackground' : 'transparent', mb: 2,
              '&:hover': { borderColor: 'primary.main', bgcolor: 'action.hover' }
            }}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onDrop={localHandleDrop}
          >
            <UploadFile sx={{ fontSize: 48, color: 'primary.main', mb: 1, opacity: 0.7 }} />
            <Typography variant="h6" sx={{ color: 'text.primary', mb: 0.5 }}>
              {file ? `Selected: ${file.name}` : 'Drag & drop XER file here'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              or click to select a file
            </Typography>
            <Button variant="outlined" color="primary" component="label" disabled={isProcessingFile}>
              Select File <input type="file" accept=".xer" hidden onChange={localHandleFileChange} />
            </Button>
          </Box>
          {parsedData && !error && (
            <Alert severity="info" variant="outlined" icon={<CheckCircleOutline />}>
              Successfully parsed: {parsedData.TASK?.length || 0} tasks.
            </Alert>
          )}
        </CardContent>
        <CardActions sx={{ justifyContent: 'flex-end', px: 2, pb: 2 }}>
          <Button
            variant="contained" color="primary" onClick={handleUpload}
            disabled={!file || isProcessingFile}
            startIcon={isProcessingFile ? <CircularProgress size={20} color="inherit" /> : null}
          >
            Process Project File
          </Button>
        </CardActions>
      </Card>

      {scheduleAnalysis && !error && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h4" gutterBottom>Schedule Health Check</Typography>
            {scheduleAnalysis.issues.length > 0 ? (
              <Alert severity="warning" icon={<WarningAmberOutlined />} sx={{ mb: 2 }}>
                <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
                  Issues Found ({scheduleAnalysis.issues.length}):
                </Typography>
                <List dense disablePadding>
                  {scheduleAnalysis.issues.map((issue, index) => (
                    <ListItem key={index} dense disableGutters>
                      <ListItemText primary={`• ${issue}`} />
                    </ListItem>
                  ))}
                </List>
              </Alert>
            ) : (
              <Alert severity="success" variant="outlined">No schedule integrity issues detected.</Alert>
            )}
            <Typography variant="h6" sx={{ mt: 2.5, mb: 0.5 }}>Baseline Project Duration</Typography>
            <Typography variant="h3" color="primary.main">
              {(scheduleAnalysis.project_duration / 24).toFixed(1)}
              <Typography variant="h5" component="span" color="text.secondary" sx={{ ml: 1 }}>days</Typography>
            </Typography>
          </CardContent>
          <CardActions sx={{ justifyContent: 'flex-end', px: 2, pb: 2 }}>
            <Button
              variant="contained" color="primary" onClick={handleRunAnalysis}
              disabled={isAnalyzing || !scheduleAnalysis || isProcessingFile}
              startIcon={isAnalyzing ? <CircularProgress size={20} color="inherit" /> : null}
            >
              Run Risk Analysis
            </Button>
          </CardActions>
        </Card>
      )}

      {monteCarloResults && !error && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h4" gutterBottom>Monte Carlo Simulation Results</Typography>
            <Typography variant="h5" sx={{ mb: 1.5 }}>Key Milestones Review</Typography>
            <TableContainer component={Paper} sx={{ maxHeight: 350, border: '1px solid', borderColor: 'divider' }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Milestone / Task Name</TableCell>
                    <TableCell align="right">Target Start</TableCell>
                    <TableCell align="right">Target End</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {monteCarloResults.tasks.filter(task => task.is_milestone).length > 0 ?
                    monteCarloResults.tasks.filter(task => task.is_milestone).map((task, index) => (
                      <TableRow key={`milestone-${index}`} hover>
                        <TableCell component="th" scope="row" sx={{ fontWeight: 500 }}>{task.task_name}</TableCell>
                        <TableCell align="right">{task.target_start_date || 'N/A'}</TableCell>
                        <TableCell align="right">{task.target_end_date || 'N/A'}</TableCell>
                      </TableRow>
                    )) :
                    <TableRow><TableCell colSpan={3} align="center" sx={{ py: 3, color: 'text.secondary' }}>No milestones identified.</TableCell></TableRow>
                  }
                </TableBody>
              </Table>
            </TableContainer>
            <Typography variant="h5" sx={{ mt: 3, mb: 1.5 }}>Project Duration Estimates (P-Values)</Typography>
            <Box sx={{ display: 'flex', flexWrap:'wrap', justifyContent: 'space-around', textAlign:'center', gap: 2, mt: 2, p:2.5, bgcolor: 'rgba(29,59,126,0.03)', borderRadius:1.5 }}>
                <Box sx={{minWidth: '120px'}}><Typography variant='overline' display="block" color="text.secondary">Optimistic (P10)</Typography><Typography variant='h4' color="primary.main">{monteCarloResults.project_optimistic.toFixed(1)}</Typography><Typography variant='body2' color="text.secondary">days</Typography></Box>
                <Box sx={{minWidth: '120px'}}><Typography variant='overline' display="block" color="text.secondary">Most Likely (P50)</Typography><Typography variant='h4' color="primary.main">{monteCarloResults.project_most_likely.toFixed(1)}</Typography><Typography variant='body2' color="text.secondary">days</Typography></Box>
                <Box sx={{minWidth: '120px'}}><Typography variant='overline' display="block" color="text.secondary">Pessimistic (P90)</Typography><Typography variant='h4' color="primary.main">{monteCarloResults.project_pessimistic.toFixed(1)}</Typography><Typography variant='body2' color="text.secondary">days</Typography></Box>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default DashboardHome;