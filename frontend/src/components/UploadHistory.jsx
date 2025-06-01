// src/components/UploadHistory.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Typography, Box, Paper, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, CircularProgress, Alert, Tooltip, IconButton,
  Button // <<<<<<<<<<<<<<<<<<<< ADDED THIS IMPORT
} from '@mui/material';
import { FileDownload as FileDownloadIcon, Refresh as RefreshIcon } from '@mui/icons-material';

const UploadHistory = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setError('Authentication token not found. Please log in again.');
        setLoading(false);
        return;
      }
      const response = await axios.get('http://localhost:8000/api/user/upload-history', {
        headers: { Authorization: `Bearer ${token}` },
      });
      setHistory(response.data || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch upload history.');
      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  if (error) {
    return (
        <Box sx={{ p: 3 }}>
            <Alert severity="error" action={
                <Tooltip title="Retry">
                    <IconButton color="inherit" size="small" onClick={fetchHistory} disabled={loading}>
                        <RefreshIcon />
                    </IconButton>
                </Tooltip>
            }>
                {error}
            </Alert>
        </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          My Upload History
        </Typography>
        <Tooltip title="Refresh History">
            <span>
                <IconButton onClick={fetchHistory} color="primary" disabled={loading}>
                    <RefreshIcon />
                </IconButton>
            </span>
        </Tooltip>
      </Box>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}><CircularProgress /></Box>
      ) : history.length === 0 ? (
        <Paper elevation={0} sx={{p:3, textAlign:'center', border: '1px dashed', borderColor: 'divider'}}>
            <Typography color="text.secondary">No XER files have been uploaded yet.</Typography>
            <Button variant="outlined" color="primary" onClick={() => window.location.href = '/dashboard'} sx={{mt:2}}>
                Upload First File
            </Button>
        </Paper>
      ) : (
        <TableContainer component={Paper} sx={{border: '1px solid', borderColor: 'divider'}}>
          <Table stickyHeader aria-label="upload history table">
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell align="right">Size (KB)</TableCell>
                <TableCell>Uploaded At</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {history.map((item) => (
                <TableRow key={item.id} hover sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                  <TableCell component="th" scope="row">
                    <Typography variant="body2" sx={{fontWeight: 500}}>{item.filename}</Typography>
                  </TableCell>
                  <TableCell align="right">
                    {item.file_size ? (item.file_size / 1024).toFixed(2) : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {new Date(item.uploaded_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Typography 
                        variant="caption" 
                        sx={{
                            color: item.status?.includes('success') ? 'success.dark' : item.status?.includes('fail') ? 'error.dark' : 'text.secondary',
                            fontWeight: 'medium',
                            bgcolor: item.status?.includes('success') ? 'success.light' : item.status?.includes('fail') ? 'error.light' : 'action.disabledBackground',
                            px: 1.2, py:0.6, borderRadius: 1, display:'inline-block', textTransform: 'capitalize'
                        }}
                    >
                        {item.status?.replace('_', ' ') || 'Unknown'}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default UploadHistory;