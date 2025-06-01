// src/components/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Make sure axios is imported
import { useNavigate, Outlet } from 'react-router-dom';
import {
  Box, Drawer, List, ListItem, ListItemButton, ListItemText, Collapse, Button,
  Typography, Avatar, Divider, Menu, MenuItem, ListItemIcon, Dialog, DialogTitle,
  DialogContent, TextField, DialogActions, Alert, CircularProgress, Slider,
  FormControl, InputLabel, Select
} from '@mui/material';
import {
  ExpandLess, ExpandMore, AccountCircle, Assessment, BarChart, Timeline,
  PieChart, ShowChart, SettingsApplications, ExitToApp, AddCircleOutline, History as HistoryIcon,
  Settings as SettingsIcon, LockReset as LockResetIcon, Dashboard as DashboardIcon,
  LightbulbOutlined as AiIcon // Icon for AI suggestions
} from '@mui/icons-material';
import DashboardContext from '../context/DashboardContext';

const drawerWidth = 280;

const navItems = [
  { text: 'Gantt Chart', icon: <Timeline />, path: 'gantt-chart' },
  { text: 'S-Curve', icon: <ShowChart />, path: 's-curve' },
  { text: 'Tornado Chart', icon: <BarChart sx={{ transform: 'rotate(90deg)' }} />, path: 'tornado-chart' },
  { text: 'Scatter Plot', icon: <PieChart />, path: 'scatter-plot' },
  { text: 'Cash Flow', icon: <Assessment />, path: 'cash-flow' },
  { text: 'Upload History', icon: <HistoryIcon />, path: 'upload-history' },
];

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [parsedData, setParsedData] = useState(null);
  const [scheduleAnalysis, setScheduleAnalysis] = useState(null);
  const [monteCarloResults, setMonteCarloResults] = useState(null);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessingFile, setIsProcessingFile] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [isRiskSettingsOpen, setIsRiskSettingsOpen] = useState(true);
  const [nSimulations, setNSimulations] = useState(1000);
  const [bufferDays, setBufferDays] = useState(0);
  const [riskId, setRiskId] = useState('R001');
  const [riskDesc, setRiskDesc] = useState('');
  const [riskProb, setRiskProb] = useState(20);
  const [riskImpact, setRiskImpact] = useState(40);
  const [riskTask, setRiskTask] = useState('');
  const [userRisks, setUserRisks] = useState([]);
  const [riskError, setRiskError] = useState('');
  const [riskSuccess, setRiskSuccess] = useState('');
  
  const navigate = useNavigate();
  const userName = localStorage.getItem('name') || 'User';
  const userEmail = localStorage.getItem('email') || 'user@example.com'; 

  const [anchorElProfileMenu, setAnchorElProfileMenu] = useState(null);
  const [openSettingsDialog, setOpenSettingsDialog] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [passwordChangeError, setPasswordChangeError] = useState('');
  const [passwordChangeSuccess, setPasswordChangeSuccess] = useState('');
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  // State for AI Mitigation Suggestions
  const [aiMitigationSuggestions, setAiMitigationSuggestions] = useState([]);
  const [isLoadingAiSuggestions, setIsLoadingAiSuggestions] = useState(false);


  useEffect(() => {
    let timer;
    if (error || successMessage || passwordChangeError || passwordChangeSuccess || riskError || riskSuccess) {
      timer = setTimeout(() => {
        setError(''); setSuccessMessage('');
        setPasswordChangeError(''); setPasswordChangeSuccess('');
        setRiskError(''); setRiskSuccess('');
      }, 6000);
    }
    return () => clearTimeout(timer);
  }, [error, successMessage, passwordChangeError, passwordChangeSuccess, riskError, riskSuccess]);

  const handleProfileMenuOpen = (event) => setAnchorElProfileMenu(event.currentTarget);
  const handleProfileMenuClose = () => setAnchorElProfileMenu(null);
  const handleOpenSettings = () => { setOpenSettingsDialog(true); handleProfileMenuClose(); };
  const handleCloseSettingsDialog = () => {
    setOpenSettingsDialog(false); setCurrentPassword(''); setNewPassword('');
    setConfirmNewPassword(''); setPasswordChangeError(''); setPasswordChangeSuccess('');
  };

  const handlePasswordChange = async () => {
    setPasswordChangeError(''); setPasswordChangeSuccess('');
    if (newPassword !== confirmNewPassword) { setPasswordChangeError("New passwords do not match."); return; }
    if (newPassword.length < 8) { setPasswordChangeError("New password must be at least 8 characters."); return; }
    setIsChangingPassword(true);
    try {
      const token = localStorage.getItem('token');
      await axios.post('http://localhost:8000/api/user/change-password',
        { current_password: currentPassword, new_password: newPassword },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setPasswordChangeSuccess("Password changed successfully! Dialog will close shortly.");
      setTimeout(handleCloseSettingsDialog, 2500);
    } catch (err) {
      setPasswordChangeError(err.response?.data?.detail || "Failed to change password.");
    } finally {
      setIsChangingPassword(false);
    }
  };
  
  const handleAddRisk = () => {
    if (!riskId.trim() || !riskDesc.trim() || !riskTask) { setRiskError('Risk ID, Description, and Affected Task are required.'); return; }
    const newRisk = { "Risk ID": riskId, Description: riskDesc, Probability: riskProb / 100, "Impact (hours)": riskImpact, "Affected Task": riskTask };
    setUserRisks(prev => [...prev, newRisk]);
    const nextRiskNum = (userRisks.length > 0 ? Math.max(0, ...userRisks.map(r => parseInt(r["Risk ID"].substring(1)))) : 0) + 1;
    setRiskId(`R${String(nextRiskNum).padStart(3, '0')}`);
    setRiskDesc(''); setRiskProb(20); setRiskImpact(40); setRiskTask(''); setRiskError('');
    setRiskSuccess(`Risk "${newRisk["Risk ID"]}" added.`);
    setAiMitigationSuggestions([]); // Clear previous AI suggestions when adding a new risk manually
  };

  const handleLogout = () => {
    localStorage.removeItem('token'); localStorage.removeItem('name');
    localStorage.removeItem('isAdmin'); localStorage.removeItem('email');
    navigate('/login');
  };
  
  const handleFileChangeContext = (selectedFile) => {
    if (selectedFile) {
        setFile(selectedFile); setError('');
        setSuccessMessage(`Selected: "${selectedFile.name}". Click "Process Project File".`);
    }
  };

  const handleUploadContext = async () => {
    if (!file) { setError('Please select an XER file.'); return; }
    setIsProcessingFile(true); setError(''); setSuccessMessage('');
    setParsedData(null); setScheduleAnalysis(null); setMonteCarloResults(null);
    const formData = new FormData(); formData.append('xer_file', file);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post('http://localhost:8000/api/upload-xer', formData, {
        headers: { 'Content-Type': 'multipart/form-data', Authorization: `Bearer ${token}` },
      });
      setParsedData(response.data.data); 
      setSuccessMessage(`File processed. Tasks: ${response.data.data.TASK?.length || 0}.`);
      const analysisResponse = await axios.post('http://localhost:8000/api/schedule-check', {
        task_data: response.data.data.TASK || [], taskpred_data: response.data.data.TASKPRED || [],
      }, { headers: { Authorization: `Bearer ${token}` } });
      setScheduleAnalysis(analysisResponse.data);
      setSuccessMessage(prev => `${prev} Schedule check complete.`);
    } catch (err) { setError(err.response?.data?.detail || 'Failed to process XER file.');
    } finally { setIsProcessingFile(false); }
  };
  
  const handleRunAnalysisContext = async () => {
    if (!parsedData || !scheduleAnalysis) { setError('Please upload and process an XER file first.'); return; }
    setIsAnalyzing(true); setError(''); setSuccessMessage(''); setMonteCarloResults(null);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post('http://localhost:8000/api/run-analysis', {
        task_data: scheduleAnalysis.tasks || [], // Use tasks from schedule check which include CPM data
        taskpred_data: parsedData.TASKPRED || [],
        n_simulations: nSimulations, buffer_days: bufferDays, user_risks: userRisks,
      }, { headers: { Authorization: `Bearer ${token}` } });
      setMonteCarloResults(response.data);
      setSuccessMessage('Monte Carlo analysis complete.');
    } catch (err) { setError(err.response?.data?.detail || 'Failed to run Monte Carlo analysis.');
    } finally { setIsAnalyzing(false); }
  };

  const handleGetAiMitigations = async () => {
    if (!riskDesc.trim()) {
        setRiskError("Please provide a risk description to get AI suggestions.");
        return;
    }
    setIsLoadingAiSuggestions(true);
    setAiMitigationSuggestions([]);
    setRiskError(''); setRiskSuccess(''); // Clear other risk messages
    try {
        const token = localStorage.getItem('token');
        const payload = {
            risk_description: riskDesc,
            risk_probability: riskProb / 100,
            risk_impact_hours: riskImpact,
            affected_task: riskTask,
            project_context: "Engineering and Construction Project for JESA S.A." // Example context
        };
        const response = await axios.post('http://localhost:8000/api/ai/suggest-mitigations', payload, {
            headers: { Authorization: `Bearer ${token}` }
        });
        setAiMitigationSuggestions(response.data.suggestions || []);
        if ((response.data.suggestions || []).length === 0) {
            setRiskSuccess("AI analyzed the risk but did not return specific formatted suggestions. Check logs or try refining the risk description.");
        } else {
            setRiskSuccess("AI suggestions loaded.");
        }
    } catch (err) {
        setRiskError(err.response?.data?.detail || "Failed to fetch AI mitigation suggestions.");
    } finally {
        setIsLoadingAiSuggestions(false);
    }
  };


  const contextValue = {
    file, setFile: handleFileChangeContext, 
    parsedData, scheduleAnalysis, monteCarloResults,
    error, successMessage, isLoading, isProcessingFile, isAnalyzing,
    setError, setSuccessMessage,
    handleUpload: handleUploadContext, 
    handleRunAnalysis: handleRunAnalysisContext,
  };

  const sidebarContent = (
    <Box sx={{ p: 1.5, display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ mb: 2, py: 1, cursor: 'pointer', textAlign: 'center' }} onClick={() => navigate('/dashboard')}>
        <img src="/jesa-logo.png" alt="JESA Logo" style={{ width: '60%', maxWidth: '120px', height: 'auto' }} />
      </Box>
      <Divider sx={{ bgcolor: 'rgba(255,255,255,0.1)', mb: 2 }}/>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, p: 1.5, borderRadius: 1, background: 'rgba(255,255,255,0.05)', cursor: 'pointer', '&:hover': { background: 'rgba(255,255,255,0.1)' } }} onClick={handleProfileMenuOpen}>
        <Avatar sx={{ bgcolor: 'secondary.main', color: 'primary.main', mr: 1.5, width: 36, height: 36 }}> <AccountCircle /> </Avatar>
        <Box> <Typography variant="body1" sx={{ color: 'primary.contrastText', fontWeight: 500, lineHeight: 1.2 }} noWrap> {userName} </Typography> <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)', lineHeight: 1.2 }} noWrap> View Profile </Typography> </Box>
      </Box>
      <Menu anchorEl={anchorElProfileMenu} open={Boolean(anchorElProfileMenu)} onClose={handleProfileMenuClose} MenuListProps={{ 'aria-labelledby': 'profile-button' }} PaperProps={{ sx: { bgcolor: 'background.paper', boxShadow: 5, borderRadius:1.5 } }}>
        <MenuItem onClick={handleOpenSettings} sx={{py:1.5, px:2.5}}> <ListItemIcon><SettingsIcon fontSize="small" color="primary"/></ListItemIcon> <ListItemText>Settings</ListItemText> </MenuItem>
        <MenuItem onClick={handleLogout} sx={{py:1.5, px:2.5}}> <ListItemIcon><ExitToApp fontSize="small" color="primary"/></ListItemIcon> <ListItemText>Logout</ListItemText> </MenuItem>
      </Menu>

      <Typography variant="overline" sx={{ color: 'rgba(255,255,255,0.6)', pl: 1, mb: 0.5, mt:1 }}> Navigation </Typography>
      <List dense>
        <ListItem disablePadding> <ListItemButton onClick={() => navigate("/dashboard")} sx={{py: 1, borderRadius: 1.5}}> <DashboardIcon sx={{ mr: 1.5 }} /> <ListItemText primary="Overview" /> </ListItemButton> </ListItem>
        {navItems.map((item) => ( <ListItem key={item.text} disablePadding> <ListItemButton onClick={() => navigate(item.path)} sx={{ py: 1, borderRadius: 1.5 }}> {React.cloneElement(item.icon, { sx: { mr: 1.5 } })} <ListItemText primary={item.text} /> </ListItemButton> </ListItem> ))}
      </List>
      <Box sx={{ flexGrow: 1 }} />
      <Box sx={{ pt: 1 }}>
        <ListItemButton onClick={() => setIsRiskSettingsOpen(!isRiskSettingsOpen)} sx={{ borderRadius: 1.5, py: 1.2 }}> <SettingsApplications sx={{ mr: 1.5 }} /> <ListItemText primary="Risk Analysis Settings" /> {isRiskSettingsOpen ? <ExpandLess /> : <ExpandMore />} </ListItemButton>
        <Collapse in={isRiskSettingsOpen} timeout="auto" unmountOnExit sx={{ maxHeight: 'calc(100vh - 520px)', overflowY: 'auto', pr: 0.5, pl: 0.5, mt: 1, mr:-0.5, ml:-0.5 }}>
          <List dense sx={{p:1, bgcolor: 'rgba(0,0,0,0.05)', borderRadius:1.5}}>
             {[
              { label: "Simulations", type: "number", value: nSimulations, onChange: setNSimulations, props: { inputProps: { min: 100, max: 10000, step: 100 } } },
              { label: "Buffer Days", type: "number", value: bufferDays, onChange: setBufferDays, props: { inputProps: { min: 0, max: 100 } } },
              { label: "Risk ID", value: riskId, onChange: setRiskId },
              { label: "Risk Description", value: riskDesc, onChange: setRiskDesc, props: { multiline: true, rows: 2 } },
            ].map(field => ( <ListItem key={field.label} sx={{ px: 0.5, py: 0.8 }}> <TextField label={field.label} type={field.type || "text"} value={field.value} onChange={(e) => field.onChange(field.type === "number" ? parseInt(e.target.value) || 0 : e.target.value)} {...field.props} fullWidth size="small"/> </ListItem> ))}
            <ListItem sx={{ px: 0.5, py: 0.8, display: 'block' }}>
              <Typography gutterBottom id="risk-prob-slider-label" sx={{fontSize: '0.8rem'}}>Probability: {riskProb}%</Typography>
              <Slider value={riskProb} onChange={(e, newValue) => setRiskProb(newValue)} aria-labelledby="risk-prob-slider-label" min={0} max={100} size="small"/>
            </ListItem>
             <ListItem sx={{ px: 0.5, py: 0.8 }}> <TextField label="Impact (hours)" type="number" value={riskImpact} onChange={(e) => setRiskImpact(Math.max(0, parseInt(e.target.value) || 0))} InputProps={{ inputProps: { min: 0 } }} fullWidth size="small"/> </ListItem>
            <ListItem sx={{ px: 0.5, py: 0.8 }}>
              <FormControl fullWidth variant="outlined" size="small"> <InputLabel id="affected-task-select-label">Affected Task</InputLabel>
                <Select labelId="affected-task-select-label" value={riskTask} label="Affected Task" onChange={(e) => setRiskTask(e.target.value)} MenuProps={{ PaperProps: { sx: { maxHeight: 200 } } }}>
                  <MenuItem value=""><em>Select Task</em></MenuItem>
                  {parsedData?.TASK?.map((task) => (<MenuItem key={task.task_id} value={task.task_name}>{task.task_id} - {task.task_name}</MenuItem>))}
                </Select>
              </FormControl>
            </ListItem>
            {/* AI Mitigation Button */}
            <ListItem sx={{ px: 0.5, py: 1 }}>
                <Button 
                    variant="outlined" 
                    color="secondary" // White button with blue text for contrast on dark sidebar section
                    onClick={handleGetAiMitigations} 
                    disabled={isLoadingAiSuggestions || !riskDesc.trim()}
                    fullWidth
                    size="small"
                    startIcon={isLoadingAiSuggestions ? <CircularProgress size={16} color="inherit" /> : <AiIcon />}
                >
                    Get AI Mitigation Ideas
                </Button>
            </ListItem>
            {/* Display AI Suggestions */}
            {aiMitigationSuggestions.length > 0 && (
                <ListItem sx={{ display: 'block', px: 0.5, mt: 0.5 }}>
                    <Typography variant="caption" component="div" sx={{fontWeight:'bold'}}>AI Suggestions:</Typography>
                    <List dense sx={{ maxHeight: 100, overflow: 'auto', bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 1, p:0.5, mt:0.5 }}>
                        {aiMitigationSuggestions.map((suggestion, index) => ( 
                            <ListItemText 
                                key={index} 
                                primary={`• ${suggestion}`}
                                primaryTypographyProps={{fontSize: '0.75rem', whiteSpace: 'normal'}}
                            /> 
                        ))}
                    </List>
                </ListItem>
            )}
            {/* Add Risk Button */}
            <ListItem sx={{ px: 0.5, py: 1, mt: (aiMitigationSuggestions.length > 0 ? 1: 0) }}> 
                <Button variant="contained" color="secondary" onClick={handleAddRisk} startIcon={<AddCircleOutline />} fullWidth> Add Risk to Analysis </Button> 
            </ListItem>
            {riskError && <ListItem sx={{px:0.5}}><Alert severity="error" sx={{width:'100%', fontSize:'0.8rem'}}>{riskError}</Alert></ListItem>}
            {riskSuccess && <ListItem sx={{px:0.5}}><Alert severity="success" sx={{width:'100%', fontSize:'0.8rem'}}>{riskSuccess}</Alert></ListItem>}
            {userRisks.length > 0 && ( <ListItem sx={{ display: 'block', px: 0.5, mt: 1 }}> <Typography variant="caption" component="div">Current User Risks ({userRisks.length}):</Typography> <List dense sx={{ maxHeight: 80, overflow: 'auto', bgcolor: 'rgba(0,0,0,0.1)', borderRadius: 1, p:0.5 }}> {userRisks.map((risk, index) => ( <ListItemText key={index} primary={`${risk["Risk ID"]}: ${risk.Description.substring(0,25)}... (${risk.Probability * 100}%)`} primaryTypographyProps={{fontSize: '0.75rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}/> ))} </List> </ListItem> )}
          </List>
        </Collapse>
      </Box>
    </Box>
  );

  return (
    <DashboardContext.Provider value={contextValue}>
      <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
        <Drawer variant="permanent" sx={{ width: drawerWidth, flexShrink: 0, '& .MuiDrawer-paper': { width: drawerWidth, boxSizing: 'border-box' } }}>
          {sidebarContent}
        </Drawer>
        <Box component="main" sx={{ flexGrow: 1, p: { xs: 2, sm: 3 }, width: `calc(100% - ${drawerWidth}px)`}}>
          <Outlet /> 
        </Box>
        <Dialog open={openSettingsDialog} onClose={handleCloseSettingsDialog} PaperProps={{sx: {width: '100%', maxWidth: '500px', borderRadius: 2}}}>
            <DialogTitle sx={{bgcolor: 'primary.main', color: 'primary.contrastText', py:1.5, px:2}}>User Settings</DialogTitle>
            <DialogContent sx={{pt: '20px !important', px:3, pb:1}}>
                <Box component="form" noValidate onSubmit={(e) => { e.preventDefault(); handlePasswordChange(); }}>
                    <Typography variant="h6" gutterBottom>Profile Information</Typography>
                    <TextField margin="dense" label="Full Name" fullWidth variant="outlined" value={userName} InputProps={{readOnly: true}} sx={{mb:1}}/>
                    <TextField margin="dense" label="Email Address" fullWidth variant="outlined" value={userEmail} InputProps={{readOnly: true}} sx={{mb:2.5}}/>
                    <Divider sx={{my:2}}/>
                    <Typography variant="h6" gutterBottom>Change Password</Typography>
                    <TextField margin="dense" label="Current Password" type="password" fullWidth variant="outlined" required value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} error={!!passwordChangeError && passwordChangeError.toLowerCase().includes("current")} helperText={ (!!passwordChangeError && passwordChangeError.toLowerCase().includes("current")) ? passwordChangeError : ""} />
                    <TextField margin="dense" label="New Password (min 8 chars)" type="password" fullWidth variant="outlined" required value={newPassword} onChange={(e) => setNewPassword(e.target.value)} error={!!passwordChangeError && (passwordChangeError.toLowerCase().includes("new pass") || passwordChangeError.toLowerCase().includes("match"))}  helperText={ (!!passwordChangeError && (passwordChangeError.toLowerCase().includes("new pass") || passwordChangeError.toLowerCase().includes("match"))) ? passwordChangeError : ""}/>
                    <TextField margin="dense" label="Confirm New Password" type="password" fullWidth variant="outlined" required value={confirmNewPassword} onChange={(e) => setConfirmNewPassword(e.target.value)} sx={{mb:1}} error={!!passwordChangeError && passwordChangeError.toLowerCase().includes("match")} helperText={ (!!passwordChangeError && passwordChangeError.toLowerCase().includes("match")) ? passwordChangeError : ""}/>
                    {passwordChangeError && !passwordChangeError.toLowerCase().includes("current") && !passwordChangeError.toLowerCase().includes("new pass") && !passwordChangeError.toLowerCase().includes("match") && <Alert severity="error" sx={{mb:2, mt:1}}>{passwordChangeError}</Alert>}
                    {passwordChangeSuccess && <Alert severity="success" sx={{mb:2, mt:1}}>{passwordChangeSuccess}</Alert>}
                    <DialogActions sx={{p:0, pt:2}}> <Button onClick={handleCloseSettingsDialog} color="inherit">Cancel</Button> <Button type="submit" variant="contained" color="primary" disabled={isChangingPassword}> {isChangingPassword ? <CircularProgress size={24} color="inherit"/> : "Change Password"} </Button> </DialogActions>
                </Box>
            </DialogContent>
        </Dialog>
      </Box>
    </DashboardContext.Provider>
  );
};

export default Dashboard;