# backend/main.py
import asyncpg
import jwt
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, OAuth2PasswordBearer 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
import logging
from datetime import datetime, timezone, timedelta
import os
import secrets
import io
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present
import pandas as pd
import numpy as np
import google.generativeai as genai # Import Gemini API library

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

app = FastAPI(title="JESA Risk Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mac@localhost:5432/jesa_risk_analysis")
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-and-strong-key-please-change") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Analysis Logic ---
# (read_xer_file_content, perform_schedule_check, calculate_cpm_and_duration, perform_monte_carlo_analysis functions
#  remain the same as the last version I provided where SyntaxErrors were fixed.)
def read_xer_file_content(file_content_str: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, List[Dict[str, Any]]] = {}
    current_table_name: Optional[str] = None
    columns: Optional[List[str]] = None
    lines = file_content_str.splitlines()
    for line_number, line_text in enumerate(lines):
        line_text = line_text.strip();
        if not line_text: continue
        try:
            if line_text.startswith('%T'):
                parts = line_text.split('\t'); current_table_name = parts[1].strip() if len(parts) > 1 else None
                if current_table_name: tables[current_table_name] = []; columns = None
                else: logger.warning(f"L{line_number}: Malformed %T: {line_text}"); continue
            if current_table_name and line_text.startswith('%F'):
                columns = [col.strip() for col in line_text.split('\t')[1:]]
                if not columns: logger.warning(f"L{line_number}: No cols for {current_table_name}: {line_text}"); current_table_name = None; continue
            if current_table_name and columns and (line_text.startswith('%R') or not line_text.startswith('%')):
                data_parts = line_text.split('\t'); data_values = data_parts[1:] if line_text.startswith('%R') else data_parts
                if len(data_values) < len(columns): data_values.extend([None] * (len(columns) - len(data_values)))
                elif len(data_values) > len(columns): data_values = data_values[:len(columns)]
                tables[current_table_name].append(dict(zip(columns, data_values)))
        except Exception as e: logger.error(f"Err L{line_number} ('{line_text}'): {e}"); continue
    dataframes: Dict[str, pd.DataFrame] = {}
    for table, rows in tables.items():
        if rows: dataframes[table] = pd.DataFrame(rows)
        else: logger.warning(f"Table '{table}' empty.")
    if 'TASK' in dataframes and not dataframes['TASK'].empty:
        df_task = dataframes['TASK'].copy() 
        date_cols = ['act_start_date', 'act_end_date', 'target_start_date', 'target_end_date', 'early_start_date', 'early_end_date', 'late_start_date', 'late_end_date', 'create_date', 'update_date']
        for col in date_cols:
            if col in df_task.columns: df_task[col] = pd.to_datetime(df_task[col], errors='coerce')
        df_task['target_drtn_hr_cnt'] = pd.to_numeric(df_task.get('target_drtn_hr_cnt'), errors='coerce').fillna(8.0)
        task_type_series = df_task.get('task_type', pd.Series(dtype=str))
        df_task['is_milestone'] = (task_type_series == 'TT_Mile') | (task_type_series == 'TT_FinMile') | (df_task['target_drtn_hr_cnt'] == 0)
        logger.info(f"Milestones identified in read_xer: {df_task['is_milestone'].sum()}")
        df_task['duration_hours'] = np.nan 
        if 'act_start_date' in df_task.columns and 'act_end_date' in df_task.columns:
            valid_actual_dates = df_task['act_start_date'].notna() & df_task['act_end_date'].notna()
            if valid_actual_dates.any():
                time_diff_actual = df_task.loc[valid_actual_dates, 'act_end_date'] - df_task.loc[valid_actual_dates, 'act_start_date']
                if not time_diff_actual.empty: df_task.loc[valid_actual_dates, 'duration_hours'] = time_diff_actual.dt.total_seconds() / 3600
        needs_target_fallback = df_task['duration_hours'].isna()
        if needs_target_fallback.any() and 'target_start_date' in df_task.columns and 'target_end_date' in df_task.columns:
            valid_target_dates = df_task['target_start_date'].notna() & df_task['target_end_date'].notna() & needs_target_fallback
            if valid_target_dates.any():
                time_diff_target = df_task.loc[valid_target_dates, 'target_end_date'] - df_task.loc[valid_target_dates, 'target_start_date']
                if not time_diff_target.empty: df_task.loc[valid_target_dates, 'duration_hours'] = time_diff_target.dt.total_seconds() / 3600
        df_task['duration_hours'] = df_task['duration_hours'].fillna(df_task['target_drtn_hr_cnt'])
        df_task['duration_hours'] = np.where(df_task['is_milestone'], 0.0, df_task['duration_hours'])
        df_task['duration_hours'] = df_task['duration_hours'].clip(lower=0.0)
        dataframes['TASK'] = df_task
    else: logger.warning("TASK table not found or empty in read_xer_file_content.")
    if 'TASKRSRC' in dataframes and not dataframes['TASKRSRC'].empty:
        df_taskrsrc = dataframes['TASKRSRC'].copy()
        cost_qty_cols = ['target_cost', 'target_qty', 'act_reg_cost', 'act_ot_cost', 'remain_cost', 'act_reg_qty', 'act_ot_qty', 'remain_qty', 'cost_per_qty']
        for col in cost_qty_cols:
            if col in df_taskrsrc.columns: df_taskrsrc[col] = pd.to_numeric(df_taskrsrc[col], errors='coerce').fillna(0.0)
        date_cols_rsrc = ['target_start_date', 'target_end_date', 'act_start_date', 'act_end_date']
        for col in date_cols_rsrc:
            if col in df_taskrsrc.columns: df_taskrsrc[col] = pd.to_datetime(df_taskrsrc[col], errors='coerce')
        dataframes['TASKRSRC'] = df_taskrsrc
        logger.info(f"--- DataFrame 'TASKRSRC' dtypes after read_xer: ---\n{df_taskrsrc.dtypes}")
        if 'target_cost' in df_taskrsrc.columns: logger.info(f"Sum of target_cost in TASKRSRC: {df_taskrsrc['target_cost'].sum()}")
    else: logger.warning("TASKRSRC table not found or empty."); dataframes['TASKRSRC'] = pd.DataFrame() 
    return dataframes

def perform_schedule_check(tasks_df: pd.DataFrame, taskpred_df: pd.DataFrame) -> List[str]:
    issues = []
    if tasks_df.empty: issues.append("No task data available for schedule check."); return issues
    if 'task_id' not in tasks_df.columns: issues.append("TASK data is missing 'task_id' column."); return issues
    if not taskpred_df.empty and 'pred_task_id' in taskpred_df.columns and 'task_id' in taskpred_df.columns:
        valid_task_ids = set(tasks_df['task_id'].astype(str)); taskpred_df_copy = taskpred_df.copy()
        taskpred_df_copy['pred_task_id_str'] = taskpred_df_copy['pred_task_id'].astype(str)
        missing_preds = taskpred_df_copy[~taskpred_df_copy['pred_task_id_str'].isin(valid_task_ids)]
        if not missing_preds.empty: issues.append(f"{len(missing_preds)} relationships point to predecessor tasks not found (e.g., {missing_preds['pred_task_id_str'].unique().tolist()[:3]}...).")
    if 'duration_hours' in tasks_df.columns and 'is_milestone' in tasks_df.columns:
        invalid_durations = tasks_df[(tasks_df['duration_hours'] <= 0) & (tasks_df['is_milestone'] == False)]
        if not invalid_durations.empty: issues.append(f"{len(invalid_durations)} non-milestone tasks with zero or negative duration (e.g., {invalid_durations.get('task_name', pd.Series(dtype=str)).tolist()[:3]}...).")
    else: issues.append("Cannot check task durations: 'duration_hours' or 'is_milestone' column missing.")
    if not issues: issues.append("Basic schedule integrity check passed.")
    return issues

def calculate_cpm_and_duration(tasks_df_orig: pd.DataFrame, taskpred_df_orig: pd.DataFrame) -> (pd.DataFrame, float):
    if tasks_df_orig.empty: logger.error("CPM: Task data empty."); return tasks_df_orig.copy(), 0.0
    required_task_cols = ['task_id', 'duration_hours']; tasks_df = tasks_df_orig.copy() 
    for col in required_task_cols:
        if col not in tasks_df.columns:
            logger.error(f"CPM: Task data missing required column: {col}.")
            if col == 'duration_hours': tasks_df[col] = 0.0 
            else: return tasks_df, 0.0
    taskpred_df = taskpred_df_orig.copy() if not taskpred_df_orig.empty else pd.DataFrame(columns=['task_id', 'pred_task_id'])
    tasks_df['task_id'] = tasks_df['task_id'].astype(str)
    if not taskpred_df.empty: 
        if 'task_id' in taskpred_df.columns: taskpred_df['task_id'] = taskpred_df['task_id'].astype(str)
        if 'pred_task_id' in taskpred_df.columns: taskpred_df['pred_task_id'] = taskpred_df['pred_task_id'].astype(str)
    task_dict = {str(row['task_id']): row.to_dict() for _, row in tasks_df.iterrows()}
    for task_id_val in task_dict: task_dict[task_id_val].update({'early_start': 0.0, 'early_finish': 0.0, 'late_start': float('inf'), 'late_finish': float('inf')})
    predecessors_map = {task_id_val: [] for task_id_val in task_dict}; successors_map = {task_id_val: [] for task_id_val in task_dict}
    if not taskpred_df.empty and 'task_id' in taskpred_df.columns and 'pred_task_id' in taskpred_df.columns:
        for _, row in taskpred_df.iterrows():
            if row['pred_task_id'] in task_dict and row['task_id'] in task_dict:
                successors_map[row['pred_task_id']].append(row['task_id']); predecessors_map[row['task_id']].append(row['pred_task_id'])
    processing_order = list(tasks_df['task_id']) 
    for _ in range(len(task_dict) + 2): 
        updated_in_pass = False
        for task_id_val in processing_order:
            if task_id_val not in task_dict: continue
            current_task_data = task_dict[task_id_val]; max_ef_of_predecessors = 0.0
            if task_id_val in predecessors_map:
                for pred_id in predecessors_map[task_id_val]:
                    if pred_id in task_dict: max_ef_of_predecessors = max(max_ef_of_predecessors, task_dict[pred_id].get('early_finish', 0.0))
            new_es = max_ef_of_predecessors; new_ef = new_es + float(current_task_data.get('duration_hours', 0))
            if abs(new_es - current_task_data.get('early_start',0.0)) > 0.001 or abs(new_ef - current_task_data.get('early_finish',0.0)) > 0.001 : 
                current_task_data['early_start'] = new_es; current_task_data['early_finish'] = new_ef; updated_in_pass = True
        if not updated_in_pass and _ > 0: break 
    project_duration_hours = max((task.get('early_finish', 0.0) for task in task_dict.values()), default=0.0)
    for task_id_val in task_dict: 
        if not successors_map.get(task_id_val): task_dict[task_id_val]['late_finish'] = project_duration_hours
        else: task_dict[task_id_val]['late_finish'] = float('inf') 
    for _ in range(len(task_dict) + 2):
        updated_in_pass = False
        for task_id_val in reversed(processing_order):
            if task_id_val not in task_dict: continue
            current_task_data = task_dict[task_id_val]; min_ls_of_successors = project_duration_hours 
            if task_id_val in successors_map and successors_map[task_id_val]:
                current_min_ls = float('inf'); has_valid_successor = False
                for succ_id in successors_map[task_id_val]:
                    if succ_id in task_dict: current_min_ls = min(current_min_ls, task_dict[succ_id].get('late_start', float('inf'))); has_valid_successor = True
                if has_valid_successor: min_ls_of_successors = current_min_ls
            new_lf = min_ls_of_successors; new_ls = new_lf - float(current_task_data.get('duration_hours', 0))
            if abs(new_lf - current_task_data.get('late_finish', float('inf'))) > 0.001 or abs(new_ls - current_task_data.get('late_start', float('inf'))) > 0.001: 
                current_task_data['late_finish'] = new_lf; current_task_data['late_start'] = new_ls; updated_in_pass = True
        if not updated_in_pass and _ > 0: break
    cpm_cols = ['early_start', 'early_finish', 'late_start', 'late_finish']
    for col in cpm_cols: tasks_df[col] = tasks_df['task_id'].map(lambda tid: task_dict.get(str(tid), {}).get(col, np.nan))
    tasks_df['total_float_hr_cnt'] = tasks_df['late_finish'] - tasks_df['early_finish']
    tasks_df['total_float_hr_cnt'] = tasks_df['total_float_hr_cnt'].fillna(float('inf'))
    min_total_float = tasks_df['total_float_hr_cnt'][np.isfinite(tasks_df['total_float_hr_cnt'])].min() if tasks_df['total_float_hr_cnt'][np.isfinite(tasks_df['total_float_hr_cnt'])].notna().any() else 0.0
    tasks_df['is_critical'] = tasks_df['total_float_hr_cnt'] <= (min_total_float + 0.001) 
    return tasks_df, project_duration_hours

def perform_monte_carlo_analysis(tasks_df_input: pd.DataFrame, taskpred_df_input: pd.DataFrame, n_simulations: int, user_risks: List[Dict[str, Any]], buffer_days: int) -> Dict[str, Any]:
    # ... (same as previously corrected version)
    if tasks_df_input.empty: logger.warning("MC: No task data."); return {"tasks": [], "project_optimistic": 0.0, "project_most_likely": 0.0, "project_pessimistic": 0.0, "task_sensitivities": [], "all_simulated_durations": [] }
    tasks_df = tasks_df_input.copy(); 
    if 'sensitivity' not in tasks_df.columns: tasks_df['sensitivity'] = 0.0 
    taskpred_df = taskpred_df_input.copy() if not taskpred_df_input.empty else pd.DataFrame()
    full_risk_register = user_risks 
    all_simulated_project_durations_days = []
    per_simulation_task_durations_hours = np.zeros((n_simulations, len(tasks_df)))
    for sim_idx in range(n_simulations):
        sim_tasks_df = tasks_df.copy()
        sim_tasks_df['sim_duration_hours'] = sim_tasks_df['duration_hours'] 
        for index, task_row in sim_tasks_df.iterrows():
            if task_row.get('is_milestone', False): sim_tasks_df.loc[index, 'sim_duration_hours'] = 0.0; continue
            current_planned_duration = float(task_row.get('duration_hours', 0.0))
            optimistic_estimate = current_planned_duration * 0.80 
            pessimistic_estimate_base = current_planned_duration * 1.30 
            task_specific_risk_impact_hours = 0.0
            for risk in full_risk_register:
                applies_to_all = risk.get("Affected Task", "").upper() == "ALL_NON_MILESTONE"
                specific_task_match = risk.get("Affected Task") == task_row.get('task_name')
                if ((applies_to_all and not task_row.get('is_milestone', False)) or specific_task_match):
                    if np.random.random() < float(risk.get('Probability', 0.0)): task_specific_risk_impact_hours += float(risk.get('Impact (hours)', 0.0))
            pessimistic_final = pessimistic_estimate_base + task_specific_risk_impact_hours
            mode_estimate = current_planned_duration
            if optimistic_estimate > mode_estimate: optimistic_estimate = mode_estimate
            if mode_estimate > pessimistic_final: pessimistic_final = mode_estimate
            if optimistic_estimate > pessimistic_final : optimistic_estimate = pessimistic_final
            if abs(optimistic_estimate - pessimistic_final) < 0.001: sim_tasks_df.loc[index, 'sim_duration_hours'] = mode_estimate
            else: sim_tasks_df.loc[index, 'sim_duration_hours'] = np.random.triangular(left=optimistic_estimate, mode=mode_estimate, right=pessimistic_final)
            task_positional_index = tasks_df.index.get_loc(index)
            per_simulation_task_durations_hours[sim_idx, task_positional_index] = sim_tasks_df.loc[index, 'sim_duration_hours']
        cpm_sim_input_df = sim_tasks_df.rename(columns={'sim_duration_hours': 'duration_hours'})
        _, simulated_project_duration_hours = calculate_cpm_and_duration(cpm_sim_input_df, taskpred_df)
        all_simulated_project_durations_days.append(simulated_project_duration_hours / 24.0)
    all_simulated_project_durations_days_np = np.array(all_simulated_project_durations_days) + float(buffer_days)
    project_p10_days = np.percentile(all_simulated_project_durations_days_np, 10)
    project_p50_days = np.percentile(all_simulated_project_durations_days_np, 50)
    project_p90_days = np.percentile(all_simulated_project_durations_days_np, 90)
    task_sensitivities_output = []
    if all_simulated_project_durations_days_np.size > 1 and per_simulation_task_durations_hours.shape[1] == len(tasks_df):
        for i in range(len(tasks_df)): 
            task_name = tasks_df['task_name'].iloc[i]
            individual_task_duration_sims = per_simulation_task_durations_hours[:, i] / 24.0
            if np.std(individual_task_duration_sims) > 1e-6 and np.std(all_simulated_project_durations_days_np) > 1e-6 :
                correlation_matrix = np.corrcoef(individual_task_duration_sims, all_simulated_project_durations_days_np)
                sensitivity_value = correlation_matrix[0, 1]
                if np.isnan(sensitivity_value): sensitivity_value = 0.0
            else: sensitivity_value = 0.0
            task_sensitivities_output.append({"task_name": task_name, "sensitivity": round(sensitivity_value, 4)})
    milestone_tasks_output = []
    if 'is_milestone' in tasks_df.columns and tasks_df['is_milestone'].dtype == 'bool':
        for _, task_row in tasks_df[tasks_df['is_milestone']].iterrows():
            milestone_tasks_output.append({
                "task_name": task_row.get('task_name', 'N/A'), "is_milestone": True,
                "target_start_date": task_row['target_start_date'].strftime('%Y-%m-%d') if pd.notna(task_row.get('target_start_date')) else 'N/A',
                "target_end_date": task_row['target_end_date'].strftime('%Y-%m-%d') if pd.notna(task_row.get('target_end_date')) else 'N/A',
            })
    else: logger.warning("'is_milestone' column not found or not boolean in tasks_df for MC milestone output.")
    return {"tasks": milestone_tasks_output, "project_optimistic": project_p10_days, "project_most_likely": project_p50_days, "project_pessimistic": project_p90_days, "task_sensitivities": task_sensitivities_output, "all_simulated_durations": all_simulated_project_durations_days_np.tolist()}

# --- Pydantic Models ---
# (Ensure all these are correctly formatted with Config on new, indented lines)
class UserBase(BaseModel):
    email: EmailStr
    name: str
class UserCreate(UserBase):
    password: str
    is_admin: Optional[bool] = False
class UserInDB(UserBase):
    id: int
    is_admin: bool
    class Config:
        from_attributes = True
class UserLogin(BaseModel):
    email: EmailStr
    password: str
class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None
class TokenResponse(BaseModel):
    access_token: str; token_type: str; user_id: int; name: str; is_admin: bool; email: EmailStr
class PasswordResetRequest(BaseModel):
    email: EmailStr
class PasswordResetVerify(BaseModel):
    email: EmailStr; code: str; new_password: str = Field(..., min_length=8)
class ChangePasswordRequest(BaseModel):
    current_password: str; new_password: str = Field(..., min_length=8)
class UploadHistoryItem(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    file_size: Optional[int] = None
    status: Optional[str] = None
    class Config:
        from_attributes = True

class TaskModel(BaseModel):
    task_id: str; task_name: Optional[str] = None; duration_hours: Optional[float] = None
    is_milestone: Optional[bool] = False; target_start_date: Optional[Any] = None 
    target_end_date: Optional[Any] = None; act_start_date: Optional[Any] = None
    act_end_date: Optional[Any] = None; early_start: Optional[float] = None
    early_finish: Optional[float] = None; late_start: Optional[float] = None
    late_finish: Optional[float] = None; total_float_hr_cnt: Optional[float] = None
    is_critical: Optional[bool] = None; wbs_id: Optional[str] = None
    status_code: Optional[str] = None; clndr_id: Optional[str] = None
    class Config:
        extra = 'ignore'; from_attributes = True

class TaskPredModel(BaseModel):
    task_id: str 
    pred_task_id: str
    pred_type: Optional[str] = None
    lag_hr_cnt: Optional[float] = None
    class Config:
        extra = 'ignore'; from_attributes = True

class TaskRsrcModel(BaseModel): 
    taskrsrc_id: Optional[str] = None 
    task_id: str
    rsrc_id: Optional[str] = None
    target_cost: Optional[float] = None
    target_qty: Optional[float] = None
    target_start_date: Optional[Any] = None 
    target_end_date: Optional[Any] = None
    class Config:
        extra = 'ignore'; from_attributes = True

class UploadXERResponseData(BaseModel): 
    TASK: List[TaskModel] 
    TASKPRED: List[TaskPredModel]
    TASKRSRC: Optional[List[TaskRsrcModel]] = []

class UploadXERResponse(BaseModel):
    message: str
    data: UploadXERResponseData

class ScheduleCheckRequest(BaseModel):
    task_data: List[Dict[str, Any]] 
    taskpred_data: List[Dict[str, Any]]

class ScheduleCheckResponse(BaseModel):
    issues: List[str]
    project_duration: float 
    tasks: List[TaskModel]

class UserRiskItem(BaseModel):
    Risk_ID: str = Field(..., alias="Risk ID") 
    Description: str; Probability: float 
    Impact_hours: float = Field(..., alias="Impact (hours)")
    Affected_Task: str = Field(..., alias="Affected Task")
    class Config:
        populate_by_name = True; from_attributes = True

class MonteCarloRequest(BaseModel):
    task_data: List[Dict[str, Any]]; taskpred_data: List[Dict[str, Any]]
    n_simulations: int; buffer_days: int; user_risks: List[UserRiskItem]

class MilestoneTaskOutput(BaseModel):
    task_name: str; is_milestone: bool; target_start_date: Optional[str]; target_end_date: Optional[str]

class TaskSensitivityItem(BaseModel):
    task_name: str
    sensitivity: float

class MonteCarloResponse(BaseModel):
    tasks: List[MilestoneTaskOutput]; project_optimistic: float; project_most_likely: float
    project_pessimistic: float; task_sensitivities: List[TaskSensitivityItem]
    all_simulated_durations: Optional[List[float]] = None

class CashFlowDataPoint(BaseModel):
    date: str 
    cumulative_cost: float

class CashFlowResponse(BaseModel):
    data: List[CashFlowDataPoint]
    total_project_cost: float

class CashFlowRequestData(BaseModel): 
    tasks: List[TaskModel] 
    task_rsrcs: Optional[List[TaskRsrcModel]] = []

# NEW Pydantic Model for AI Mitigation Request
class AISuggestMitigationRequest(BaseModel):
    risk_description: str
    risk_probability: Optional[float] = None
    risk_impact_hours: Optional[float] = None
    affected_task: Optional[str] = None
    project_context: Optional[str] = None


# --- Database Class & Dependency, JWT Auth, Helper Functions, Auth Endpoints, User Profile Endpoint ---
# (These sections remain unchanged)
class Database: # ...
    def __init__(self): self.pool: Optional[asyncpg.Pool] = None
    async def connect(self):
        if not self.pool:
            try: self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10); logger.info("DB pool created.")
            except Exception as e: logger.error(f"DB pool creation failed: {e}"); raise
    async def disconnect(self):
        if self.pool: await self.pool.close(); self.pool = None; logger.info("DB pool closed.")
db = Database()
@app.on_event("startup")
async def startup_db_client(): await db.connect()
@app.on_event("shutdown")
async def shutdown_db_client(): await db.disconnect()
async def get_db_conn():
    if not db.pool: raise HTTPException(status_code=503, detail="DB service unavailable.")
    async with db.pool.acquire() as conn: yield conn

async def get_current_user(token: str = Depends(HTTPBearer(auto_error=False)), conn: asyncpg.Connection = Depends(get_db_conn)) -> UserInDB:
    if token is None: raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    credentials_exception = HTTPException(status_code=401,detail="Could not validate credentials",headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("email"); user_id: Optional[int] = payload.get("user_id")
        if email is None or user_id is None: raise credentials_exception
    except jwt.ExpiredSignatureError: raise HTTPException(status_code=401, detail="Token has expired", headers={"WWW-Authenticate": "Bearer"})
    except jwt.PyJWTError: raise credentials_exception
    user_record = await conn.fetchrow("SELECT id, email, name, is_admin FROM users WHERE id = $1 AND email = $2", user_id, email)
    if user_record is None: raise credentials_exception
    return UserInDB(**dict(user_record))

async def get_current_active_admin(current_user: UserInDB = Depends(get_current_user)):
    if not current_user.is_admin: raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire_time = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire_time, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
async def send_reset_code_email(email: str, code: str): logger.info(f"Password reset code for {email}: {code}")

@app.post("/api/auth/signup", response_model=UserInDB, status_code=201)
async def signup_endpoint(user_data: UserCreate, conn: asyncpg.Connection = Depends(get_db_conn)):
    logger.debug(f"Signup request for email: {user_data.email}")
    existing_user = await conn.fetchrow("SELECT email FROM users WHERE email = $1", user_data.email)
    if existing_user: raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = pwd_context.hash(user_data.password)
    try:
        new_user_record = await conn.fetchrow("INSERT INTO users (name, email, password, is_admin) VALUES ($1, $2, $3, $4) RETURNING id, name, email, is_admin", user_data.name, user_data.email, hashed_password, user_data.is_admin)
        if not new_user_record: raise HTTPException(status_code=500, detail="Could not create user")
        return UserInDB(**dict(new_user_record))
    except asyncpg.UniqueViolationError: raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e: logger.error(f"Error during signup: {e}"); raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/api/auth/login", response_model=TokenResponse)
async def login_endpoint(form_data: UserLogin, conn: asyncpg.Connection = Depends(get_db_conn)):
    logger.debug(f"Login attempt for email: {form_data.email}")
    user_record = await conn.fetchrow("SELECT * FROM users WHERE email = $1", form_data.email)
    if not user_record or not pwd_context.verify(form_data.password, user_record["password"]): raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    user_id = user_record["id"]; user_name = user_record["name"]; user_is_admin = user_record["is_admin"]; user_email_val = user_record["email"]
    access_token = create_access_token(data={"email": user_email_val, "user_id": user_id})
    return TokenResponse(access_token=access_token, token_type="bearer", user_id=user_id, name=user_name, is_admin=user_is_admin, email=user_email_val)

@app.post("/api/auth/reset-password/request", status_code=200)
async def request_password_reset_endpoint(data: PasswordResetRequest, background_tasks: BackgroundTasks, conn: asyncpg.Connection = Depends(get_db_conn)):
    user = await conn.fetchrow("SELECT id FROM users WHERE email = $1", data.email)
    if not user: logger.info(f"Pwd reset req for non-existent email: {data.email}"); return {"message": "If registered, code sent."}
    reset_code = secrets.token_urlsafe(32); expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    await conn.execute("INSERT INTO reset_codes (email, code, expires_at) VALUES ($1, $2, $3) ON CONFLICT (email) DO UPDATE SET code = EXCLUDED.code, expires_at = EXCLUDED.expires_at", data.email, reset_code, expires_at)
    background_tasks.add_task(send_reset_code_email, data.email, reset_code)
    return {"message": "If registered, code sent."}

@app.post("/api/auth/reset-password/verify", status_code=200)
async def verify_password_reset_endpoint(data: PasswordResetVerify, conn: asyncpg.Connection = Depends(get_db_conn)):
    stored_code_data = await conn.fetchrow("SELECT code, expires_at FROM reset_codes WHERE email = $1", data.email)
    if not stored_code_data or not secrets.compare_digest(stored_code_data["code"], data.code) or datetime.now(timezone.utc) > stored_code_data["expires_at"]:
        await conn.execute("DELETE FROM reset_codes WHERE email = $1", data.email); raise HTTPException(status_code=400, detail="Invalid/expired code.")
    hashed_password = pwd_context.hash(data.new_password)
    await conn.execute("UPDATE users SET password = $1 WHERE email = $2", hashed_password, data.email)
    await conn.execute("DELETE FROM reset_codes WHERE email = $1", data.email)
    return {"message": "Password reset successfully."}

@app.post("/api/user/change-password", status_code=200)
async def user_change_password_endpoint(data: ChangePasswordRequest, current_user: UserInDB = Depends(get_current_user), conn: asyncpg.Connection = Depends(get_db_conn)):
    user_record_with_password = await conn.fetchrow("SELECT password FROM users WHERE id = $1", current_user.id) 
    if not user_record_with_password: raise HTTPException(status_code=404, detail="User not found")
    if not pwd_context.verify(data.current_password, user_record_with_password["password"]): raise HTTPException(status_code=400, detail="Incorrect current password")
    hashed_new_password = pwd_context.hash(data.new_password)
    await conn.execute("UPDATE users SET password = $1 WHERE id = $2", hashed_new_password, current_user.id)
    return {"message": "Password changed successfully"}

@app.post("/api/upload-xer", response_model=UploadXERResponse)
async def upload_xer_file_endpoint(xer_file: UploadFile = File(...), current_user: UserInDB = Depends(get_current_user), conn: asyncpg.Connection = Depends(get_db_conn)):
    logger.info(f"User {current_user.email} (ID: {current_user.id}) uploaded file: {xer_file.filename}")
    status_message = "uploaded"; file_size = None
    try:
        contents = await xer_file.read(); file_size = len(contents); await xer_file.seek(0)
        try: file_content_str = contents.decode('latin-1')
        except UnicodeDecodeError as ude: logger.error(f"UnicodeDecodeError for {xer_file.filename}: {ude}"); raise HTTPException(status_code=400, detail=f"Error decoding file. Ensure 'latin-1' encoding.")
        
        parsed_dataframes = read_xer_file_content(file_content_str)
        if not parsed_dataframes or 'TASK' not in parsed_dataframes or parsed_dataframes['TASK'].empty:
            logger.error(f"XER parsing failed or returned no TASK data for {xer_file.filename}"); status_message = "parsing_failed_no_task_data"
            raise HTTPException(status_code=400, detail="Failed to parse critical TASK data from XER file.")
        status_message = "parsed_successfully"
        logger.info(f"Successfully parsed {len(parsed_dataframes.get('TASK',[]))} tasks from {xer_file.filename}")
        
        response_data_dict = {}
        for name, df in parsed_dataframes.items():
            if not df.empty:
                df_cleaned = df.copy()
                for col in df_cleaned.columns:
                    if df_cleaned[col].dtype == 'datetime64[ns]': df_cleaned[col] = df_cleaned[col].apply(lambda x: x if pd.notna(x) else None)
                    elif np.issubdtype(df_cleaned[col].dtype, np.number): df_cleaned[col] = df_cleaned[col].apply(lambda x: x if np.isfinite(x) else None)
                response_data_dict[name] = df_cleaned.to_dict(orient='records')
            else: response_data_dict[name] = []
        
        final_response_data = UploadXERResponseData(TASK=response_data_dict.get('TASK', []), TASKPRED=response_data_dict.get('TASKPRED', []), TASKRSRC=response_data_dict.get('TASKRSRC', []))
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error processing XER file {xer_file.filename} for user {current_user.id}: {e}", exc_info=True); status_message = "processing_failed_exception"
        if file_size is None: 
            try: await xer_file.seek(0); file_contents_for_size = await xer_file.read(); file_size = len(file_contents_for_size)
            except: pass 
        await conn.execute("INSERT INTO upload_history (user_id, filename, file_size, status) VALUES ($1, $2, $3, $4)", current_user.id, xer_file.filename, file_size, status_message)
        raise HTTPException(status_code=500, detail=f"Internal server error processing XER file.")
    finally: await xer_file.close()
    await conn.execute("INSERT INTO upload_history (user_id, filename, file_size, status) VALUES ($1, $2, $3, $4)", current_user.id, xer_file.filename, file_size, status_message)
    return UploadXERResponse(message="File parsed successfully", data=final_response_data)

@app.post("/api/schedule-check", response_model=ScheduleCheckResponse)
async def schedule_check_api_endpoint(data: ScheduleCheckRequest, current_user: UserInDB = Depends(get_current_user)):
    logger.info(f"Schedule check by {current_user.email} for {len(data.task_data)} tasks.")
    if not data.task_data: raise HTTPException(status_code=400, detail="No task data provided.")
    try:
        tasks_df = pd.DataFrame(data.task_data); taskpred_df = pd.DataFrame(data.taskpred_data) if data.taskpred_data else pd.DataFrame()
        date_cols = ['target_start_date', 'target_end_date', 'act_start_date', 'act_end_date'] 
        for col in date_cols:
            if col in tasks_df.columns: tasks_df[col] = pd.to_datetime(tasks_df[col], errors='coerce')
            else: tasks_df[col] = pd.NaT
        num_cols = ['duration_hours', 'early_start', 'early_finish', 'late_start', 'late_finish', 'total_float_hr_cnt']
        for col in num_cols:
             if col in tasks_df.columns: tasks_df[col] = pd.to_numeric(tasks_df[col], errors='coerce')
             else: tasks_df[col] = np.nan 
        bool_cols = ['is_milestone', 'is_critical']
        for col in bool_cols:
            if col in tasks_df.columns: tasks_df[col] = tasks_df[col].astype(bool)
            else: tasks_df[col] = False
        if 'task_name' not in tasks_df.columns: tasks_df['task_name'] = "Unknown Task"
        if 'task_id' not in tasks_df.columns: tasks_df['task_id'] = [f"temp_id_{i}" for i in range(len(tasks_df))]
        
        issues = perform_schedule_check(tasks_df.copy(), taskpred_df.copy()) 
        updated_tasks_df_with_cpm, project_duration_hours = calculate_cpm_and_duration(tasks_df.copy(), taskpred_df.copy()) 
        logger.info(f"Schedule check complete. Issues: {len(issues)}, Duration: {project_duration_hours} hrs")
        tasks_for_response = updated_tasks_df_with_cpm.replace({pd.NaT: None, np.nan: None}).to_dict(orient='records')
        return ScheduleCheckResponse(issues=issues, project_duration=project_duration_hours, tasks=tasks_for_response)
    except Exception as e: logger.error(f"Error during schedule check: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Error performing schedule check.")

@app.post("/api/run-analysis", response_model=MonteCarloResponse)
async def run_analysis_api_endpoint(data: MonteCarloRequest, current_user: UserInDB = Depends(get_current_user)):
    logger.info(f"Monte Carlo run by {current_user.email} ({data.n_simulations} sims).")
    if not data.task_data: raise HTTPException(status_code=400, detail="No task data for Monte Carlo.")
    try:
        tasks_df = pd.DataFrame(data.task_data); taskpred_df = pd.DataFrame(data.taskpred_data) if data.taskpred_data else pd.DataFrame()
        date_cols = ['target_start_date', 'target_end_date'] 
        for col in date_cols:
            if col in tasks_df.columns: tasks_df[col] = pd.to_datetime(tasks_df[col], errors='coerce')
            else: tasks_df[col] = pd.NaT
        num_cols = ['duration_hours', 'early_start', 'early_finish', 'late_start', 'late_finish', 'total_float_hr_cnt']
        for col in num_cols:
             if col in tasks_df.columns: tasks_df[col] = pd.to_numeric(tasks_df[col], errors='coerce').fillna(0.0)
             else: tasks_df[col] = 0.0
        bool_cols = ['is_milestone', 'is_critical']
        for col in bool_cols:
            if col in tasks_df.columns: tasks_df[col] = tasks_df[col].astype(bool)
            else: tasks_df[col] = False
        if 'task_name' not in tasks_df.columns: tasks_df['task_name'] = "Unknown Task"
        if 'task_id' not in tasks_df.columns: tasks_df['task_id'] = [f"temp_id_{i}" for i in range(len(tasks_df))]
        
        user_risks_list_of_dicts = [risk.dict(by_alias=True) for risk in data.user_risks]
        analysis_results_dict = perform_monte_carlo_analysis(tasks_df.copy(), taskpred_df.copy(), data.n_simulations, user_risks_list_of_dicts, data.buffer_days)
        return MonteCarloResponse(**analysis_results_dict)
    except Exception as e: logger.error(f"Error during Monte Carlo analysis: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Error performing Monte Carlo analysis.")

@app.post("/api/cash-flow-data", response_model=CashFlowResponse) # Endpoint definition
async def get_cash_flow_data_endpoint(data: CashFlowRequestData, current_user: UserInDB = Depends(get_current_user)):
    logger.info(f"Cash flow data request for user {current_user.email}")
    if not data.tasks: raise HTTPException(status_code=400, detail="Task data is required for cash flow analysis.")
    try:
        tasks_df = pd.DataFrame([task.dict(exclude_none=True, by_alias=True) for task in data.tasks])
        if tasks_df.empty: logger.info("CashFlow: tasks_df is empty after Pydantic conversion."); return CashFlowResponse(data=[], total_project_cost=0.0)
        date_cols = ['target_start_date', 'target_end_date', 'act_start_date', 'act_end_date']
        for col in date_cols:
            if col in tasks_df.columns: tasks_df[col] = pd.to_datetime(tasks_df[col], errors='coerce')
            else: tasks_df[col] = pd.NaT
        tasks_df['effective_start_date'] = tasks_df['target_start_date'].fillna(tasks_df['act_start_date'])
        tasks_df['effective_end_date'] = tasks_df['target_end_date'].fillna(tasks_df['act_end_date'])
        if 'duration_hours' in tasks_df.columns: tasks_df['duration_hours'] = pd.to_numeric(tasks_df['duration_hours'], errors='coerce').fillna(0.0)
        else: tasks_df['duration_hours'] = 0.0
        tasks_df.dropna(subset=['effective_start_date', 'effective_end_date'], inplace=True)
        # Ensure end date is after start date
        if not tasks_df.empty : tasks_df = tasks_df[tasks_df['effective_start_date'] <= tasks_df['effective_end_date']]
        
        if tasks_df.empty: logger.info("CashFlow: tasks_df is empty after date filtering."); return CashFlowResponse(data=[], total_project_cost=0.0)
        task_costs_map = {}; total_project_cost = 0.0
        if data.task_rsrcs:
            rsrc_df = pd.DataFrame([rsrc.dict(exclude_none=True, by_alias=True) for rsrc in data.task_rsrcs])
            if not rsrc_df.empty and 'task_id' in rsrc_df.columns and 'target_cost' in rsrc_df.columns:
                rsrc_df['target_cost'] = pd.to_numeric(rsrc_df['target_cost'], errors='coerce').fillna(0.0)
                rsrc_df['task_id'] = rsrc_df['task_id'].astype(str) 
                task_total_costs_series = rsrc_df.groupby('task_id')['target_cost'].sum()
                task_costs_map = task_total_costs_series.to_dict()
                total_project_cost = float(task_total_costs_series.sum())
                logger.info(f"CashFlow: Calculated total_project_cost from rsrc: {total_project_cost}")
        tasks_df['task_id'] = tasks_df['task_id'].astype(str) 
        tasks_df['total_cost'] = tasks_df['task_id'].map(task_costs_map).fillna(0.0)
        tasks_with_costs_df = tasks_df[tasks_df['total_cost'] > 0].copy()
        logger.info(f"CashFlow: Number of tasks with costs > 0: {len(tasks_with_costs_df)}")
        if tasks_with_costs_df.empty: return CashFlowResponse(data=[], total_project_cost=total_project_cost)
        project_start_date = tasks_with_costs_df['effective_start_date'].min()
        project_end_date = tasks_with_costs_df['effective_end_date'].max()
        if pd.isna(project_start_date) or pd.isna(project_end_date) or project_start_date > project_end_date:
            logger.warning(f"CashFlow: Invalid project timeline. Start: {project_start_date}, End: {project_end_date}"); return CashFlowResponse(data=[], total_project_cost=total_project_cost)
        timeline_end_date = project_end_date + pd.Timedelta(days=1)
        date_range = pd.date_range(start=project_start_date, end=timeline_end_date, freq='D')
        if date_range.empty: logger.warning("CashFlow: Date range for cash flow is empty."); return CashFlowResponse(data=[], total_project_cost=total_project_cost)
        cash_flow_data = []; cumulative_cost = 0.0
        for current_date in date_range:
            daily_cost = 0.0
            for _, task in tasks_with_costs_df.iterrows():
                if pd.notna(task['effective_start_date']) and pd.notna(task['effective_end_date']) and task['effective_start_date'] <= current_date <= task['effective_end_date']:
                    task_duration_days = (task['effective_end_date'] - task['effective_start_date']).days + 1
                    if task_duration_days > 0 and task['total_cost'] > 0 : daily_cost += task['total_cost'] / task_duration_days
            cumulative_cost += daily_cost
            cash_flow_data.append(CashFlowDataPoint(date=current_date.strftime('%Y-%m-%d'), cumulative_cost=round(cumulative_cost, 2)))
        logger.info(f"Generated cash flow data with {len(cash_flow_data)} points. Final cumulative cost: {cumulative_cost}")
        return CashFlowResponse(data=cash_flow_data, total_project_cost=round(total_project_cost,2))
    except Exception as e: logger.error(f"Error in cash flow endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error generating cash flow data.")

# NEW AI Mitigation Endpoint
@app.post("/api/ai/suggest-mitigations")
async def suggest_mitigations_endpoint(
    data: AISuggestMitigationRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in environment variables.")
        raise HTTPException(status_code=500, detail="AI service not configured (API key missing).")
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt_parts = [
            "You are an expert project risk management assistant for engineering and construction projects.",
            "Given the following project risk details, please suggest 3 to 5 concise, distinct, and actionable mitigation strategies. Format them as a simple list.",
            f"Risk Description: {data.risk_description}",
        ]
        if data.risk_probability is not None:
            prompt_parts.append(f"Estimated Probability: {data.risk_probability*100:.0f}%")
        if data.risk_impact_hours is not None:
            prompt_parts.append(f"Estimated Impact: {data.risk_impact_hours} hours of delay")
        if data.affected_task:
            prompt_parts.append(f"Potentially Affected Task: {data.affected_task}")
        if data.project_context:
            prompt_parts.append(f"Project Context: {data.project_context}")
        
        prompt_parts.append("\nSuggested Mitigation Strategies:")
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"Sending prompt to Gemini for risk mitigation: '{prompt[:200]}...'") # Log snippet
        response = await model.generate_content_async(prompt)
        
        suggestions = []
        if response.candidates and response.candidates[0].content.parts:
            text_response = response.text
            logger.info(f"Gemini raw response for mitigations: {text_response}")
            # Attempt to parse numbered or bulleted list items
            suggestions = [s.strip().lstrip('-*0123456789.').strip() for s in text_response.splitlines() if s.strip()]
            suggestions = [s for s in suggestions if s] # Remove empty strings after parsing
        
        logger.info(f"Parsed suggestions: {suggestions}")
        return {"suggestions": suggestions}

    except Exception as e:
        logger.error(f"Error calling Gemini API for mitigations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get AI suggestions: {str(e)}")

# --- File Upload History Endpoint ---
@app.get("/api/user/upload-history", response_model=List[UploadHistoryItem])
async def get_upload_history_for_user_endpoint(current_user: UserInDB = Depends(get_current_user), conn: asyncpg.Connection = Depends(get_db_conn)):
    history_records = await conn.fetch("SELECT id, filename, uploaded_at, file_size, status FROM upload_history WHERE user_id = $1 ORDER BY uploaded_at DESC", current_user.id)
    processed_history = []
    for record_dict in map(dict, history_records):
        if record_dict.get('file_size') is not None:
            try: record_dict['file_size'] = int(record_dict['file_size'])
            except (ValueError, TypeError): logger.warning(f"Could not convert file_size '{record_dict['file_size']}' to int for history ID {record_dict['id']}. Setting to None."); record_dict['file_size'] = None
        processed_history.append(UploadHistoryItem(**record_dict))
    return processed_history

# --- Admin Endpoints ---
@app.get("/api/admin/users", response_model=List[UserInDB])
async def get_all_users_endpoint(admin_user: UserInDB = Depends(get_current_active_admin), conn: asyncpg.Connection = Depends(get_db_conn)):
    user_records = await conn.fetch("SELECT id, name, email, is_admin FROM users ORDER BY id ASC")
    return [UserInDB(**dict(user)) for user in user_records]

@app.delete("/api/admin/users/{user_id_to_delete}", status_code=200)
async def delete_user_by_admin_endpoint(user_id_to_delete: int, admin_user: UserInDB = Depends(get_current_active_admin), conn: asyncpg.Connection = Depends(get_db_conn)):
    if user_id_to_delete == admin_user.id: raise HTTPException(status_code=400, detail="Admin cannot delete self.")
    target_user = await conn.fetchrow("SELECT id FROM users WHERE id = $1", user_id_to_delete)
    if not target_user: raise HTTPException(status_code=404, detail="User to delete not found.")
    await conn.execute("DELETE FROM users WHERE id = $1", user_id_to_delete)
    return {"message": "User deleted successfully."}

@app.put("/api/admin/users/{user_id_to_toggle}/toggle-admin", response_model=UserInDB)
async def toggle_admin_status_endpoint(user_id_to_toggle: int, admin_user: UserInDB = Depends(get_current_active_admin), conn: asyncpg.Connection = Depends(get_db_conn)):
    if user_id_to_toggle == admin_user.id: raise HTTPException(status_code=400, detail="Admin cannot toggle own admin status.")
    target_user_record = await conn.fetchrow("SELECT id, name, email, is_admin FROM users WHERE id = $1", user_id_to_toggle)
    if not target_user_record: raise HTTPException(status_code=404, detail="User not found.")
    new_status = not target_user_record["is_admin"]
    updated_user_record = await conn.fetchrow("UPDATE users SET is_admin = $1 WHERE id = $2 RETURNING id, name, email, is_admin", new_status, user_id_to_toggle)
    if not updated_user_record: raise HTTPException(status_code=500, detail="Failed to update user admin status.")
    return UserInDB(**dict(updated_user_record))

@app.get("/api/admin/check", status_code=200)
async def check_admin_status_for_current_user_endpoint(current_user: UserInDB = Depends(get_current_user)):
    return {"is_admin": current_user.is_admin}

# --- Root endpoint ---
@app.get("/")
async def root(): return {"message": "JESA Risk Analysis API is running!"}