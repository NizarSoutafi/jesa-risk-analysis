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

# --- NEW IMPORTS FOR ANALYSIS LOGIC ---
import pandas as pd
import numpy as np
# --- END NEW IMPORTS ---

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI(title="JESA Risk Analysis API", version="0.1.0")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mac@localhost:5432/jesa_risk_analysis")

# --- JWT Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-and-strong-key-please-change") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Analysis Logic (from your provided script) ---

def read_xer_file_content(file_content_str: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, List[Dict[str, Any]]] = {}
    current_table_name: Optional[str] = None
    columns: Optional[List[str]] = None

    lines = file_content_str.splitlines()
    for line_number, line_text in enumerate(lines):
        line_text = line_text.strip()
        if not line_text:
            continue
        try:
            if line_text.startswith('%T'):
                parts = line_text.split('\t')
                if len(parts) > 1:
                    current_table_name = parts[1].strip()
                    tables[current_table_name] = []
                    columns = None 
                else:
                    logger.warning(f"Line {line_number}: Malformed %T line: {line_text}")
                    current_table_name = None
                continue
            if current_table_name and line_text.startswith('%F'):
                columns = [col.strip() for col in line_text.split('\t')[1:]]
                if not columns:
                    logger.warning(f"Line {line_number}: No columns found for table {current_table_name} in %F line: {line_text}")
                    current_table_name = None
                continue
            if current_table_name and columns and (line_text.startswith('%R') or not line_text.startswith('%')):
                data_parts = line_text.split('\t')
                data_values = data_parts[1:] if line_text.startswith('%R') else data_parts
                if len(data_values) < len(columns):
                    data_values.extend([None] * (len(columns) - len(data_values)))
                elif len(data_values) > len(columns):
                    data_values = data_values[:len(columns)]
                tables[current_table_name].append(dict(zip(columns, data_values)))
        except Exception as e:
            logger.error(f"Error processing line {line_number} ('{line_text}'): {e}")
            continue

    dataframes: Dict[str, pd.DataFrame] = {}
    for table, rows in tables.items():
        if rows: dataframes[table] = pd.DataFrame(rows)
        else: logger.warning(f"Table '{table}' had no rows after parsing.")

    if 'TASK' in dataframes and not dataframes['TASK'].empty:
        df_task = dataframes['TASK']
        date_cols = ['act_start_date', 'act_end_date', 'target_start_date', 'target_end_date',
                     'early_start_date', 'early_end_date', 'late_start_date', 'late_end_date',
                     'create_date', 'update_date']
        for col in date_cols:
            if col in df_task.columns:
                # REMOVED infer_datetime_format=True
                df_task[col] = pd.to_datetime(df_task[col], errors='coerce') 
        
        df_task['target_drtn_hr_cnt'] = pd.to_numeric(df_task['target_drtn_hr_cnt'], errors='coerce').fillna(8.0)
        task_type_series = df_task.get('task_type', pd.Series(dtype=str))
        df_task['is_milestone'] = (task_type_series == 'TT_Mile') | \
                                  (task_type_series == 'TT_FinMile') | \
                                  (df_task['target_drtn_hr_cnt'] == 0)
        logger.info(f"Number of milestones identified: {df_task['is_milestone'].sum()}")

        # Initialize duration_hours as float
        df_task['duration_hours'] = np.nan 
        
        if 'act_start_date' in df_task.columns and 'act_end_date' in df_task.columns:
            valid_actual_dates = df_task['act_start_date'].notna() & df_task['act_end_date'].notna()
            # Ensure subtraction results in Timedelta before getting total_seconds
            time_diff_actual = df_task.loc[valid_actual_dates, 'act_end_date'] - df_task.loc[valid_actual_dates, 'act_start_date']
            df_task.loc[valid_actual_dates, 'duration_hours'] = time_diff_actual.dt.total_seconds() / 3600
        
        needs_target_fallback = df_task['duration_hours'].isna().sum() > 0
        if needs_target_fallback and 'target_start_date' in df_task.columns and 'target_end_date' in df_task.columns:
            valid_target_dates = df_task['target_start_date'].notna() & df_task['target_end_date'].notna() & df_task['duration_hours'].isna()
            time_diff_target = df_task.loc[valid_target_dates, 'target_end_date'] - df_task.loc[valid_target_dates, 'target_start_date']
            df_task.loc[valid_target_dates, 'duration_hours'] = time_diff_target.dt.total_seconds() / 3600

        df_task['duration_hours'] = df_task['duration_hours'].fillna(df_task['target_drtn_hr_cnt'])
        df_task['duration_hours'] = np.where(df_task['is_milestone'], 0.0, df_task['duration_hours']) # Milestones have 0.0 duration
        df_task['duration_hours'] = df_task['duration_hours'].clip(lower=0.0)
        dataframes['TASK'] = df_task
    else:
        logger.warning("TASK table not found or is empty after parsing.")

    # Before returning, log dtypes to help debug Pydantic serialization errors
    for name, df_content in dataframes.items():
        logger.debug(f"DataFrame '{name}' dtypes:\n{df_content.dtypes}")
        # Example: Ensure problematic columns are of correct type for Pydantic models
        # if name == 'TASK' and 'some_int_field' in df_content.columns:
        #    dataframes[name]['some_int_field'] = df_content['some_int_field'].fillna(0).astype(int)

    return dataframes

# ... (Rest of your analysis functions: perform_schedule_check, calculate_cpm_and_duration, perform_monte_carlo_analysis) ...
# ... (These functions from your script should be here) ...

# Schedule check (Copied from your script, ensure it aligns with DataFrame structures)
def perform_schedule_check(tasks_df: pd.DataFrame, taskpred_df: pd.DataFrame) -> List[str]:
    issues = []
    if tasks_df.empty: issues.append("No task data available for schedule check."); return issues
    if 'task_id' not in tasks_df.columns: issues.append("TASK data is missing 'task_id' column."); return issues
    if not taskpred_df.empty and 'pred_task_id' in taskpred_df.columns and 'task_id' in taskpred_df.columns:
        valid_task_ids = set(tasks_df['task_id'].astype(str))
        taskpred_df['pred_task_id_str'] = taskpred_df['pred_task_id'].astype(str)
        missing_preds = taskpred_df[~taskpred_df['pred_task_id_str'].isin(valid_task_ids)]
        if not missing_preds.empty: issues.append(f"{len(missing_preds)} relationships point to predecessor tasks not found (e.g., {missing_preds['pred_task_id_str'].tolist()[:3]}...).")
    if 'duration_hours' in tasks_df.columns and 'is_milestone' in tasks_df.columns:
        invalid_durations = tasks_df[(tasks_df['duration_hours'] <= 0) & (tasks_df['is_milestone'] == False)]
        if not invalid_durations.empty: issues.append(f"{len(invalid_durations)} non-milestone tasks with zero or negative duration (e.g., {invalid_durations['task_name'].tolist()[:3]}...).")
    else: issues.append("Cannot check task durations: 'duration_hours' or 'is_milestone' column missing.")
    if not issues: issues.append("Basic schedule integrity check passed.")
    return issues

# backend/main.py

# ... (other imports and functions) ...

# CPM calculation (Corrected)
def calculate_cpm_and_duration(tasks_df_orig: pd.DataFrame, taskpred_df_orig: pd.DataFrame) -> (pd.DataFrame, float):
    if tasks_df_orig.empty:
        logger.error("CPM: Task data is empty.")
        # Return original df and 0 duration to prevent further errors down the line
        # Or raise an error if this state is unacceptable
        return tasks_df_orig, 0.0
    
    required_task_cols = ['task_id', 'duration_hours']
    for col in required_task_cols:
        if col not in tasks_df_orig.columns:
            logger.error(f"CPM: Task data is missing required column: {col}.")
            # Initialize missing columns to prevent key errors, though results will be incorrect
            if col == 'duration_hours':
                 tasks_df_orig[col] = 0.0 
            else: # task_id is critical, if missing, CPM is not possible
                 return tasks_df_orig, 0.0


    tasks_df = tasks_df_orig.copy()
    taskpred_df = taskpred_df_orig.copy() if not taskpred_df_orig.empty else pd.DataFrame(columns=['task_id', 'pred_task_id'])

    tasks_df['task_id'] = tasks_df['task_id'].astype(str)
    if not taskpred_df.empty:
        taskpred_df['task_id'] = taskpred_df['task_id'].astype(str)
        taskpred_df['pred_task_id'] = taskpred_df['pred_task_id'].astype(str)

    # Convert DataFrame rows to dictionaries for easier manipulation
    task_dict = {str(row['task_id']): row.to_dict() for _, row in tasks_df.iterrows()}
    
    # Initialize ES, EF, LS, LF in the task_dict
    for task_id_val in task_dict:
        task_dict[task_id_val].update({
            'early_start': 0.0,
            'early_finish': 0.0,
            'late_start': float('inf'),
            'late_finish': float('inf')
        })

    predecessors_map = {task_id_val: [] for task_id_val in task_dict}
    successors_map = {task_id_val: [] for task_id_val in task_dict}
    
    if not taskpred_df.empty:
        for _, row in taskpred_df.iterrows():
            # Ensure both task_id and pred_task_id from TASKPRED exist as tasks in task_dict
            if row['pred_task_id'] in task_dict and row['task_id'] in task_dict:
                successors_map[row['pred_task_id']].append(row['task_id'])
                predecessors_map[row['task_id']].append(row['pred_task_id'])

    # Forward Pass - This simplified version assumes tasks are somewhat ordered.
    # For a robust solution, a topological sort of tasks is recommended.
    # Iterating multiple times can help converge for simple non-cyclic graphs.
    
    # Using the task_id list from the original dataframe order
    # If your tasks are not topologically sorted, this CPM might not be fully accurate.
    # For PFE, this iterative approach might be acceptable for simple schedules.
    # Consider a proper graph-based topological sort for complex cases.
    
    # Get a processing order (e.g., task_ids as they appear, or try to find start nodes)
    # This is a simplified pass. For robustness, a graph & topological sort is needed.
    processing_order = list(tasks_df['task_id']) # Simple order for now

    for _ in range(len(task_dict)): # Iterate to allow propagation
        updated_in_pass = False
        for task_id_val in processing_order:
            if task_id_val not in task_dict: continue # Should not happen if processing_order from tasks_df
            
            current_task_data = task_dict[task_id_val]
            max_ef_of_predecessors = 0.0
            if task_id_val in predecessors_map:
                for pred_id in predecessors_map[task_id_val]:
                    if pred_id in task_dict: # Ensure predecessor exists
                        max_ef_of_predecessors = max(max_ef_of_predecessors, task_dict[pred_id]['early_finish'])
            
            new_es = max_ef_of_predecessors
            new_ef = new_es + float(current_task_data.get('duration_hours', 0))

            if new_es != current_task_data['early_start'] or new_ef != current_task_data['early_finish']:
                current_task_data['early_start'] = new_es
                current_task_data['early_finish'] = new_ef
                updated_in_pass = True
        if not updated_in_pass and _ > 0: # Optimization: if a pass makes no changes, values have converged
            break


    project_duration_hours = max(task['early_finish'] for task in task_dict.values()) if task_dict else 0.0
    
    # Initialize late_finish for end nodes
    for task_id_val in task_dict:
        if not successors_map.get(task_id_val): # It's an end node
            task_dict[task_id_val]['late_finish'] = project_duration_hours

    # Backward Pass - Iterative
    for _ in range(len(task_dict)):
        updated_in_pass = False
        for task_id_val in reversed(processing_order):
            if task_id_val not in task_dict: continue

            current_task_data = task_dict[task_id_val]
            min_ls_of_successors = project_duration_hours # Default for tasks with no successors
            
            if task_id_val in successors_map and successors_map[task_id_val]:
                current_min_ls = float('inf')
                has_valid_successor = False
                for succ_id in successors_map[task_id_val]:
                    if succ_id in task_dict: # Ensure successor exists
                        current_min_ls = min(current_min_ls, task_dict[succ_id]['late_start'])
                        has_valid_successor = True
                if has_valid_successor:
                     min_ls_of_successors = current_min_ls
            
            new_lf = min_ls_of_successors
            new_ls = new_lf - float(current_task_data.get('duration_hours', 0))

            if new_lf != current_task_data['late_finish'] or new_ls != current_task_data['late_start']:
                current_task_data['late_finish'] = new_lf
                current_task_data['late_start'] = new_ls
                updated_in_pass = True
        if not updated_in_pass and _ > 0:
            break

    # Update the DataFrame with all calculated CPM values
    # Create new columns if they don't exist
    cpm_cols = ['early_start', 'early_finish', 'late_start', 'late_finish']
    for col in cpm_cols:
        tasks_df[col] = tasks_df['task_id'].map(lambda tid: task_dict.get(tid, {}).get(col, np.nan))

    tasks_df['total_float_hr_cnt'] = tasks_df['late_finish'] - tasks_df['early_finish']
    
    # Handle potential NaNs in total_float_hr_cnt before finding min or comparing
    tasks_df['total_float_hr_cnt'].fillna(float('inf'), inplace=True) 

    min_total_float = tasks_df['total_float_hr_cnt'].min() if not tasks_df['total_float_hr_cnt'].empty else 0
    tasks_df['is_critical'] = tasks_df['total_float_hr_cnt'] <= (min_total_float + 0.001) # Epsilon for float comparison

    return tasks_df, project_duration_hours

# ... (rest of your main.py: perform_monte_carlo_analysis, Pydantic models, endpoints, etc.)

# Monte Carlo analysis (Copied from your script, ensure it aligns)
def perform_monte_carlo_analysis(tasks_df_input: pd.DataFrame, taskpred_df_input: pd.DataFrame, n_simulations: int, user_risks: List[Dict[str, Any]], buffer_days: int) -> Dict[str, Any]:
    if tasks_df_input.empty: logger.warning("MC: No task data."); return {"tasks": [], "project_optimistic": 0, "project_most_likely": 0, "project_pessimistic": 0}
    tasks_df = tasks_df_input.copy(); taskpred_df = taskpred_df_input.copy() if not taskpred_df_input.empty else pd.DataFrame()
    full_risk_register = [
        # {"Risk ID": "R001", "Description": "Default Weather Delay", ...}, # Example default
    ] + user_risks
    all_simulated_project_durations_days = []
    for _ in range(n_simulations):
        sim_tasks_df = tasks_df.copy()
        for index, task_row in sim_tasks_df.iterrows():
            if task_row.get('is_milestone', False): sim_tasks_df.loc[index, 'sim_duration_hours'] = 0; continue
            planned_duration = task_row['duration_hours']; optimistic = planned_duration * 0.80; pessimistic_base = planned_duration * 1.30
            current_task_impact_hours = 0
            for risk in full_risk_register:
                applies_to_all = risk.get("Affected Task", "").upper() == "ALL_NON_MILESTONE"
                specific_task_match = risk.get("Affected Task") == task_row['task_name']
                if ((applies_to_all and not task_row.get('is_milestone', False)) or specific_task_match) and np.random.random() < risk.get('Probability', 0):
                    current_task_impact_hours += risk.get('Impact (hours)', 0)
            pessimistic_with_risk = pessimistic_base + current_task_impact_hours
            if optimistic > planned_duration: optimistic = planned_duration
            if planned_duration > pessimistic_with_risk: pessimistic_with_risk = planned_duration
            if optimistic >= pessimistic_with_risk : # If op >= pess (or equal), use planned or slightly varied
                sim_tasks_df.loc[index, 'sim_duration_hours'] = planned_duration if optimistic == pessimistic_with_risk else np.random.uniform(pessimistic_with_risk, optimistic) # Handle this edge
            else:
                sim_tasks_df.loc[index, 'sim_duration_hours'] = np.random.triangular(left=optimistic, mode=planned_duration, right=pessimistic_with_risk)
        cpm_input_df = sim_tasks_df.rename(columns={'sim_duration_hours': 'duration_hours'})
        _, simulated_project_duration_hours = calculate_cpm_and_duration(cpm_input_df, taskpred_df)
        all_simulated_project_durations_days.append(simulated_project_duration_hours / 24.0)
    all_simulated_project_durations_days = np.array(all_simulated_project_durations_days) + buffer_days
    p10 = np.percentile(all_simulated_project_durations_days, 10)
    p50 = np.percentile(all_simulated_project_durations_days, 50)
    p90 = np.percentile(all_simulated_project_durations_days, 90)
    milestone_tasks_output = []
    for _, task_row in tasks_df[tasks_df.get('is_milestone', pd.Series(dtype=bool))].iterrows(): # Handle if 'is_milestone' is missing
        milestone_tasks_output.append({
            "task_name": task_row['task_name'], "is_milestone": True,
            "target_start_date": task_row['target_start_date'].strftime('%Y-%m-%d') if pd.notna(task_row.get('target_start_date')) else 'N/A',
            "target_end_date": task_row['target_end_date'].strftime('%Y-%m-%d') if pd.notna(task_row.get('target_end_date')) else 'N/A',
        })
    return {"tasks": milestone_tasks_output, "project_optimistic": p10, "project_most_likely": p50, "project_pessimistic": p90}


# --- Pydantic Models ---
class UserBase(BaseModel):
     email: EmailStr;
     name: str
     
class UserCreate(UserBase):
     password: str;
     is_admin: Optional[bool] = False

class UserInDB(UserBase):
     id: int;
     is_admin: bool;
     class Config: from_attributes = True

class UserLogin(BaseModel):
     email: EmailStr;
     password: str

class TokenData(BaseModel):
     email: Optional[str] = None;
     user_id: Optional[int] = None

class TokenResponse(BaseModel):
     access_token: str;
     token_type: str;
     user_id: int;
     name: str;
     is_admin: bool;
     email: EmailStr

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetVerify(BaseModel):
     email: EmailStr;
     code: str;
     new_password: str = Field(..., min_length=8)

class ChangePasswordRequest(BaseModel):
     current_password: str;
     new_password: str = Field(..., min_length=8)

class UploadHistoryItem(BaseModel):
     id: int;
     filename: str;
     uploaded_at: datetime;
     file_size: Optional[int] = None;
     status: Optional[str] = None;
     class Config: from_attributes = True

class TaskModel(BaseModel):
    task_id: str
    task_name: str
    duration_hours: Optional[float] = None
    is_milestone: Optional[bool] = False
    target_start_date: Optional[Any] = None # Allow str or datetime initially
    target_end_date: Optional[Any] = None
    # Add ALL columns from XER that your analysis/display needs.
    # This is a common source of Pydantic errors if df.to_dict() has more keys.
    # Example:
    # clndr_id: Optional[str] = None # if it's str in df
    # phys_complete_pct: Optional[float] = None # if it's float in df

    # To handle potential NaT from pandas for dates not yet parsed by Pydantic
    # you might need custom validators or ensure they are stringified before Pydantic model creation
    # if the Pydantic field is Optional[str]. Or use Optional[datetime] and ensure conversion.
    
    class Config:
        extra = 'ignore' # Allow extra fields from DataFrame to_dict if not all defined here

class TaskPredModel(BaseModel):
    # Define fields based on your TASKPRED DataFrame columns
    task_id: str 
    pred_task_id: str
    # pred_type: Optional[str] = None
    # lag_hr_cnt: Optional[float] = None
    class Config:
        extra = 'ignore'

class UploadXERResponseData(BaseModel):
    TASK: List[TaskModel] 
    TASKPRED: List[TaskPredModel]

class UploadXERResponse(BaseModel):
    message: str
    data: UploadXERResponseData

class ScheduleCheckRequest(BaseModel):
    task_data: List[Dict[str, Any]] # Keep as Dict for flexibility from frontend
    taskpred_data: List[Dict[str, Any]]

class ScheduleCheckResponse(BaseModel):
    issues: List[str]
    project_duration: float 

class UserRiskItem(BaseModel):
    Risk_ID: str = Field(..., alias="Risk ID")
    Description: str
    Probability: float 
    Impact_hours: float = Field(..., alias="Impact (hours)")
    Affected_Task: str = Field(..., alias="Affected Task")
    class Config: populate_by_name = True; from_attributes = True

class MonteCarloRequest(BaseModel):
    task_data: List[Dict[str, Any]]
    taskpred_data: List[Dict[str, Any]]
    n_simulations: int
    buffer_days: int
    user_risks: List[UserRiskItem]

class MilestoneTaskOutput(BaseModel):
    task_name: str
    is_milestone: bool
    target_start_date: Optional[str]
    target_end_date: Optional[str]

class MonteCarloResponse(BaseModel):
    tasks: List[MilestoneTaskOutput]
    project_optimistic: float 
    project_most_likely: float
    project_pessimistic: float

# --- Database Class & Dependency ---
# ... (Keep Database, startup, shutdown, get_db_conn as before) ...
class Database:
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

# --- JWT Authentication & User Fetching ---
# ... (Keep get_current_user, get_current_active_admin as before) ...
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

# --- Helper Functions ---
# ... (Keep create_access_token, send_reset_code_email as before) ...
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire_time = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire_time, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
async def send_reset_code_email(email: str, code: str): logger.info(f"Password reset code for {email}: {code}")

# --- Authentication Endpoints ---
# ... (Keep /api/auth/signup, /api/auth/login, password reset endpoints as before) ...
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
    user = UserInDB(**dict(user_record))
    access_token = create_access_token(data={"email": user.email, "user_id": user.id})
    return TokenResponse(access_token=access_token, token_type="bearer", user_id=user.id, name=user.name, is_admin=user.is_admin, email=user.email)

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

# --- User Profile Endpoint ---
@app.post("/api/user/change-password", status_code=200)
async def user_change_password_endpoint(data: ChangePasswordRequest, current_user: UserInDB = Depends(get_current_user), conn: asyncpg.Connection = Depends(get_db_conn)):
    user_record_with_password = await conn.fetchrow("SELECT password FROM users WHERE id = $1", current_user.id)
    if not user_record_with_password: raise HTTPException(status_code=404, detail="User not found")
    if not pwd_context.verify(data.current_password, user_record_with_password["password"]): raise HTTPException(status_code=400, detail="Incorrect current password")
    hashed_new_password = pwd_context.hash(data.new_password)
    await conn.execute("UPDATE users SET password = $1 WHERE id = $2", hashed_new_password, current_user.id)
    return {"message": "Password changed successfully"}

# --- MODIFIED/NEW File Upload and Analysis Endpoints ---
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
        
        response_data_dict = {name: df.to_dict(orient='records') for name, df in parsed_dataframes.items() if not df.empty}
        final_response_data = UploadXERResponseData(TASK=response_data_dict.get('TASK', []), TASKPRED=response_data_dict.get('TASKPRED', []))
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error processing XER file {xer_file.filename} for user {current_user.id}: {e}", exc_info=True); status_message = "processing_failed_exception"
        if file_size is None: # Try to get size if read failed early
            try: await xer_file.seek(0); file_contents_for_size = await xer_file.read(); file_size = len(file_contents_for_size)
            except: pass # Ignore if can't get size
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
        for col in ['task_id', 'duration_hours', 'is_milestone', 'task_name']: # Ensure essential cols for robustness
            if col not in tasks_df.columns: tasks_df[col] = None # Add if missing, or handle error
        
        issues = perform_schedule_check(tasks_df, taskpred_df)
        _, project_duration_hours = calculate_cpm_and_duration(tasks_df, taskpred_df)
        logger.info(f"Schedule check complete. Issues: {len(issues)}, Duration: {project_duration_hours} hrs")
        return ScheduleCheckResponse(issues=issues, project_duration=project_duration_hours)
    except Exception as e: logger.error(f"Error during schedule check: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Error performing schedule check.")

@app.post("/api/run-analysis", response_model=MonteCarloResponse)
async def run_analysis_api_endpoint(data: MonteCarloRequest, current_user: UserInDB = Depends(get_current_user)):
    logger.info(f"Monte Carlo run by {current_user.email} ({data.n_simulations} sims).")
    if not data.task_data: raise HTTPException(status_code=400, detail="No task data for Monte Carlo.")
    try:
        tasks_df = pd.DataFrame(data.task_data); taskpred_df = pd.DataFrame(data.taskpred_data) if data.taskpred_data else pd.DataFrame()
        for col in ['task_id', 'duration_hours', 'is_milestone', 'task_name', 'target_start_date', 'target_end_date']: # Ensure essential cols
             if col not in tasks_df.columns: tasks_df[col] = None
        # Ensure date columns are datetime if perform_monte_carlo_analysis expects them
        if 'target_start_date' in tasks_df.columns: tasks_df['target_start_date'] = pd.to_datetime(tasks_df['target_start_date'], errors='coerce')
        if 'target_end_date' in tasks_df.columns: tasks_df['target_end_date'] = pd.to_datetime(tasks_df['target_end_date'], errors='coerce')

        user_risks_list_of_dicts = [risk.dict(by_alias=True) for risk in data.user_risks]
        analysis_results = perform_monte_carlo_analysis(tasks_df, taskpred_df, data.n_simulations, user_risks_list_of_dicts, data.buffer_days)
        return MonteCarloResponse(**analysis_results)
    except Exception as e: logger.error(f"Error during Monte Carlo analysis: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Error performing Monte Carlo analysis.")

# --- File Upload History Endpoint ---
@app.get("/api/user/upload-history", response_model=List[UploadHistoryItem])
async def get_upload_history_for_user_endpoint(current_user: UserInDB = Depends(get_current_user), conn: asyncpg.Connection = Depends(get_db_conn)):
    history_records = await conn.fetch("SELECT id, filename, uploaded_at, file_size, status FROM upload_history WHERE user_id = $1 ORDER BY uploaded_at DESC", current_user.id)
    return [UploadHistoryItem(**dict(record)) for record in history_records]

# --- Admin Endpoints ---
# ... (Keep admin endpoints as before, using get_current_active_admin) ...
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
async def toggle_admin_status_endpoint(user_id_to_toggle: int, admin__user: UserInDB = Depends(get_current_active_admin), conn: asyncpg.Connection = Depends(get_db_conn)):
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