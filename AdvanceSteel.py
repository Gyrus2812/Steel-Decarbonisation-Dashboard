# steel_full_dashboard.py
# --------------------------------------------------------------------
# Steel & Power Decarbonisation Dashboard (FULL)
# - FIX: Empty widget (container) removed from PF Correction and Load Shift tabs.
# --------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# ===== Optional ML =====
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# ===== SVM/SVR (Requires scikit-learn) =====
try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    SVR_AVAILABLE = True
except Exception:
    SVR = None
    StandardScaler = None
    SVR_AVAILABLE = False

# ===== VIF (statsmodels) =====
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
except Exception:
    variance_inflation_factor = None
    sm = None

# ====================================================================
# ===== CORE HELPERS (MOVED TO TOP FOR NAMEERROR FIX) =================
# ====================================================================

def prepare_data(df: pd.DataFrame):
    """Parse timestamp, compute engineered fields, derive PF from kWh & kVArh if needed."""
    # Normalize date column name
    if 'date' not in df.columns:
        for alt in ['Date','DATE','timestamp','Timestamp','time','Time']:
            if alt in df.columns:
                df = df.rename(columns={alt:'date'})
                break

    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # sampling interval (hours)
    if len(df) >= 2:
        interval_h = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()/3600.0
        if interval_h <= 0:
            interval_h = 0.25
    else:
        interval_h = 0.25

    # time features
    df['date_only']   = df['timestamp'].dt.date
    df['hour']        = df['timestamp'].dt.hour
    df['month']       = df['timestamp'].dt.month
    df['weekday_idx'] = df['timestamp'].dt.weekday
    if 'WeekStatus' not in df.columns:
        df['WeekStatus'] = np.where(df['weekday_idx']>=5, 'Weekend', 'Weekday')

    # Standardize expected columns
    rename_map = {}
    if 'Usage_kWh' not in df.columns:
        for alt in ['kWh','kwh','Usage','energy_kWh','Energy_kWh','Energy']:
            if alt in df.columns: rename_map[alt] = 'Usage_kWh'; break
    if 'CO2(tCO2)' not in df.columns:
        for alt in ['co2_t','CO2_t','CO2','tCO2','CO2 (t)']:
            if alt in df.columns: rename_map[alt] = 'CO2(tCO2)'; break
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ['Usage_kWh','CO2(tCO2)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # PF cols (sometimes %)
    for col in ['Lagging_Current_Power_Factor','Leading_Current_Power_Factor']:
        if col not in df.columns: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    pf_raw = np.nanmax(
        np.vstack([df['Lagging_Current_Power_Factor'].values,
                   df['Leading_Current_Power_Factor'].values]),
        axis=0
    )
    pf_from_cols = pf_raw / (100.0 if np.nanmax(pf_raw) > 1.5 else 1.0)
    pf_from_cols = np.clip(pf_from_cols, 0.0, 1.0)

    # kvarh
    df['kVArh_lag']  = pd.to_numeric(df.get('Lagging_Current_Reactive.Power_kVarh', np.nan), errors='coerce')
    df['kVArh_lead'] = pd.to_numeric(df.get('Leading_Current_Reactive_Power_kVarh', np.nan), errors='coerce')
    df['kVArh_total']= df[['kVArh_lag','kVArh_lead']].fillna(0).sum(axis=1)

    # derived power & intensity
    df['kW'] = df['Usage_kWh'] / interval_h
    df['carbon_intensity'] = np.where(df['Usage_kWh']>0, df['CO2(tCO2)']/df['Usage_kWh'], np.nan)

    # ---- Derived PF from energies (robust) ----
    pf_calc = np.where(
        (df['Usage_kWh']>0) & (df['kVArh_total']>0),
        df['Usage_kWh'] / np.sqrt(df['Usage_kWh']**2 + df['kVArh_total']**2),
        np.nan
    )
    use_calc = (np.nanmean(pf_from_cols) > 0.98) and (np.nansum(df['kVArh_total']) > 0)
    df['pf_used'] = np.clip(np.where(use_calc & ~np.isnan(pf_calc), pf_calc, pf_from_cols), 0.01, 0.999)
    df['kVA'] = np.where(df['pf_used']>0, df['kW']/df['pf_used'], np.nan)

    return df, float(interval_h)

def kpis(df: pd.DataFrame, interval_h: float):
    return {
        "rows": len(df),
        "interval_hours": interval_h,
        "period_start": df['timestamp'].min(),
        "period_end": df['timestamp'].max(),
        "total_kWh": df['Usage_kWh'].sum(),
        "total_CO2_t": df['CO2(tCO2)'].sum(),
        "avg_pf": df['pf_used'].replace(0, np.nan).mean(),
        "avg_intensity": df['carbon_intensity'].mean()
    }

def daily_summary(df: pd.DataFrame):
    return df.groupby('date_only').agg(
        kWh=('Usage_kWh','sum'),
        CO2_t=('CO2(tCO2)','sum'),
        avg_pf=('pf_used','mean'),
        intensity_t_per_kWh=('carbon_intensity','mean')
    ).reset_index()

def hourly_profile(df: pd.DataFrame):
    return df.groupby('hour').agg(
        avg_kWh=('Usage_kWh','mean'),
        avg_CO2_t=('CO2(tCO2)','mean'),
        avg_intensity=('carbon_intensity','mean'),
        avg_pf=('pf_used','mean')
    ).reset_index()

def ws_loadtype_summary(df: pd.DataFrame):
    if 'Load_Type' not in df.columns:
        return pd.DataFrame()
    g = df.groupby(['WeekStatus','Load_Type']).agg(
        mean_kWh=('Usage_kWh','mean'),
        mean_CO2_t=('CO2(tCO2)','mean'),
        mean_intensity=('carbon_intensity','mean'),
        mean_pf=('pf_used','mean'),
    ).reset_index()
    g['kvarh_per_kWh_plantwide'] = df['kVArh_total'].sum()/max(df['Usage_kWh'].sum(), 1e-9)
    return g

def correlation_matrix(df: pd.DataFrame):
    cols = ['Usage_kWh','CO2(tCO2)','kVArh_total','pf_used','kVA','carbon_intensity','hour','month','weekday_idx']
    sub = df[cols].copy()
    return sub.corr(numeric_only=True)

def worst_intervals(df: pd.DataFrame, q=0.99, n=200):
    thr = np.nanquantile(df['carbon_intensity'], q)
    return (df[df['carbon_intensity'] >= thr][
        ['timestamp','Usage_kWh','CO2(tCO2)','carbon_intensity','pf_used','Load_Type','WeekStatus']
    ].sort_values('carbon_intensity', ascending=False).head(n))

# ====================== Scenarios (unchanged) ======================
def scenario_pf_correction(df: pd.DataFrame, interval_h: float, target_pf: float = 0.95):
    p = df['kW'].values
    pf = df['pf_used'].values.clip(0.01, 0.999)
    needs = pf < target_pf

    if needs.sum() == 0:
        return pd.DataFrame([{
            "avg_pf_now": float(np.nanmean(df['pf_used'])),
            "avg_pf_target": float(target_pf),
            "reactive_energy_now_kVArh": float(np.nansum(df['kVArh_total'])),
            "reactive_energy_reduction_kVArh": 0.0,
            "relative_line_loss_reduction_%": 0.0
        }])

    theta_now = np.arccos(pf[needs])
    theta_tgt = np.arccos(np.full(np.sum(needs), target_pf))
    tan_now   = np.tan(theta_now)
    tan_tgt   = np.tan(theta_tgt)

    Qc_kvar = (p[needs] * (tan_now - tan_tgt)) * interval_h
    reactive_now_kvarh = float(np.nansum(df['kVArh_total']))

    S_old = p[needs] / pf[needs]
    S_new = p[needs] / target_pf
    num = np.nansum(S_old**2 - S_new**2)
    den = np.nansum(S_old**2)
    rel_drop = float(num/den) if den > 0 else 0.0
    rel_drop = max(rel_drop, 0.0)

    return pd.DataFrame([{
        "avg_pf_now": float(np.nanmean(df['pf_used'])),
        "avg_pf_target": float(target_pf),
        "reactive_energy_now_kVArh": reactive_now_kvarh,
        "reactive_energy_reduction_kVArh": float(np.nansum(Qc_kvar) if Qc_kvar.size else 0.0),
        "relative_line_loss_reduction_%": float(100.0 * rel_drop)
    }])

def scenario_load_shift_energy_weighted(df: pd.DataFrame,
                                        shift_pct: float = 0.10,
                                        dirty_q: float = 0.80,
                                        clean_q: float = 0.20,
                                        cost_per_kWh: float = 8.0):
    d = df[['Usage_kWh','CO2(tCO2)','carbon_intensity']].dropna(subset=['carbon_intensity','Usage_kWh']).copy()
    if d.empty or d['Usage_kWh'].sum() <= 0:
        return pd.DataFrame([{
            "baseline_total_CO2_t": float(df['CO2(tCO2)'].sum()),
            "energy_shifted_kWh": 0.0,
            "dirty_I_weighted": np.nan,
            "clean_I_weighted": np.nan,
            "estimated_CO2_savings_t": 0.0,
            "estimated_CO2_reduction_%": np.nan,
            "money_saved": 0.0,
            "dirty_rows": 0, "clean_rows_used": 0
        }])

    q_dirty = np.nanquantile(d['carbon_intensity'], dirty_q)
    q_clean = np.nanquantile(d['carbon_intensity'], clean_q)
    dirty = d[d['carbon_intensity'] >= q_dirty].copy()
    clean_pool = d[d['carbon_intensity'] <= q_clean].copy()

    de = dirty['Usage_kWh'].sum()
    if de <= 0:
        return pd.DataFrame([{
            "baseline_total_CO2_t": float(df['CO2(tCO2)'].sum()),
            "energy_shifted_kWh": 0.0,
            "dirty_I_weighted": np.nan,
            "clean_I_weighted": np.nan,
            "estimated_CO2_savings_t": 0.0,
            "estimated_CO2_reduction_%": np.nan,
            "money_saved": 0.0,
            "dirty_rows": len(dirty), "clean_rows_used": 0
        }])

    dirty_Iw = float((dirty['carbon_intensity']*dirty['Usage_kWh']).sum() / max(de,1e-9))
    shift_energy = shift_pct * de

    if clean_pool['Usage_kWh'].sum() < shift_energy:
        clean_pool = d.sort_values('carbon_intensity', ascending=True)
    else:
        clean_pool = clean_pool.sort_values('carbon_intensity', ascending=True)

    cum = clean_pool['Usage_kWh'].cumsum().values
    idx = np.searchsorted(cum, shift_energy, side='right')
    if idx == 0:
        used = clean_pool.iloc[[0]].copy()
        frac = shift_energy / max(used['Usage_kWh'].iloc[0], 1e-9)
        used['Usage_kWh'] *= frac
    else:
        used = clean_pool.iloc[:idx].copy()
        excess = used['Usage_kWh'].sum() - shift_energy
        if excess > 0:
            last_idx = used.index[-1]
            used.loc[last_idx, 'Usage_kWh'] -= excess

    ce = used['Usage_kWh'].sum()
    clean_Iw = float((used['carbon_intensity']*used['Usage_kWh']).sum() / max(ce,1e-9))

    baseline_co2 = float(df['CO2(tCO2)'].sum())
    co2_savings  = float(shift_energy * (dirty_Iw - clean_Iw))
    pct          = float(100*co2_savings/baseline_co2) if baseline_co2>0 else np.nan
    money_saved  = shift_energy * cost_per_kWh

    return pd.DataFrame([{
        "baseline_total_CO2_t": baseline_co2,
        "energy_shifted_kWh": float(shift_energy),
        "dirty_I_weighted": dirty_Iw,
        "clean_I_weighted": clean_Iw,
        "estimated_CO2_savings_t": co2_savings,
        "estimated_CO2_reduction_%": pct,
        "money_saved": money_saved,
        "dirty_rows": int(len(dirty)),
        "clean_rows_used": int(len(used))
    }])

# ====================== Gauges (unchanged) ======================
def gauge_pf(title: str, value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0.0, min(1.0, float(value))) if not np.isnan(value) else 0.0,
        number={'valueformat': '.3f'},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'thickness': 0.2},
            'steps': [
                {'range': [0.0, 0.8], 'color': "rgba(255,0,0,0.4)"},
                {'range': [0.8, 0.9], 'color': "rgba(255,165,0,0.4)"},
                {'range': [0.9, 1.0], 'color': "rgba(0,128,0,0.4)"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=0))
    return fig

def gauge_percent(title: str, value_pct: float, max_pct: float = 20.0):
    val = 0.0 if (value_pct is None or np.isnan(value_pct)) else float(value_pct)
    upper = max(5.0, min(max_pct, max(5.0, val*1.5)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix': " %", 'valueformat': '.2f'},
        title={'text': title},
        gauge={
            'axis': {'range': [0, upper]},
            'bar': {'thickness': 0.2},
            'steps': [
                {'range': [0, upper*0.25], 'color': "rgba(255,0,0,0.35)"},
                {'range': [upper*0.25, upper*0.5], 'color': "rgba(255,165,0,0.35)"},
                {'range': [upper*0.5, upper], 'color': "rgba(0,128,0,0.35)"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=0))
    return fig

# ====================== Forecast helpers & metrics (WAPE FIXES IMPLEMENTED HERE) ======================
def _rmse(y_true, y_pred):
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    if yt.size == 0 or yp.size == 0: return np.nan
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def _mape(y_true, y_pred):
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = yt != 0
    if mask.sum()==0: return np.nan
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)

def _smape(y_true, y_pred):
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    mask = denom != 0
    if mask.sum()==0: return np.nan
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100)

def _wape(y_true, y_pred):
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = np.sum(np.abs(yt))
    if denom <= 0: return np.nan
    return float(np.sum(np.abs(yt - yp)) / denom * 100)

def _clean_series_for_metrics(df_eval, target_col, pred_col):
    """Drop rows where target or prediction is missing/invalid (NaN/inf)."""
    de = df_eval.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col, pred_col])
    return de

# --- WAPE FIX 1: Enhanced Feature Engineering (Fourier, Longer Lags/Rolls) ---
def add_lags_rolls(df, target_col, max_lag=28, roll_windows=(3, 7, 14, 21, 28)):
    d = df.copy()

    for L in range(1, max_lag + 1):
        d[f'{target_col}_lag{L}'] = d[target_col].shift(L)

    for w in roll_windows:
        d[f'{target_col}_rmean{w}'] = d[target_col].shift(1).rolling(w).mean()
        d[f'{target_col}_rstd{w}']  = d[target_col].shift(1).rolling(w).std()
        d[f'{target_col}_rmed{w}']  = d[target_col].shift(1).rolling(w).median()

    d['weekday'] = d['ds'].dt.weekday
    d['month'] = d['ds'].dt.month
    d['is_weekend'] = (d['weekday'] >= 5).astype(int)
    d[f'{target_col}_same_wd_last_week'] = d[target_col].shift(7)

    dayofyear = d['ds'].dt.dayofyear
    freq = 365.25
    for k in range(1, 3):
        d[f'sin_y_{k}'] = np.sin(2 * np.pi * k * dayofyear / freq)
        d[f'cos_y_{k}'] = np.cos(2 * np.pi * k * dayofyear / freq)

    return d.dropna().reset_index(drop=True)

# --- WAPE FIX 2: CatBoost Training Function ---
def _fit_cat(train_df, target_col):
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost not installed. Run: pip install catboost")
        
    features = [c for c in train_df.columns if c not in ['ds', target_col]]
    X, y = train_df[features], train_df[target_col]
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5.0,
        loss_function='RMSE',
        random_seed=42,
        subsample=0.8,                
        rsm=0.9, 
        verbose=False
    )
    model.fit(X, y) 
    return model, features, None # CatBoost doesn't require a separate scaler

# --- NEW: SVR Training Function (uses RBF kernel) ---
def _fit_svr(train_df, target_col):
    if not SVR_AVAILABLE:
        raise RuntimeError("SVR requires scikit-learn. Run: pip install scikit-learn")

    features = [c for c in train_df.columns if c not in ['ds', target_col]]
    X_train_raw, y_train_raw = train_df[features], train_df[target_col]

    # Scaling is CRITICAL for SVR (features and target)
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    y_train_scaled = Y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()

    # SVR Model (RBF kernel is default)
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    model.fit(X_train_scaled, y_train_scaled)

    # Returning tuple of (X_scaler, Y_scaler)
    return model, features, (X_scaler, Y_scaler)


def walkforward_eval(hist_df, target_col, horizon, model_choice):
    """Evaluate on the last `horizon` days with forward-only predictions."""
    hist = hist_df[['ds', target_col]].dropna().copy()
    if len(hist) <= horizon + 30: 
        return None, None
    cutoff = len(hist) - horizon
    train_df, test_df = hist.iloc[:cutoff], hist.iloc[cutoff:]

    feat_train = add_lags_rolls(train_df, target_col)
    if len(feat_train) == 0:
        return None, None

    # Fit Model
    if model_choice == 'CatBoost':
        model, features, _ = _fit_cat(feat_train, target_col)
        X_scaler, Y_scaler = None, None
    elif model_choice == 'SVR':
        model, features, scalers = _fit_svr(feat_train, target_col)
        X_scaler, Y_scaler = scalers
    else:
        return None, None

    # Setup for step-ahead prediction
    preds_eval = []
    cur_feat = feat_train.copy()
    last_date = cur_feat['ds'].max()
    LAG_RANGE = range(1, 29) 
    ROLL_WINDOWS = (3, 7, 14, 21, 28)

    for step in range(1, horizon+1):
        day = pd.to_datetime(last_date) + timedelta(days=step)
        
        # Build the features for the new day
        row = {'ds': day, 
               'weekday': day.weekday(), 
               'month': day.month, 
               'is_weekend': 1 if day.weekday()>=5 else 0,
               f'{target_col}_same_wd_last_week': cur_feat[target_col].iloc[-7]
              }
        
        dayofyear = day.dayofyear
        freq = 365.25
        for k in range(1, 3):
            row[f'sin_y_{k}'] = np.sin(2 * np.pi * k * dayofyear / freq)
            row[f'cos_y_{k}'] = np.cos(2 * np.pi * k * dayofyear / freq)
        for L in LAG_RANGE:
            row[f'{target_col}_lag{L}'] = cur_feat[target_col].iloc[-L]
        for w in ROLL_WINDOWS:
            row[f'{target_col}_rmean{w}'] = cur_feat[target_col].iloc[-w:].mean()
            row[f'{target_col}_rstd{w}']  = cur_feat[target_col].iloc[-w:].std()
            row[f'{target_col}_rmed{w}']  = cur_feat[target_col].iloc[-w:].median()
        
        Xnew = pd.DataFrame([row])[features]
        
        # --- Model-specific Prediction Logic ---
        if model_choice == 'SVR':
            Xnew_scaled = X_scaler.transform(Xnew)
            yhat_scaled = model.predict(Xnew_scaled)[0]
            # Inverse transform the prediction
            yhat = float(Y_scaler.inverse_transform(np.array(yhat_scaled).reshape(-1, 1))[0][0])
        else: # CatBoost
            yhat = float(model.predict(Xnew)[0])
        
        row[target_col]=yhat
        cur_feat = pd.concat([cur_feat,pd.DataFrame([row])],ignore_index=True)
        preds_eval.append({'ds':day,'yhat':yhat})
    
    eval_pred_df = pd.DataFrame(preds_eval)
    eval_df = test_df.merge(eval_pred_df, on='ds', how='left')

    # Future Forecast (using the same model/features/scaler as the evaluation run)
    feat_full = add_lags_rolls(hist, target_col)
    if len(feat_full) == 0:
        return eval_df, None

    # Refit model on FULL history for future forecast
    if model_choice == 'CatBoost':
        model_full, features_full, _ = _fit_cat(feat_full, target_col)
        X_scaler_full, Y_scaler_full = None, None
    else: # SVR
        model_full, features_full, scalers_full = _fit_svr(feat_full, target_col)
        X_scaler_full, Y_scaler_full = scalers_full


    preds_future = []
    cur_feat2 = feat_full.copy()
    last_date2 = cur_feat2['ds'].max()
    
    for step in range(1, horizon+1):
        day = pd.to_datetime(last_date2) + timedelta(days=step)
        row = {'ds': day, 'weekday': day.weekday(), 'month': day.month, 'is_weekend': 1 if day.weekday()>=5 else 0}
        
        dayofyear = day.dayofyear
        freq = 365.25
        for k in range(1, 3):
            row[f'sin_y_{k}'] = np.sin(2 * np.pi * k * dayofyear / freq)
            row[f'cos_y_{k}'] = np.cos(2 * np.pi * k * dayofyear / freq)
        
        for L in LAG_RANGE:
            row[f'{target_col}_lag{L}'] = cur_feat2[target_col].iloc[-L]
        for w in ROLL_WINDOWS:
            row[f'{target_col}_rmean{w}'] = cur_feat2[target_col].iloc[-w:].mean()
            row[f'{target_col}_rstd{w}']  = cur_feat2[target_col].iloc[-w:].std()
            row[f'{target_col}_rmed{w}']  = cur_feat2[target_col].iloc[-w:].median()
        row[f'{target_col}_same_wd_last_week'] = cur_feat2[target_col].iloc[-7]
        
        Xnew = pd.DataFrame([row])[features_full]
        
        # --- Model-specific Prediction Logic ---
        if model_choice == 'SVR':
            Xnew_scaled = X_scaler_full.transform(Xnew)
            yhat_scaled = model_full.predict(Xnew_scaled)[0]
            yhat = float(Y_scaler_full.inverse_transform(np.array(yhat_scaled).reshape(-1, 1))[0][0])
        else: # CatBoost
            yhat = float(model_full.predict(Xnew)[0])
            
        row[target_col]=yhat
        cur_feat2 = pd.concat([cur_feat2,pd.DataFrame([row])],ignore_index=True)
        preds_future.append({'ds':day,'yhat':yhat})

    future_df = pd.DataFrame(preds_future)
    return eval_df, future_df

# ---------- Backtest (Hold-out) with Model Selection ----------
def backtest_holdout(hist_df, target_col, back_days, model_choice):
    """
    Robust hold-out backtest with model selection.
    """
    hist = hist_df[['ds', target_col]].dropna().copy()
    if len(hist) <= back_days + 30: 
        return None

    feat_all = add_lags_rolls(hist, target_col)
    if len(feat_all) == 0:
        return None

    cutoff_date = hist['ds'].iloc[-back_days]
    train = feat_all[feat_all['ds'] < cutoff_date].copy()
    test  = feat_all[feat_all['ds'] >= cutoff_date].copy()

    if len(train) < 25 or len(test) == 0:
        return None

    # Fit Model
    if model_choice == 'CatBoost':
        model, features, _ = _fit_cat(train, target_col)
        X_scaler, Y_scaler = None, None
    elif model_choice == 'SVR':
        model, features, scalers = _fit_svr(train, target_col)
        X_scaler, Y_scaler = scalers
    else:
        return None

    X_test = test[features]

    # --- Model-specific Prediction Logic ---
    if model_choice == 'SVR':
        X_test_scaled = X_scaler.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        # Inverse transform the predictions
        yhat = Y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    else: # CatBoost
        yhat = model.predict(X_test)

    pred_df = pd.DataFrame({'ds': test['ds'].values, 'yhat': yhat})
    truth = hist.merge(pred_df, on='ds', how='right')
    return truth

# ====================== EDA Visual helpers (unchanged) ======================
def infer_column_types(df: pd.DataFrame):
    rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    discrete = []
    continuous = []
    categorical = non_numeric_cols.copy()

    for c in numeric_cols:
        nunique = df[c].nunique(dropna=True)
        if c in ['hour','month','weekday_idx']:
            discrete.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and nunique <= min(20, max(10, int(rows*0.05))):
            discrete.append(c)
        else:
            continuous.append(c)

    return categorical, discrete, continuous

def plot_density_hist(df, col):
    fig = px.histogram(df, x=col, nbins=50, histnorm='probability density', opacity=0.85)
    fig.update_layout(title=f"Density (normalized histogram) — {col}", xaxis_title=col, yaxis_title="Density")
    return fig

def plot_box(df, col):
    fig = px.box(df, y=col, points=False)
    fig.update_layout(title=f"Box plot — {col}", yaxis_title=col)
    return fig

def fmt_num(x, unit=""):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "—"
    return f"{x:,.2f}{unit}"


# ====================================================================
# ===== STREAMLIT APP EXECUTION START ================================
# ====================================================================
st.set_page_config(page_title="Steel Decarbonisation Dashboard", layout="wide")

# --- Light UI styling ---
st.markdown("""
<style>
.smallcaps { letter-spacing: .04em; text-transform: uppercase; opacity:.8; font-size:0.8rem; }
.card { background: rgba(255,255,255,0.03); padding: 16px 18px; border:1px solid rgba(250,250,250,0.08);
        border-radius: 14px; }
.kpi { background: rgba(255,255,255,0.05); padding: 12px 14px; border-radius: 12px; margin-bottom: 8px; }
.kpi h3 { margin: 0 0 6px 0; font-size: .9rem; opacity:.85; }
.kpi p  { margin: 0; font-size: 1.2rem; font-weight: 600; }
.section-title { font-weight: 700; font-size: 1.05rem; margin: 6px 0 10px; }
hr.sep { border: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.2), transparent); }
.explain { font-size:.95rem; opacity:.9; }
.insight { font-size:.95rem; }
</style>
""", unsafe_allow_html=True)


st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload Steel_industry_data.csv", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    df, interval_h = prepare_data(df)

    st.title("⚙️ Steel & Power Decarbonisation Dashboard")

    # ------------------ KPIs ------------------
    st.subheader("Key Indicators")
    k = kpis(df, interval_h)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown('<div class="kpi"><h3>Rows</h3><p>{:,}</p></div>'.format(k['rows']), unsafe_allow_html=True)
    c2.markdown('<div class="kpi"><h3>Interval (h)</h3><p>{:.2f}</p></div>'.format(k['interval_hours']), unsafe_allow_html=True)
    c3.markdown('<div class="kpi"><h3>Total kWh</h3><p>{:,.0f}</p></div>'.format(k['total_kWh']), unsafe_allow_html=True)
    c4.markdown('<div class="kpi"><h3>Total CO₂ (t)</h3><p>{:.3f}</p></div>'.format(k['total_CO2_t']), unsafe_allow_html=True)
    c5.markdown('<div class="kpi"><h3>Avg PF (used)</h3><p>{:.3f}</p></div>'.format(k['avg_pf']), unsafe_allow_html=True)
    c6.markdown('<div class="kpi"><h3>Avg CI (tCO₂/kWh)</h3><p>{:.6f}</p></div>'.format(k['avg_intensity']), unsafe_allow_html=True)
    st.markdown('<p class="explain">**What this means:** kWh/CO₂ give your total scale; average PF shows electrical efficiency; carbon intensity (CI) is emissions per kWh (lower is better).</p>', unsafe_allow_html=True)

    pf_target_hint = min(0.99, max(0.95, (k['avg_pf'] if not np.isnan(k['avg_pf']) else 0.93) + 0.02))
    st.markdown(f'<p class="insight">**Inference:** Average PF ≈ {k["avg_pf"]:.3f}. A practical target is **{pf_target_hint:.2f}–0.99** using capacitor banks.</p>', unsafe_allow_html=True)

    # ------------------ Daily Trends ------------------
    st.subheader("Daily Trends (kWh, CO₂, CI) + interpretation")
    daily = daily_summary(df)
    st.plotly_chart(px.line(daily, x='date_only', y='kWh', title="Daily Electricity Consumption (kWh)"), use_container_width=True)
    st.plotly_chart(px.line(daily, x='date_only', y='CO2_t', title="Daily CO₂ Emissions (t)"), use_container_width=True)
    st.plotly_chart(px.line(daily, x='date_only', y='intensity_t_per_kWh', title="Daily Carbon Intensity (tCO₂/kWh)"), use_container_width=True)

    if len(daily) > 5:
        peak_day = daily.loc[daily['kWh'].idxmax(), 'date_only']
        min_ci_day = daily.loc[daily['intensity_t_per_kWh'].idxmin(), 'date_only']
        st.markdown(f'<p class="insight">**Inference:** Peak daily energy on **{peak_day}**; lowest daily CI on **{min_ci_day}**. Schedule non-critical loads near low-CI days if possible.</p>', unsafe_allow_html=True)

    # ------------------ Hourly Profiles ------------------
    st.subheader("Average Hourly Profiles + interpretation")
    hourly = hourly_profile(df)
    st.plotly_chart(px.line(hourly, x='hour', y='avg_kWh', title="Average kWh by Hour"), use_container_width=True)
    st.plotly_chart(px.line(hourly, x='hour', y='avg_intensity', title="Average Carbon Intensity by Hour"), use_container_width=True)
    if len(hourly) > 0:
        h_peak = int(hourly.loc[hourly['avg_kWh'].idxmax(),'hour'])
        h_clean = int(hourly.loc[hourly['avg_intensity'].idxmin(),'hour'])
        st.markdown(f'<p class="insight">**Inference:** Peak hourly demand ≈ **{h_peak}:00**; cleanest average CI around **{h_clean}:00**. Shifting energy from peak/dirty hours to cleaner windows reduces both cost and CO₂.</p>', unsafe_allow_html=True)

    # ------------------ WeekStatus × Load-Type ------------------
    st.subheader("WeekStatus × Load-Type Benchmark + interpretation")
    wslt = ws_loadtype_summary(df)
    if wslt.empty:
        st.info("Column 'Load_Type' not found; skipping WS×Load-Type view.")
    else:
        st.dataframe(wslt, use_container_width=True)
        st.markdown('<p class="insight">**Inference:** Compare mean PF/CI across categories; low-PF/high-CI categories are improvement targets (tuning, scheduling, maintenance).</p>', unsafe_allow_html=True)

    # ------------------ Correlation Matrix ------------------
    st.subheader("Correlation Matrix + interpretation")
    corr = correlation_matrix(df)
    st.dataframe(corr.style.background_gradient(cmap="Blues"), use_container_width=True)
    c_abs = corr.abs().unstack().sort_values(ascending=False)
    top_pairs = [(i,j,v) for (i,j),v in c_abs.items() if i<j and v>=0.7][:5]
    if top_pairs:
        bullets = "<br>".join([f"• **{i}–{j}**: {v:.2f}" for i,j,v in top_pairs])
        st.markdown(f'<p class="insight">**Strong relationships (|r|≥0.70):**<br>{bullets}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="insight">**Inference:** No very strong linear relationships (|r|≥0.70) beyond expected (kWh–CO₂).</p>', unsafe_allow_html=True)

    # ------------------ Highest CI ------------------
    st.subheader("Highest Carbon-Intensity Intervals (Top 1%) + interpretation")
    worst = worst_intervals(df)
    st.dataframe(worst, use_container_width=True, height=300)
    st.markdown('<p class="insight">**Inference:** These timestamps are the dirtiest supply windows; moving flexible loads away from them yields the largest CO₂ cuts.</p>', unsafe_allow_html=True)

    # =================== EDA TAB ===================
    st.subheader("Exploratory Data Analysis (EDA)")
    tab_types, tab_univariate, tab_patterns, tab_pf_vif = st.tabs([
        "Column Types & Summary", "Univariate (Density & Box)", "CI & Load Patterns", "PF Multicollinearity (VIF)"
    ])

    with tab_types:
        cat_cols, disc_cols, cont_cols = infer_column_types(df)
        cA, cB, cC = st.columns(3)
        cA.markdown("**Categorical columns**"); cA.dataframe(pd.DataFrame({"categorical": cat_cols}))
        cB.markdown("**Discrete numeric**");   cB.dataframe(pd.DataFrame({"discrete": disc_cols}))
        cC.markdown("**Continuous numeric**"); cC.dataframe(pd.DataFrame({"continuous": cont_cols}))

        num_summary = df[disc_cols + cont_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T
        st.markdown("**Numeric Summary (discrete + continuous)**")
        st.dataframe(num_summary, use_container_width=True)
        st.markdown('<p class="explain">**What to look for:** ranges, skew, and outliers. Long right tails in kWh/CO₂ suggest peak events; PF near 1.0 with low variance is ideal.</p>', unsafe_allow_html=True)

    with tab_univariate:
        options = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not options:
            st.info("No numeric columns found.")
        else:
            col = st.selectbox("Choose a numeric column for distribution analysis", options, index=options.index('Usage_kWh') if 'Usage_kWh' in options else 0)
            left, right = st.columns(2)
            with left:  st.plotly_chart(plot_density_hist(df, col), use_container_width=True)
            with right: st.plotly_chart(plot_box(df, col), use_container_width=True)
            st.markdown('<p class="explain">**Interpretation:** Density shows common values & tails; boxplot spotlights median/IQR/outliers. For PF, look for values &lt;0.9 (correction potential).</p>', unsafe_allow_html=True)

    with tab_patterns:
        ci_hour = df.groupby('hour', dropna=True)['carbon_intensity'].mean().reset_index()
        ci_wd   = df.groupby('weekday_idx', dropna=True)['carbon_intensity'].mean().reset_index()
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(ci_hour, x='hour', y='carbon_intensity', title="Average CI by Hour"), use_container_width=True)
        c2.plotly_chart(px.line(ci_wd, x='weekday_idx', y='carbon_intensity', title="Average CI by Weekday (0=Mon)"), use_container_width=True)
        st.markdown('<p class="explain">**Interpretation:** If CI dips late-night/weekends, shift non-critical operations there. Combine with tariff windows for cost savings.</p>', unsafe_allow_html=True)

    with tab_pf_vif:
        st.markdown("**PF multicollinearity (VIF)** — high VIF (&gt;5) means variables move together; that is expected among kW/kVA/kVArh due to physics.")
        candidates = ['kW','kVArh_total','kVA','carbon_intensity','hour','month','weekday_idx']
        X = df[candidates].copy()
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) >= 8 and variance_inflation_factor is not None:
            for c in list(X.columns):
                if X[c].std(ddof=0) == 0:
                    X.drop(columns=[c], inplace=True)
            X_const = sm.add_constant(X, has_constant='add')
            vif_rows = []
            for i, colname in enumerate(X_const.columns):
                if colname == 'const':  continue
                vif_val = variance_inflation_factor(X_const.values, i)
                vif_rows.append({"feature": colname, "VIF": float(vif_val)})
            vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
            st.dataframe(vif_df, use_container_width=True)
            st.markdown('<p class="insight">**Inference:** If PF drivers (kW/kVA/kVArh) show high VIF, avoid putting all of them simultaneously in simple linear regressions.</p>', unsafe_allow_html=True)
        else:
            st.info("Not enough clean numeric data or statsmodels not installed to compute VIF.")

    # =================== Scenarios ===================
    st.subheader("Scenarios")
    tab_pf, tab_shift, tab_fc = st.tabs(["⚡ PF Correction", "♻️ Load Shift", "🔮 Forecasts"])

    # ---- PF Correction (FIXED: Empty container removed)
    with tab_pf:
        # st.markdown('<div class="card">', unsafe_allow_html=True) <-- REMOVED EMPTY CONTAINER
        suggested = float(min(0.99, max(0.80, (k['avg_pf'] if not np.isnan(k['avg_pf']) else 0.93) + 0.02)))
        tgt_pf = st.slider("Target PF", 0.80, 0.99, suggested, 0.01)
        baseline_ll_pct = st.slider("Assumed baseline line-loss (%) of total energy", 0.0, 10.0, 2.0, 0.1)
        st.caption("Improving PF reduces reactive current and I²R losses. Baseline line-loss% is feeder/wiring loss share (typ. 1–5%).")
        scenA = scenario_pf_correction(df, interval_h, target_pf=tgt_pf)
        # st.markdown('</div>', unsafe_allow_html=True) <-- REMOVED EMPTY CONTAINER

        rel_drop = float(scenA["relative_line_loss_reduction_%"].iloc[0]) / 100.0
        total_kwh = df['Usage_kWh'].sum()
        avg_intensity = df['carbon_intensity'].mean()
        est_kwh_saved = total_kwh * (baseline_ll_pct/100.0) * rel_drop
        est_co2_saved = est_kwh_saved * (avg_intensity if not np.isnan(avg_intensity) else 0.0)

        c1, c2 = st.columns(2)
        c1.markdown('<div class="kpi"><h3>Reactive energy (kVArh)</h3><p>{:,.2f}</p></div>'.format(
            float(scenA["reactive_energy_now_kVArh"].iloc[0])), unsafe_allow_html=True)
        c2.markdown('<div class="kpi"><h3>Reduction (kVArh)</h3><p>{:,.2f}</p></div>'.format(
            float(scenA["reactive_energy_reduction_kVArh"].iloc[0])), unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        g1.plotly_chart(gauge_pf("Current PF (used)", float(scenA["avg_pf_now"].iloc[0])), use_container_width=True)
        g2.plotly_chart(gauge_pf("Target PF", float(scenA["avg_pf_target"].iloc[0])), use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.markdown('<div class="kpi"><h3>Estimated kWh saved</h3><p>{:,.0f} kWh</p></div>'.format(est_kwh_saved), unsafe_allow_html=True)
        k2.markdown('<div class="kpi"><h3>Estimated CO₂ saved</h3><p>{:,.4f} t</p></div>'.format(est_co2_saved), unsafe_allow_html=True)
        k3.markdown('<div class="kpi"><h3>Relative line-loss reduction</h3><p>{:.2f}%</p></div>'.format(
            float(scenA["relative_line_loss_reduction_%"].iloc[0])), unsafe_allow_html=True)
        st.markdown('<p class="explain">**Interpretation:** This is a what-if impact of capacitor bank sizing/retuning to reach the target PF.</p>', unsafe_allow_html=True)

    # ---- Load Shift (FIXED: Empty container removed)
    with tab_shift:
        # st.markdown('<div class="card">', unsafe_allow_html=True) <-- REMOVED EMPTY CONTAINER
        base_kwh = float(df['Usage_kWh'].sum()); base_co2 = float(df['CO2(tCO2)'].sum())
        preview = scenario_load_shift_energy_weighted(df, shift_pct=0.10, dirty_q=0.80, clean_q=0.20, cost_per_kWh=8.0)
        dirty_Iw = preview["dirty_I_weighted"].iloc[0]; clean_Iw = preview["clean_I_weighted"].iloc[0]
        hA, hB, hC, hD = st.columns([1.1, 1.1, 1.1, 1.1])
        hA.markdown(f'<div class="kpi"><h3>Baseline kWh</h3><p>{base_kwh:,.0f}</p></div>', unsafe_allow_html=True)
        hB.markdown(f'<div class="kpi"><h3>Baseline CO₂ (t)</h3><p>{base_co2:,.4f}</p></div>', unsafe_allow_html=True)
        hC.markdown(f'<div class="kpi"><h3>Dirty I\u2093 (t/kWh)</h3><p>{(0 if np.isnan(dirty_Iw) else dirty_Iw):.6f}</p></div>', unsafe_allow_html=True)
        hD.markdown(f'<div class="kpi"><h3>Clean I\u2093 (t/kWh)</h3><p>{(0 if np.isnan(clean_Iw) else clean_Iw):.6f}</p></div>', unsafe_allow_html=True)

        shift_pct = st.slider("Shift % of DIRTY energy (to the cleanest hours)", 0.00, 0.50, 0.10, 0.01)
        cost_rate = st.slider("Electricity cost (₹/kWh)", 4, 20, 8, 1)
        with st.expander("Advanced: dirty/clean quantiles"):
            dirty_q = st.slider("Dirty quantile", 0.50, 0.99, 0.80, 0.01)
            clean_q = st.slider("Clean quantile", 0.01, 0.50, 0.20, 0.01)
        # st.markdown('</div>', unsafe_allow_html=True) <-- REMOVED EMPTY CONTAINER

        scenB = scenario_load_shift_energy_weighted(df, shift_pct=shift_pct, dirty_q=dirty_q, clean_q=clean_q, cost_per_kWh=cost_rate)
        kwh_shifted = float(scenB["energy_shifted_kWh"].iloc[0])
        co2_saved   = float(scenB["estimated_CO2_savings_t"].iloc[0])
        co2_pct     = float(scenB["estimated_CO2_reduction_%"].iloc[0]) if not np.isnan(scenB["estimated_CO2_reduction_%"].iloc[0]) else 0.0
        money_saved = float(scenB["money_saved"].iloc[0])

        c1, c2 = st.columns(2)
        c1.markdown('<div class="kpi"><h3>Electricity shifted</h3><p>{:,.0f} kWh</p></div>'.format(kwh_shifted), unsafe_allow_html=True)
        c2.markdown('<div class="kpi"><h3>CO₂ saved</h3><p>{:,.4f} t</p></div>'.format(co2_saved), unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        g1.markdown('<div class="kpi"><h3>Money saved (indicative)</h3><p>₹ {:,.0f}</p></div>'.format(money_saved), unsafe_allow_html=True)
        g2.plotly_chart(gauge_percent("CO₂ reduction %", co2_pct, max_pct=20.0), use_container_width=True)
        st.markdown('<p class="explain">**Interpretation:** We reroute a portion of the dirtiest energy (by CI) into the cleanest windows. The CO₂ reduction % compares to plant baseline.</p>', unsafe_allow_html=True)

    # ---- Forecasts (WITH MODEL SELECTION)
    with tab_fc:
        if CatBoostRegressor is None and not SVR_AVAILABLE:
            st.error("Neither CatBoost nor SVR is installed. Please run: pip install catboost scikit-learn")
        else:
            model_options = []
            if CatBoostRegressor is not None: model_options.append('CatBoost')
            if SVR_AVAILABLE: model_options.append('SVR')
            
            st.markdown("### Forecasts: two evaluation styles")
            st.markdown("- **Walk-forward (rolling origin)**: metrics change with **horizon** and simulate real forward use.\n- **Backtest (hold-out)**: metrics change with **backtest window**; quick average skill.")
            
            # --- GLOBAL MODEL SELECTION ---
            selected_model = st.radio("Select Forecasting Model", model_options, key='global_model_select', horizontal=True)

            daily_d = daily_summary(df).copy()
            daily_d['ds'] = pd.to_datetime(daily_d['date_only'])
            targets_available = ['kWh','CO2_t']

            tab_wf, tab_bt = st.tabs(["Walk-forward (H-linked)", "Backtest (Hold-out)"])

            # ---- Walk-forward
            with tab_wf:
                target_wf = st.selectbox("Target (walk-forward)", targets_available, index=0, key="wf_tgt")
                hist_wf = daily_d[['ds', target_wf]].dropna().copy()
                if len(hist_wf) < 60:
                    st.warning("History is short; walk-forward metrics may be unstable.")
                max_h = max(7, min(60, int(max(7, len(hist_wf) * 0.3))))
                horizon = st.slider("Forecast horizon H (days)", 7, max_h, min(14, max_h), 1, key="wf_h_slider")

                if len(hist_wf) <= horizon + 30: 
                    st.warning("Not enough history for chosen horizon (need >30 days).")
                else:
                    eval_df, future_df = walkforward_eval(hist_wf, target_wf, horizon, selected_model)
                    if eval_df is None:
                        st.warning(f"Walk-forward could not run with current settings or {selected_model} failed to fit.")
                    else:
                        eval_df = _clean_series_for_metrics(eval_df, target_wf, 'yhat')
                        y_true = eval_df[target_wf].values
                        y_pred = eval_df['yhat'].values
                        st.markdown(f"**Walk-forward metrics (last {horizon} days) - Model: {selected_model}**")
                        st.write(f"- **WAPE:** {fmt_num(_wape(y_true,y_pred),'%')}")
                        st.write(f"- **sMAPE:** {fmt_num(_smape(y_true,y_pred),'%')}")
                        st.write(f"- **RMSE:** {fmt_num(_rmse(y_true,y_pred), f' {target_wf}')} ")
                        st.write(f"- **MAPE:** {fmt_num(_mape(y_true,y_pred),'%')}")

                        fig = px.line()
                        fig.add_scatter(x=hist_wf['ds'], y=hist_wf[target_wf], name="Historical")
                        fig.add_scatter(x=eval_df['ds'], y=eval_df['yhat'], name=f"Walk-forward pred (last {horizon}d)", line=dict(dash="dot"))
                        if future_df is not None and len(future_df):
                            fig.add_scatter(x=future_df['ds'], y=future_df['yhat'], name=f"Future forecast (+{horizon}d)", line=dict(dash="dot"))
                        fig.update_layout(title=f"Daily {target_wf}: Walk-forward eval (last {horizon}d) + Future forecast",
                                          xaxis_title="Date", yaxis_title=target_wf)
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Tables (evaluation & future)"):
                            st.dataframe(eval_df[['ds', target_wf, 'yhat']].round(3), use_container_width=True)
                            if future_df is not None and len(future_df):
                                st.dataframe(future_df.round(3), use_container_width=True)

                        st.markdown('<p class="explain">**Interpretation:** Horizon-linked metrics show how accuracy decays as you predict further out.</p>', unsafe_allow_html=True)

            # ---- Backtest
            with tab_bt:
                target_bt = st.selectbox("Target (backtest)", targets_available, index=0, key="bt_tgt")
                hist_bt = daily_d[['ds', target_bt]].dropna().copy()
                if len(hist_bt) < 40:
                    st.warning("History is short; use small backtest windows.")
                max_bt = max(7, min(90, int(max(7, len(hist_bt) * 0.4))))
                back_days = st.slider("Backtest window (days)", 7, max_bt, min(14, max_bt), 1, key="bt_h_slider")

                eval_bt = backtest_holdout(hist_bt, target_bt, back_days, selected_model)
                if eval_bt is None or len(eval_bt)==0:
                    st.warning(f"Backtest could not run with current settings or {selected_model} failed to fit.")
                else:
                    eval_bt = _clean_series_for_metrics(eval_bt, target_bt, 'yhat')
                    y_true = eval_bt[target_bt].values
                    y_pred = eval_bt['yhat'].values
                    st.markdown(f"**Backtest metrics (last {back_days} days) - Model: {selected_model}**")
                    st.write(f"- **WAPE:** {fmt_num(_wape(y_true,y_pred),'%')}")
                    st.write(f"- **sMAPE:** {fmt_num(_smape(y_true,y_pred),'%')}")
                    st.write(f"- **RMSE:** {fmt_num(_rmse(y_true,y_pred), f' {target_bt}')} ")
                    st.write(f"- **MAPE:** {fmt_num(_mape(y_true,y_pred),'%')}")

                    fig2 = px.line()
                    fig2.add_scatter(x=hist_bt['ds'], y=hist_bt[target_bt], name="Historical")
                    fig2.add_scatter(x=eval_bt['ds'], y=eval_bt['yhat'], name=f"Backtest pred (last {back_days}d)", line=dict(dash="dot"))
                    fig2.update_layout(title=f"Daily {target_bt}: Backtest on last {back_days} days",
                                      xaxis_title="Date", yaxis_title=target_bt)
                    st.plotly_chart(fig2, use_container_width=True)

                    with st.expander("Backtest table"):
                        st.dataframe(eval_bt[['ds', target_bt, 'yhat']].round(3), use_container_width=True)

                st.markdown('<p class="explain">**Interpretation:** Backtest metrics give a quick average sense of fit on a recent hold-out period.</p>', unsafe_allow_html=True)

else:
    st.info("📄 Upload a CSV in the sidebar to start analysis.")