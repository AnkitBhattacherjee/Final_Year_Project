import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class DataAnalyzer:
    """
    Core deterministic data engine executing safe Pandas analytical operations.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Strip trailing/leading whitespace from column names if present
        if any(isinstance(c, str) and (c != c.strip()) for c in self.df.columns):
            self.df.columns = [str(c).strip() for c in self.df.columns]
        
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Returns macro overview metrics for dashboard top display cards."""
        total_rows, total_cols = self.df.shape
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        missing_cells = int(self.df.isna().sum().sum())
        total_cells = total_rows * total_cols
        missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
        duplicate_rows = int(self.df.duplicated().sum())
        
        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "numeric_columns_count": len(numeric_cols),
            "categorical_columns_count": len(categorical_cols),
            "missing_cells": missing_cells,
            "missing_percentage": missing_pct,
            "duplicate_rows": duplicate_rows
        }

    def get_preview(self, rows: int = 15) -> List[Dict[str, Any]]:
        """Returns JSON-compatible list of record dictionaries for table rendering."""
        preview_df = self.df.head(rows).copy()
        for c in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[c]):
                try:
                    preview_df[c] = preview_df[c].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                except Exception:
                    preview_df[c] = preview_df[c].astype(str).replace({"NaT": "", "nan": ""})
            else:
                # Clean NaNs and infs safely
                if pd.api.types.is_numeric_dtype(preview_df[c]):
                    preview_df[c] = preview_df[c].replace([np.inf, -np.inf], np.nan).fillna("")
                else:
                    preview_df[c] = preview_df[c].fillna("")
        return preview_df.to_dict(orient="records")

    def get_column_profiles(self) -> List[Dict[str, Any]]:
        """Generates detailed metadata profiles per dataset column."""
        profiles = []
        total_rows = len(self.df)
        
        for col in self.df.columns:
            series = self.df[col]
            missing_val = int(series.isna().sum())
            missing_pct = round((missing_val / total_rows * 100), 2) if total_rows > 0 else 0.0
            unique_count = int(series.nunique())
            dtype_str = str(series.dtype)
            
            profile = {
                "column_name": col,
                "data_type": dtype_str,
                "unique_values": unique_count,
                "missing_values": missing_val,
                "missing_percentage": missing_pct,
                "mean": "N/A",
                "median": "N/A",
                "min": "N/A",
                "max": "N/A"
            }
            
            if pd.api.types.is_numeric_dtype(series):
                profile.update({
                    "mean": round(float(series.mean()), 2) if not series.dropna().empty else "N/A",
                    "median": round(float(series.median()), 2) if not series.dropna().empty else "N/A",
                    "min": round(float(series.min()), 2) if not series.dropna().empty else "N/A",
                    "max": round(float(series.max()), 2) if not series.dropna().empty else "N/A"
                })
                
            profiles.append(profile)
            
        return profiles

    def get_quality_report(self) -> Dict[str, Any]:
        """Analyzes missing values, duplicates, and outliers to calculate a health score."""
        total_rows = len(self.df)
        total_cells = self.df.size
        missing_total = int(self.df.isna().sum().sum())
        missing_pct = round((missing_total / total_cells * 100), 2) if total_cells > 0 else 0.0
        duplicates = int(self.df.duplicated().sum())
        
        outliers_detected = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            clean_s = self.df[col].dropna()
            if len(clean_s) > 4:
                q1 = clean_s.quantile(0.25)
                q3 = clean_s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    count = int(((clean_s < lower) | (clean_s > upper)).sum())
                    if count > 0:
                        outliers_detected[col] = count

        # Determine overall quality status and numerical score
        if missing_pct < 1.0 and duplicates == 0 and len(outliers_detected) == 0:
            status = "Good"
            score = 98
            summary_msg = "Dataset is clean with minimal issues."
        elif missing_pct < 8.0 and duplicates < (total_rows * 0.05):
            status = "Warning"
            score = 78
            summary_msg = "Minor data quality flags detected. Review details below."
        else:
            status = "Needs Attention"
            score = 55
            summary_msg = "Significant missing values, duplicate records, or extreme outliers detected."

        recommendations = []
        if missing_pct > 0:
            recommendations.append(f"Fill or drop missing values (representing {missing_pct}% of cells).")
        if duplicates > 0:
            recommendations.append(f"Remove {duplicates} duplicate row records to prevent double-counting.")
        if outliers_detected:
            cols_str = ", ".join(list(outliers_detected.keys())[:3])
            recommendations.append(f"Investigate potential numerical outliers in column(s): {cols_str}.")
        if not recommendations:
            recommendations.append("Data formatting is excellent. No immediate cleaning actions needed.")

        return {
            "status": status,
            "score": score,
            "summary_message": summary_msg,
            "missing_percentage": missing_pct,
            "duplicate_rows": duplicates,
            "outliers_detected": outliers_detected,
            "recommendations": recommendations
        }

    def get_schema_summary(self) -> str:
        """
        Builds a compact schema string for caching at upload time.
        This is passed to DataAIAgent so it avoids recomputing the schema
        on every chat request (important for large datasets with many columns).
        Format: column name | dtype | range or unique sample values
        """
        rows, cols = self.df.shape
        lines = [f"Dataset: {rows:,} rows × {cols} columns"]
        for col in self.df.columns:
            series = self.df[col]
            dtype = str(series.dtype)
            try:
                nunique = int(series.nunique())
            except Exception:
                nunique = 0

            if pd.api.types.is_numeric_dtype(series):
                clean_s = series.dropna()
                if not clean_s.empty:
                    try:
                        mn = float(clean_s.min())
                        mx = float(clean_s.max())
                        range_str = f"range: {mn:.2f}–{mx:.2f}"
                    except Exception:
                        range_str = f"range: {clean_s.min()}–{clean_s.max()}"
                else:
                    range_str = "all null"
                lines.append(
                    f"  {col} | {dtype} | {range_str} | {nunique} unique"
                )
            elif pd.api.types.is_datetime64_any_dtype(series):
                clean_dt = series.dropna()
                if not clean_dt.empty:
                    mn = clean_dt.min()
                    mx = clean_dt.max()
                    dt_str = f"{str(mn)[:10]}–{str(mx)[:10]}"
                else:
                    dt_str = "all null"
                lines.append(
                    f"  {col} | datetime | {dt_str} | {nunique} unique"
                )
            else:
                try:
                    sample_vals = series.dropna().unique()[:4]
                    sample_str = ", ".join(str(v) for v in sample_vals)
                except Exception:
                    sample_str = "N/A"
                lines.append(
                    f"  {col} | {dtype} | {nunique} unique (e.g. {sample_str})"
                )
        return "\n".join(lines)

    def safe_aggregate(self, group_col: str, target_col: str, agg_func: str = "sum", top_n: int = 10) -> pd.DataFrame:
        """Safely performs aggregate group operations on columns."""
        if group_col not in self.df.columns or target_col not in self.df.columns:
            return pd.DataFrame()
            
        allowed_funcs = ["sum", "mean", "min", "max", "count"]
        if agg_func.lower() not in allowed_funcs:
            agg_func = "sum"
            
        grouped = self.df.groupby(group_col, as_index=False)[target_col].agg(agg_func)
        grouped = grouped.sort_values(by=target_col, ascending=False).head(top_n)
        return grouped