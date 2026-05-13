"""
Data preprocessing module for LOF Anomaly Detection.
Handles data loading, cleaning, missing value imputation, and feature scaling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles data loading, cleaning, and scaling for the LOF pipeline.
    """
    
    def __init__(self, config):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        config : module
            Configuration module containing paths and parameters.
        """
        self.config = config
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the raw SCADA data from CSV.
        
        Parameters:
        -----------
        filepath : Path, optional
            Path to the CSV file. If None, uses config default.
            
        Returns:
        --------
        pd.DataFrame
            Raw data loaded from CSV.
        """
        if filepath is None:
            filepath = self.config.DATA_RAW / self.config.RAW_DATA_FILE
        
        print(f"Loading data from {filepath}...")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def inspect_data(self, df: pd.DataFrame) -> dict:
        """
        Perform initial data inspection and return summary statistics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
            
        Returns:
        --------
        dict
            Dictionary containing inspection results.
        """
        inspection = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print("\n=== Data Inspection ===")
        print(f"Shape: {inspection['shape']}")
        print(f"Missing values:\n{inspection['missing_values']}")
        print(f"Duplicate rows: {inspection['duplicates']}")
        print(f"Memory usage: {inspection['memory_usage_mb']:.2f} MB")
        
        return inspection
    
    def clean_data(self, df: pd.DataFrame, 
                   drop_columns: Optional[list] = None,
                   drop_duplicates: bool = True,
                   imputation_method: str = 'mean') -> pd.DataFrame:
        """
        Clean the dataset by removing unwanted columns, duplicates, and handling missing values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        drop_columns : list, optional
            Columns to drop. If None, uses config default.
        drop_duplicates : bool
            Whether to drop duplicate rows.
        imputation_method : str
            Method for imputing missing values ('mean', 'median', 'forward_fill').
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe.
        """
        df_clean = df.copy()
        
        # Drop specified columns
        if drop_columns is None:
            drop_columns = self.config.COLUMNS_TO_DROP
        
        cols_to_drop = [col for col in drop_columns if col in df_clean.columns]
        if cols_to_drop:
            print(f"\nDropping columns: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Drop duplicates
        if drop_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = initial_rows - len(df_clean)
            if removed > 0:
                print(f"Removed {removed} duplicate rows.")
        
        # Handle missing values
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        if missing_cols:
            print(f"\nHandling missing values in: {missing_cols}")
            if imputation_method == 'mean':
                df_clean[missing_cols] = df_clean[missing_cols].fillna(df_clean[missing_cols].mean())
            elif imputation_method == 'median':
                df_clean[missing_cols] = df_clean[missing_cols].fillna(df_clean[missing_cols].median())
            elif imputation_method == 'forward_fill':
                df_clean[missing_cols] = df_clean[missing_cols].fillna(method='ffill').fillna(method='bfill')
            
            print(f"Missing values after imputation: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only numeric columns for analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with only numeric columns.
        """
        df_numeric = df.select_dtypes(include=[np.number])
        print(f"\nSelected {df_numeric.shape[1]} numeric features.")
        self.feature_columns = df_numeric.columns.tolist()
        return df_numeric
    
    def scale_features(self, X_train: pd.DataFrame, 
                      X_test: Optional[pd.DataFrame] = None,
                      method: str = 'standard') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler or MinMaxScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training data.
        X_test : pd.DataFrame, optional
            Test data. If None, only returns scaled training data.
        method : str
            Scaling method ('standard' or 'minmax').
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled) or (X_train_scaled, None)
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        print(f"\nScaling features using {method} scaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Scaling complete. Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, filepath: Optional[Path] = None,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Complete preprocessing pipeline: load → clean → scale.
        
        Parameters:
        -----------
        filepath : Path, optional
            Path to raw data.
        test_size : float
            Proportion of data for testing (not used if no labels available).
        random_state : int
            Random seed for reproducibility.
            
        Returns:
        --------
        tuple
            (X_scaled, feature_columns, metadata_df)
        """
        # Load and inspect
        df = self.load_data(filepath)
        self.inspect_data(df)
        
        # Clean
        df_clean = self.clean_data(df)
        
        # Select numeric features
        X_numeric = self.select_numeric_features(df_clean)
        
        # Scale
        X_scaled, _ = self.scale_features(X_numeric, method=self.config.SCALING_METHOD)
        
        return X_scaled, self.feature_columns, df_clean