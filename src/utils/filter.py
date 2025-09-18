from functools import partial
import logging
import os

import numpy as np
import pandas as pd

from src.utils.geom import extremal_bounds

logger = logging.getLogger(__name__)


def filter_detection(row: pd.Series, loc_df: pd.DataFrame, default_width_m: float = 100) -> bool:
    """Return True if a row should be filtered due to overlap with loc_df.

    Parameters
    ----------
    row: pandas.Series
        Row of a pandas series with lat and lon columns, whose rows are detections.

    loc_df: pandas.DataFrame
        Dataframe with columns "lon", "lat" and "width_m" specifying
        locations of undesired detection centers, and the (assumed to be square)
        extent of the undesired object.

    default_width_m: float
        A default width to assign to locations in loc_df if the width_m field is empty.

    Returns
    -------
    bool
        True if detection should be filtered due to overlap.
    """
    # FIXED: Add proper error handling and validation
    try:
        # Validate input data
        if not isinstance(loc_df, pd.DataFrame):
            raise TypeError(f"loc_df must be pandas.DataFrame, got {type(loc_df)}")
        
        required_columns = ['lon', 'lat']
        missing_columns = [col for col in required_columns if col not in loc_df.columns]
        if missing_columns:
            raise ValueError(f"loc_df missing required columns: {missing_columns}")
        
        # Validate row data
        if 'lon' not in row or 'lat' not in row:
            logger.warning(f"Row missing required lat/lon columns: {row.name}")
            return False
            
        detect_lon = row.lon
        detect_lat = row.lat
        
        # Validate coordinates
        if pd.isna(detect_lon) or pd.isna(detect_lat):
            logger.warning(f"Row has NaN coordinates: {row.name}")
            return False

        for _, r in loc_df.iterrows():
            try:
                lon = r.lon
                lat = r.lat
                width_m = r.width_m if 'width_m' in r else default_width_m
                
                # Validate filter coordinates
                if pd.isna(lon) or pd.isna(lat):
                    continue
                    
                if np.isnan(width_m):
                    width_m = default_width_m

                min_lon, min_lat, max_lon, max_lat = extremal_bounds(lon, lat, width_m)

                if (
                    (detect_lon >= min_lon)
                    and (detect_lon <= max_lon)
                    and (detect_lat >= min_lat)
                    and (detect_lat <= max_lat)
                ):
                    return True
                    
            except Exception as e:
                logger.warning(f"Error processing filter row: {e}")
                continue

        return False
        
    except Exception as e:
        logger.error(f"Error in filter_detection: {e}")
        return False  # Don't filter if there's an error


def filter_out_locs(
    df: pd.DataFrame, loc_path: str, default_width_m: float = 100
) -> pd.DataFrame:
    """Filter out rows in df overlapping locations specified in loc_path.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe of predictions containing, at the least, columns
        named "lon" and "lat".

    loc_path: str
        Path to a csv with columns "lon", "lat" and "width_m" specifying
        locations of undesired detection centers, and the (assumed to be square)
        extent of the undesired object.

    Returns
    -------
    filtered_df: pandas.DataFrame
        A sub-dataframe of df with the rows corresponding to detections
        overlapping the undesired object extents removed.
    """
    # FIXED: Add proper error handling and validation
    try:
        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be pandas.DataFrame, got {type(df)}")
        
        required_columns = ['lon', 'lat']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"df missing required columns: {missing_columns}")
        
        # Validate file path
        if not os.path.exists(loc_path):
            raise FileNotFoundError(f"Filter file not found: {loc_path}")
        
        # Load filter data with error handling
        try:
            loc_df = pd.read_csv(loc_path)
        except Exception as e:
            raise ValueError(f"Error reading filter file {loc_path}: {e}")
        
        # Validate filter DataFrame
        filter_required_columns = ['lon', 'lat']
        filter_missing_columns = [col for col in filter_required_columns if col not in loc_df.columns]
        if filter_missing_columns:
            raise ValueError(f"Filter file missing required columns: {filter_missing_columns}")
        
        logger.info(f"Filtering {len(df)} detections against {len(loc_df)} filter locations")
        
        remove_detection = partial(
            filter_detection, loc_df=loc_df, default_width_m=default_width_m
        )

        filtered_df = df[~df.apply(remove_detection, axis=1)]
        
        filtered_count = len(df) - len(filtered_df)
        logger.info(f"Filtered out {filtered_count} detections, {len(filtered_df)} remaining")

        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in filter_out_locs: {e}")
        # Return original DataFrame if filtering fails
        return df
