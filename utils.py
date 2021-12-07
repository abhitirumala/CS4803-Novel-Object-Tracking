import numpy as np
import pandas as pd
import cv2

def parse_boxes(filepath):
    """
    Parse the bounding box file into a pandas dataframe.
    format is a txt file with each line containing:
    track_id, xmin, ymin, xmax, ymax, frame, lost, occuluded, generated, label separated by space
    """
    df = pd.read_csv(filepath, sep=' ', header=None,
                     names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])
    return df

def parse_multiple_boxes(filepaths):
    """
    Parse multiple bounding box files into a pandas dataframe.
    format is a txt file with each line containing:
    track_id, xmin, ymin, xmax, ymax, frame, lost, occuluded, generated, label separated by space
    """
    dfs = []
    for path in filepaths:
        df = parse_boxes(path)
        if not df.empty:
            dfs += [df]
        
    return pd.concat(dfs)

def convert_box_coords(coors):
    """
    Convert the bounding box from xmin, ymin, xmax, ymax to x, y, w, h
    """
    xmin, ymin, xmax, ymax = coors
    coors = np.array((xmin, ymin, xmax-xmin, ymax-ymin))
    return coors