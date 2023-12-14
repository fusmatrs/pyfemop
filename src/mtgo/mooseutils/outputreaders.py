#
# Moose output readers
#
import pandas as pd
from materialmodeloptimizer.fullfielddata import spatialdatawrapper as sdw
import os

def output_csv_reader(filename):
    """Outputs a dict of the last row of the csv output. 

    Args:   
        filename (str): Path to the file to be read.
    """
    try:
        data = pd.read_csv(filename,
                    delimiter=',',
                    header= 0)

        output = data.iloc[-1].to_dict()
    except(FileNotFoundError):
       print('Likely model did not run, setting data as none.')
       output =None
    return output


def output_exodus_reader(filename,dic_filter=True,filter_spacing=0.2,dic_data=None,data_range='all',window_size=5):
    """Reads the exodus file into the pyvista data format. 
    Optionally processes to run the DIC filter either on actual data or
    a regularly spaced grid.

    Args:   
        filename (str): Path to the file to be read.
        dic_filter (): Whether to apply a DIC filter 
    """
    if not os.path.exists(filename):
       print('Likely model did not run, setting data as none.')
       return None
    
    # Import Moose Data
    moose_data = sdw.moose_to_spatialdata(filename)
    
    if dic_filter and dic_data is None:
        moose_data_int = sdw.interpolate_multiblock_regular(moose_data,filter_spacing)
        sdw.window_differentation(moose_data_int,data_range,window_size)
        output = moose_data_int

    elif dic_filter and dic_data is not None:
        print('Do something using actual DIC data.')
        # To do later on

    else: # Just return the Moose data
        output = moose_data
    
    return output

class OutputCSVReader():
    """Class to support reading in of csv files in parallels
    An instance of this class will be passed to the mooseherder 
    which will call read() in paralle.
    """

    def __init__(self):
        self._data_type = 'csv'
        self._extension = 'csv'
    
    def read(self,filename):
        """Read file

        Args:
            filename (str): Read file at path filename

        Returns:
            dict : Dict of the moose csv outputs at the last timestep.
        """
            
        try:
            data = pd.read_csv(filename,
                        delimiter=',',
                        header= 0)

            output = data.iloc[-1].to_dict()
        except(FileNotFoundError):
            print('Likely model did not run, setting data as none.')
            output =None
        return output

class OutputExodusReader():
    """Class to support reading in of exodus files in parallel
    An instance of this class will be passed to the mooseherder 
    which will call read() in parallel.
    """

    def __init__(self,dic_filter=True,filter_spacing=0.2,dic_data=None,data_range='all',window_size=5):
        self._data_type = 'exodus'
        self._extension = 'e'
        self._dic_filter = dic_filter
        self._data_range = data_range
        self._filter_spacing = filter_spacing
        self._dic_data = dic_data
        self._window_size = window_size
    
    def read(self,filename):
        """Read file

        Args:
            filename (str): Read file at path filename

        Returns:
            dict : Dict of the moose csv outputs at the last timestep.
        """
        
        if not os.path.exists(filename):
            print('Likely model did not run, setting data as none.')
            return None
        
        # Import Moose Data
        moose_data = sdw.moose_to_spatialdata(filename)
        
        if self._dic_filter and self._dic_data is None:
            moose_data_int = sdw.interpolate_spatialdata_grid(moose_data,self._filter_spacing)
            sdw.window_differentation(moose_data_int,self._data_range,self._window_size)
            output = moose_data_int

        elif self._dic_filter and self._dic_data is not None:
            print('Do something using actual DIC data.')
            # To do later on

        else: # Just return the Moose data
            output = moose_data
        
        return output
