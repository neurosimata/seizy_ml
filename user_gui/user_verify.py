# -*- coding: utf-8 -*-

### ---------------- IMPORTS ------------------ ###
import os
import tables
from pick import pick
import numpy as np
# User Defined
from helper.array_helper import find_szr_idx
### ------------------------------------------- ###
       
class UserVerify:
    """
        Class for user verification of detected seizures.
    """
    
    # class constructor (data retrieval)
    def __init__(self, settings):
        """

        Parameters
        ----------
        settings : dict
        """
        
        # pass settings to object attributes
        for key, value in settings.items():
            setattr(self, key, value)
            
        # set full paths       
        self.org_dir = os.path.join(self.main_path, self.org_dir)
        self.rawpred_dir = os.path.join(self.main_path, self.rawpred_dir)
        self.verpred_dir = os.path.join(self.main_path, self.verpred_dir)

        # make path if it doesn't exist
        if os.path.exists(self.verpred_dir) is False:
            os.mkdir(self.verpred_dir)
            

    def select_file(self):
        """
        Select file to load from list. Adds stars next to files that have been scored already.
        
        Returns
        -------
        option : Str, selection of file id
        """
       
        # get all files in raw predictions folder 
        rawpredlist = list(filter(lambda k: '.csv' in k, os.listdir(self.rawpred_dir)))
       
        # get all files in user verified predictions
        verpredlist = list(filter(lambda k: '.csv' in k, os.listdir(self.verpred_dir)))
       
        # get unique list
        not_analyzed_filelist = list(set(rawpredlist) - set(verpredlist))
        
        # remaining filelist
        analyzed_filelist = list(set(rawpredlist) - set(not_analyzed_filelist))
        
        # filelist
        filelist = [' *** ' + s for s in analyzed_filelist] +  not_analyzed_filelist           
        
        # select from command list
        title = 'Please select file for analysis: '
        option, index = pick(filelist, title, indicator = '-> ')
        return option.replace(' *** ','')


    def get_bounds(self, file_id):
        """

        Parameters
        ----------
        file_id : String

        Returns
        -------
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        idx_bounds : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures)

        """
        
        print('-> File being analyzed: ', file_id)

        # Get predictions
        pred_path = os.path.join(self.rawpred_dir, file_id)
        bin_pred = np.loadtxt(pred_path, delimiter=',', skiprows=0)
        idx_bounds = find_szr_idx(bin_pred, dur=1)
        
        # load raw data for visualization
        data_path = os.path.join(self.org_dir, file_id.replace('.csv','.h5'))
        f = tables.open_file(data_path, mode='r')
        data = f.root.data[:]
        f.close()
        
        # check whether to continue
        print('>>>>',idx_bounds.shape[0] ,'seizures detected')
        
        return data, idx_bounds

              
    def save_emptyidx(self, data_len, file_id):
         """
         Save user predictions to csv file as binary
        
         Returns
         -------
         None.
        
         """
         # pre allocate file with zeros
         ver_pred = np.zeros(data_len)
         
         # save file
         np.savetxt(os.path.join(self.verpred_dir, file_id), ver_pred, delimiter=',',fmt='%i')
         print('Verified predictions for ', file_id, ' were saved\n')
    
    

       
        
        
        
        
