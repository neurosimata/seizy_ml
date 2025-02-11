# -*- coding: utf-8 -*-

### ---------------------- IMPORTS ---------------------- ###
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from PyQt5 import QtCore
### ----------------------------------------------------- ###

class VerifyGui(object):
    """
        Matplotlib GUI for user seizure verification.
    """
    
    def __init__(self, parent_path, settings, file_id, data, idx_bounds, color_array=None):
        """  

        Parameters
        ----------
        parent_path, str, path parent directory
        settings : dict, with configuration settings
        file_id : str, file name
        data : 3D Numpy array, (1D = seizure segments, 2D =  columns (samples: window*sampling rate), 3D = channels) 
        idx_bounds : 2D Numpy array (1D = seizure segments, 2D, 1 = start, 2 = stop index)
        color_array: array-like, with colors to plot depending on verification status (accepted, rejected, unverified)

        Returns
        -------
        None.

        """
        
        # settings
        self.file_id = file_id                                                  # file name
        self.model_win = settings['win']                                        # model window in seconds
        self.gui_win = settings['gui_win']                                      # gui window in seconds
        self.fs = settings['fs']                                                # sampling rate
        self.bounds = 60                                                        # surrounding region in seconds
        self.bounds_bins = round(self.bounds/self.gui_win)                      # get surround time in bins
        self.ch_list = np.array(settings['channels'])                           # channel names
        self.verpred_dir = os.path.join(parent_path, settings['verified_predictions_dir'])
        self.accepted_color = 'palegreen'
        self.rejected_color = 'salmon'
        self.unverified_color = 'w'
        self.wait_time = 0.1                                                    # in seconds
        self.ind = 0

        # check if data were verified
        if color_array is None:
            self.color_array = [self.unverified_color]*idx_bounds.shape[0]
        else:
            self.color_array = list(color_array)
        
        # adjust index to match gui window
        self.data = data.reshape(-1, int(self.fs * self.gui_win), data.shape[2])
        self.idx = (np.copy(idx_bounds) * (self.model_win / self.gui_win)).astype(int)

        # create figure and axis
        self.fig, self.axs = plt.subplots(data.shape[2], 1, sharex=True, figsize=(12,8))
        if data.shape[2] == 1:
            self.axs = [self.axs]
        for i in range(self.axs.shape[0]): 
            self.axs[i].spines["top"].set_visible(False)
            self.axs[i].spines["right"].set_visible(False)
            self.axs[i].spines["bottom"].set_visible(False)
        self.plot_data()
           
        # connect callbacks and add key legend 
        plt.subplots_adjust(bottom=0.15)
        self.fig.suptitle('To Select boundaries drag mouse : '+ self.file_id, fontsize=12)         # title  
        self.fig.text(0.5, 0.09,'Time  (Seconds)', ha="center")         # xlabel
        self.fig.text(.02, .5, 'Amp. (V)', ha='center', va='center', rotation='vertical')          # ylabel
        self.fig.text(0.5, 0.04, 
                      "** Accept/Reject = a/r,      Previous/Next = ←/→,    ctrl+Previous/Next = -10/+10, \n Enter = Save, Esc = close(no Save), Drag cursor (left click) to adjust bounds**" ,
                      ha="center", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))
        self.fig.canvas.callbacks.connect('key_press_event', self.keypress)
        
        # disable x button
        win = plt.gcf().canvas.manager.window
        win.setWindowFlags(win.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        win.setWindowFlags(win.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        
        # span selector
        _ = SpanSelector(self.axs[0], self.onselect, 'horizontal', useblit=True,
            rectprops=dict(alpha=0.5, facecolor='tab:blue'))
        plt.show()
    
    @staticmethod
    def get_hours(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        time_str = '{}:{}:{}'.format(int(hours), int(minutes), int(seconds))
        return time_str
          
    def save_idx(self):
        """
        Save user predictions to csv file as binary
        Returns
        -------
        None.
        """
        # create bool array with verified predictions
        ver_pred = np.zeros(self.data.shape[0])
        for idx, color in zip(self.idx, self.color_array):
            if color == self.accepted_color:
                ver_pred[idx[0]:idx[1]+1] = 1
            
        # save file
        np.savetxt(os.path.join(self.verpred_dir, self.file_id), ver_pred, delimiter=',',fmt='%i')
        np.savetxt(os.path.join(self.verpred_dir, 'color_'+self.file_id.replace('.csv', '.txt')), np.array(self.color_array), delimiter=',', fmt='%s')
        print('Verified predictions for ', self.file_id, ' were saved.\n')
    
    ## ----- User Selection ----##        
    def onselect(self, xmin, xmax):
        """
        Parameters
        ----------
        xmin : Float
            Xmin-user selection.
        xmax : Float
            Xmax-user selection.
        """
               
        # find user segment index from plot
        indmin = int(xmin)
        indmax = int(xmax)
        
        # pass to index
        self.idx[self.i,0] = indmin
        self.idx[self.i,1] = indmax
        
        # highlight user selected region
        self.plot_data(user_start=indmin, user_stop=indmax)
            
    def plot_data(self, user_start=None, user_stop=None):
        """
        Plot channels with highlighted seizures.

        Returns
        -------
        None.

        """

        # get index, start and stop times
        self.i = self.ind % self.idx.shape[0]
        start,stop = self.idx[self.i]
        if user_start is not None:
            start,stop = user_start, user_stop
        
        # get seizure time
        timestr = VerifyGui.get_hours(start*self.gui_win)
        timestr = '#' + str(self.i+1) + ' - '+ timestr

        # plot channels
        for i in range(self.axs.shape[0]): 
            start_idx = max(0, start - self.bounds_bins)
            stop_idx = min(self.data.shape[0], stop + self.bounds_bins)
            y = self.data[start_idx:stop_idx, :, i].flatten()
            t = np.linspace(start - self.bounds_bins, stop + self.bounds_bins, len(y))*self.gui_win
            self.axs[i].clear()
            self.axs[i].plot(t, y, color='k', linewidth=0.75, alpha=0.9, label=timestr)
            self.axs[i].set_facecolor(self.color_array[self.i])
            self.axs[i].legend(loc='upper right')
            self.axs[i].set_title(self.ch_list[i], loc='left')
            
            # plot highlighted region
            yzoom = self.data[start: stop+1,:,i].flatten()
            tzoom = np.linspace(start, stop+1, len(yzoom))*self.gui_win
            self.axs[i].plot(tzoom, yzoom, color='orange', linewidth=0.75, alpha=0.9)
        self.fig.canvas.draw()
       
    ## ------  Keyboard press ------ ##     
    def keypress(self, event):

        # navigate one event
        if event.key == 'right':
            self.ind += 1
            self.plot_data()
        
        if event.key == 'left':
            self.ind -= 1
            self.plot_data()
            
        # navigate ten events
        if event.key == 'ctrl+right':
            self.ind += 10
            self.plot_data()
        
        if event.key == 'ctrl+left':
            self.ind -= 10
            self.plot_data()
        
        # accept and reject events
        if event.key == 'a':
            self.color_array[self.i] = self.accepted_color
            self.plot_data()
            self.fig.canvas.draw()
            plt.pause(self.wait_time)
            self.ind += 1
            self.plot_data()
                
        if event.key == 'r':
            self.color_array[self.i] = self.rejected_color
            self.plot_data()
            self.fig.canvas.draw()
            plt.pause(self.wait_time)
            self.ind += 1
            self.plot_data()
        
        if event.key == 'enter': 
            plt.close()
            self.save_idx()
            
            # get number of accepted, rejected, and unverified seizures
            verified_array = np.array(self.color_array)
            not_verified = np.sum(verified_array == self.unverified_color)
            if not_verified > 0:
                print(f'--> Warning {not_verified} were not verified!')
            total_seizures = len(verified_array)
            accepted = np.sum(verified_array== self.accepted_color)
            rejected = np.sum(verified_array == self.rejected_color)
            print(f'Total predicted events = {total_seizures}. Accepted = {accepted}, Rejected = {rejected}, Unverified = {not_verified}.\n')
            
        if event.key == 'escape': 
            plt.close()
