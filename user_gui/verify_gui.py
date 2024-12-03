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
    
    ind = 0 # set internal counter
       
    def __init__(self, settings, file_id, data, idx_bounds):
        """  

        Parameters
        ----------
        settings : dict, with configuration settings
        file_id : str, file name
        data : 3D Numpy array, (1D = seizure segments, 2D =  columns (samples: window*sampling rate), 3D = channels) 
        idx_bounds : 2D Numpy array (1D = seizure segments, 2D, 1 = start, 2 = stop index)

        Returns
        -------
        None.

        """

        # pass object attributes to class
        self.data = data                                                        # data
        self.file_id = file_id                                                  # file name
        self.idx = np.copy(idx_bounds)                                          # original index from model
        self.idx_out = np.copy(idx_bounds)                                      # output index
        self.facearray = ['w']*idx_bounds.shape[0]                              # color list
        self.bounds = 60                                                        # surrounding region in seconds
        self.win = settings['win']                                              # window (column size)
        self.fs = settings['fs']                                                # sampling rate
        self.ch_list = np.array(settings['channels'])                           # channel names
        self.verpred_dir = os.path.join(settings['parent_path'], settings['verified_predictions_dir'])
        self.wait_time = 0.1 # in seconds
        
        # create figure and axis
        self.fig, self.axs = plt.subplots(data.shape[2], 1, sharex = True, figsize=(12,8))

        # remove all axes except left 
        for i in range(self.axs.shape[0]): 
            self.axs[i].spines["top"].set_visible(False)
            self.axs[i].spines["right"].set_visible(False)
            self.axs[i].spines["bottom"].set_visible(False)
            
        # create first plot
        self.plot_data()
           
        # connect callbacks and add key legend 
        plt.subplots_adjust(bottom=0.15)
        self.fig.suptitle('To Select boundaries drag mouse : '+ self.file_id, fontsize=12)         # title  
        self.fig.text(0.5, 0.09,'Time Bins (' + str(self.win) + ' Sec.)', ha="center")             # xlabel
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
        """
        Parameters
        ----------
        seconds : Int
        Returns
        -------
        str : Str
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        str = '{}:{}:{}'.format(int(hours), int(minutes), int(seconds))
        return (str)
        
        
    def save_idx(self):
        """
        Save user predictions to csv file as binary
        Returns
        -------
        None.
        """
        # pre allocate file with zeros
        ver_pred = np.zeros(self.data.shape[0])

        for i in range(self.idx_out.shape[0]): # assign index to 1
        
            if self.idx_out[i,0] > 0:
                # add 1 to stop bound because of python indexing
                ver_pred[self.idx_out[i,0]:self.idx_out[i,1]+1] = 1
            
        # save file
        np.savetxt(os.path.join(self.verpred_dir,self.file_id), ver_pred, delimiter=',',fmt='%i')
        print('Verified predictions for ', self.file_id, ' were saved.\n')    
        
        
    def get_index(self):
        """
        get i, start and stop
        Returns
        -------
        None.
        """
        self.seg = round(self.bounds/self.win)      # get surround time
        self.i = self.ind % self.idx.shape[0]       # get index
        
        if self.idx_out[self.i,1] == -1:            # if seizure rejected
            self.start = self.idx[self.i,0]         # get start
            self.stop = self.idx[self.i,1]          # get stop
        else: 
            self.start = self.idx_out[self.i,0]     # get start
            self.stop = self.idx_out[self.i,1]      # get stop
        
        
    def plot_data(self, user_start=None, user_stop=None):
        """
 
        Plot channels with highlighted seizures.

        Parameters
        ----------
        usr_start : None/int/float
        user_stop : None/int/float

        Returns
        -------
        None.

        """

        # get index, start and stop times
        self.get_index()
        
        # get seizure time
        timestr = VerifyGui.get_hours(self.start*self.win)
        timestr = '#' + str(self.i+1) + ' - '+ timestr
        
        # get boundaries for highlighted region
        if user_start is not None:   
            start = user_start; stop = user_stop # plot user defined
        else:                       
            start = self.start; stop = self.stop # plot model defined

        # plot channels
        for i in range(self.axs.shape[0]): 
            y = self.data[self.start - self.seg : self.stop + self.seg,:, i].flatten()
            t = np.linspace(self.start - self.seg, self.stop + self.seg, len(y))# get time
            self.axs[i].clear() # clear graph
            self.axs[i].plot(t, y, color='k', linewidth=0.75, alpha=0.9, label= timestr) 
            self.axs[i].set_facecolor(self.facearray[self.i]);
            self.axs[i].legend(loc = 'upper right')
            self.axs[i].set_title(self.ch_list[i], loc ='left')
            
            # plot highlighted region
            yzoom = self.data[start: stop+1,:,i].flatten() # get y values of highlighted region
            tzoom = np.linspace(start, stop+1, len(yzoom)) # get time of highlighted region
            self.axs[i].plot(tzoom, yzoom, color='orange', linewidth=0.75, alpha=0.9) # plot

        self.fig.canvas.draw()
       
          
    ## ------  Keyboard press ------ ##     
    def keypress(self, event):

        # navigate one event
        if event.key == 'right':
            self.ind += 1
            self.plot_data() # plot
        
        if event.key == 'left':
            self.ind -= 1
            self.plot_data() # plot
            
        # navigate ten events
        if event.key == 'ctrl+right':
            self.ind += 10
            self.plot_data() # plot
        
        if event.key == 'ctrl+left':
            self.ind -= 10
            self.plot_data() # plot
            
        if event.key == 'a':
            self.facearray[self.i] = 'palegreen'
            self.plot_data() # plot
            
            if self.idx_out[self.i,1] == -1:
                self.idx_out[self.i,:] = self.idx[self.i,:]
            else:
                self.idx_out[self.i,:] = self.idx_out[self.i,:]
                self.fig.canvas.draw()
                
            plt.pause(self.wait_time)
            self.ind += 1
            self.plot_data()
                
        if event.key == 'r':
            self.facearray[self.i] = 'salmon'
            self.plot_data() # plot
            self.idx_out[self.i,:] = -1  
            self.fig.canvas.draw()
            
            plt.pause(self.wait_time)
            # plot next event
            self.ind += 1
            self.plot_data()
            
        if event.key == 'enter': 
            plt.close()
            self.save_idx()
            print(self.idx_out)
            print(self.idx_out.shape[0]-np.sum(self.idx_out[:,0] == -1),'Seizures accepted.\n')
            
        if event.key == 'escape': 
            plt.close()


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
        indmin = int(xmin); indmax = int(xmax)
        
        # pass to index
        self.idx_out[self.i,0] = indmin
        self.idx_out[self.i,1] = indmax
        
        # highlight user selected region
        self.plot_data(user_start=indmin, user_stop=indmax)
