# settings_editor.py
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

class SettingsEditor(ctk.CTk):
    def __init__(self, current_settings):
        super().__init__()
        # This attribute will hold updated settings (or None if the user cancels)
        self.new_settings = None
        # Work on a copy so we don't modify the original until saving
        self.settings = current_settings.copy()

        self.title("Settings Editor")
        self.geometry("600x600")
        
        # Create a Tabview widget for grouping settings
        self.tabview = ctk.CTkTabview(self, width=580, height=500)
        self.tabview.pack(pady=10, padx=10)
        
        # Add tabs
        self.tabview.add("Data Settings")
        self.tabview.add("GUI Settings")
        self.tabview.add("Features")
        self.tabview.add("Post Processing")
        self.tabview.add("Directories")
        
        # Build content for each tab
        self.create_data_settings_tab()
        self.create_gui_settings_tab()
        self.create_features_tab()
        self.create_post_processing_tab()
        self.create_directories_tab()
        
        # Frame for Save and Cancel buttons
        self.button_frame = ctk.CTkFrame(self, width=580, height=50)
        self.button_frame.pack(pady=10)
        self.save_button = ctk.CTkButton(self.button_frame, text="Save", command=self.save_settings)
        self.save_button.pack(side="left", padx=10)
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="left", padx=10)

    def create_data_settings_tab(self):
        tab = self.tabview.tab("Data Settings")
        # Channels
        ctk.CTkLabel(tab, text="Channels (comma separated):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.channels_entry = ctk.CTkEntry(tab, width=300)
        self.channels_entry.grid(row=0, column=1, padx=10, pady=5)
        self.channels_entry.insert(0, ', '.join(self.settings['channels']))
        
        # Sampling frequency (fs)
        ctk.CTkLabel(tab, text="Sampling frequency (fs):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.fs_entry = ctk.CTkEntry(tab, width=100)
        self.fs_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.fs_entry.insert(0, str(self.settings['fs']))
        
        # Window length (win)
        ctk.CTkLabel(tab, text="Window length (win):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.win_entry = ctk.CTkEntry(tab, width=100)
        self.win_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.win_entry.insert(0, str(self.settings['win']))
    
    def create_gui_settings_tab(self):
        tab = self.tabview.tab("GUI Settings")
        ctk.CTkLabel(tab, text="GUI Window (gui_win):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.gui_win_entry = ctk.CTkEntry(tab, width=100)
        self.gui_win_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.gui_win_entry.insert(0, str(self.settings['gui_win']))
    
    def create_features_tab(self):
        tab = self.tabview.tab("Features")
        ctk.CTkLabel(tab, text="Features (comma separated):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.features_entry = ctk.CTkEntry(tab, width=400)
        self.features_entry.grid(row=0, column=1, padx=10, pady=5)
        self.features_entry.insert(0, ', '.join(self.settings['features']))
        
        ctk.CTkLabel(tab, text="Feature Select Threshold:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.feature_thresh_entry = ctk.CTkEntry(tab, width=100)
        self.feature_thresh_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.feature_thresh_entry.insert(0, str(self.settings['feature_select_thresh']))
        
        ctk.CTkLabel(tab, text="Feature Size (comma separated):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.feature_size_entry = ctk.CTkEntry(tab, width=200)
        self.feature_size_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.feature_size_entry.insert(0, ', '.join(map(str, self.settings['feature_size'])))
        
        ctk.CTkLabel(tab, text="N least correlated:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.nleast_corr_entry = ctk.CTkEntry(tab, width=100)
        self.nleast_corr_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.nleast_corr_entry.insert(0, str(self.settings['nleast_corr']))
    
    def create_post_processing_tab(self):
        tab = self.tabview.tab("Post Processing")
        ctk.CTkLabel(tab, text="Post Processing Method:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.pp_method_combobox = ctk.CTkComboBox(tab, values=["dual_threshold", "dilation_erosion", "erosion_dilation"])
        self.pp_method_combobox.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.pp_method_combobox.set(self.settings['post_processing_method'])
        
        ctk.CTkLabel(tab, text="Rolling Window:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.rolling_window_entry = ctk.CTkEntry(tab, width=100)
        self.rolling_window_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.rolling_window_entry.insert(0, str(self.settings['rolling_window']))
        
        ctk.CTkLabel(tab, text="Event Threshold:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.event_threshold_entry = ctk.CTkEntry(tab, width=100)
        self.event_threshold_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.event_threshold_entry.insert(0, str(self.settings['event_threshold']))
        
        ctk.CTkLabel(tab, text="Boundary Threshold:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.boundary_threshold_entry = ctk.CTkEntry(tab, width=100)
        self.boundary_threshold_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.boundary_threshold_entry.insert(0, str(self.settings['boundary_threshold']))
        
        ctk.CTkLabel(tab, text="Dilation:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.dilation_entry = ctk.CTkEntry(tab, width=100)
        self.dilation_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.dilation_entry.insert(0, str(self.settings['dilation']))
        
        ctk.CTkLabel(tab, text="Erosion:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.erosion_entry = ctk.CTkEntry(tab, width=100)
        self.erosion_entry.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.erosion_entry.insert(0, str(self.settings['erosion']))
    
    def create_directories_tab(self):
        tab = self.tabview.tab("Directories")
        ctk.CTkLabel(tab, text="Data Directory:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.data_dir_entry = ctk.CTkEntry(tab, width=200)
        self.data_dir_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.data_dir_entry.insert(0, self.settings['data_dir'])
        
        ctk.CTkLabel(tab, text="Processed Directory:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.processed_dir_entry = ctk.CTkEntry(tab, width=200)
        self.processed_dir_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.processed_dir_entry.insert(0, self.settings['processed_dir'])
        
        ctk.CTkLabel(tab, text="Model Predictions Directory:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.model_predictions_entry = ctk.CTkEntry(tab, width=200)
        self.model_predictions_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.model_predictions_entry.insert(0, self.settings['model_predictions_dir'])
        
        ctk.CTkLabel(tab, text="Verified Predictions Directory:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.verified_predictions_entry = ctk.CTkEntry(tab, width=200)
        self.verified_predictions_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.verified_predictions_entry.insert(0, self.settings['verified_predictions_dir'])
        
        ctk.CTkLabel(tab, text="Trained Model Directory:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.trained_model_entry = ctk.CTkEntry(tab, width=200)
        self.trained_model_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.trained_model_entry.insert(0, self.settings['trained_model_dir'])
    
    def save_settings(self):
        try:
            new_settings = {}
            # Data Settings
            new_settings['channels'] = [s.strip() for s in self.channels_entry.get().split(',') if s.strip()]
            new_settings['fs'] = int(self.fs_entry.get())
            new_settings['win'] = int(self.win_entry.get())
            
            # GUI Settings
            new_settings['gui_win'] = int(self.gui_win_entry.get())
            
            # Features
            new_settings['features'] = [s.strip() for s in self.features_entry.get().split(',') if s.strip()]
            new_settings['feature_select_thresh'] = float(self.feature_thresh_entry.get())
            new_settings['feature_size'] = [int(s.strip()) for s in self.feature_size_entry.get().split(',') if s.strip()]
            new_settings['nleast_corr'] = int(self.nleast_corr_entry.get())
            
            # Post Processing
            new_settings['post_processing_method'] = self.pp_method_combobox.get()
            new_settings['rolling_window'] = int(self.rolling_window_entry.get())
            new_settings['event_threshold'] = float(self.event_threshold_entry.get())
            new_settings['boundary_threshold'] = float(self.boundary_threshold_entry.get())
            new_settings['dilation'] = int(self.dilation_entry.get())
            new_settings['erosion'] = int(self.erosion_entry.get())
            
            # Directories
            new_settings['data_dir'] = self.data_dir_entry.get()
            new_settings['processed_dir'] = self.processed_dir_entry.get()
            new_settings['model_predictions_dir'] = self.model_predictions_entry.get()
            new_settings['verified_predictions_dir'] = self.verified_predictions_entry.get()
            new_settings['trained_model_dir'] = self.trained_model_entry.get()
            
            self.new_settings = new_settings  # Save the new settings
            self.destroy()  # Close the GUI
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def cancel(self):
        # If the user cancels, simply close the GUI without saving changes.
        self.destroy()


def edit_settings_gui(current_settings):
    """
    Launch the settings editor GUI with the current settings.
    Returns the updated settings if saved, or the original settings if canceled.
    """
    app = SettingsEditor(current_settings)
    app.mainloop()
    # Return the new settings if they exist, else the original settings.
    return app.new_settings if app.new_settings is not None else current_settings

# ----------------------------------------------------------------
