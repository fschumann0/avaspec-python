from avaspec import *
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from scipy.optimize import curve_fit
from scipy import constants

class SpectrometerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrometer Control Panel")
        self.root.geometry("1200x800")
        
        self.handle = None
        self.wavelengths = []
        self.pixels = 0
        self.measconfig = None
        self.is_live = False
        self.dark_spectrum = None
        self.current_spectrum = None
        self.dark_subtraction_enabled = False
        self.planck_fit = None
        self.planck_temperature = None
        self.planck_r_squared = None
        self.planck_enabled = False
        self.peak_wavelength = None
        self.peak_intensity = None
        self.show_peak = True
        self.show_crosshair = False
        self.crosshair_lines = None

        self.sensitivity_wavelengths = None
        self.sensitivity_values = None
        self.sensitivity_loaded = False
        
        
        self.init_spectrometer()
        
        self.create_widgets()
        
        self.update_plot()
        
    def init_spectrometer(self):
        
        try:
ret = AVS_Init(-1)
ret = AVS_UpdateUSBDevices()
            device_list = AVS_GetList()
            
            if len(device_list) == 0:
                messagebox.showerror("Error", "No spectrometer found!")
                self.root.quit()
                return
            
            self.handle = AVS_Activate(device_list[0])
            ret = AVS_GetParameter(self.handle)
            self.pixels = AVS_GetNumPixels(self.handle)
            lamb = AVS_GetLambda(self.handle)
            
            self.wavelengths = [lamb[i] for i in range(self.pixels)]
            
            ret = AVS_UseHighResAdc(self.handle, True)
            
            self.measconfig = MeasConfigType()
            self.measconfig.m_StartPixel = 0
            self.measconfig.m_StopPixel = self.pixels - 1
            self.measconfig.m_IntegrationTime = 50.0
            self.measconfig.m_IntegrationDelay = 0
            self.measconfig.m_NrAverages = 1
            self.measconfig.m_CorDynDark_m_Enable = 0
            self.measconfig.m_CorDynDark_m_ForgetPercentage = 100
            self.measconfig.m_Smoothing_m_SmoothPix = 0
            self.measconfig.m_Smoothing_m_SmoothModel = 0
            self.measconfig.m_SaturationDetection = 0
            self.measconfig.m_Trigger_m_Mode = 0
            self.measconfig.m_Trigger_m_Source = 0
            self.measconfig.m_Trigger_m_SourceType = 0
            self.measconfig.m_Control_m_StrobeControl = 0
            self.measconfig.m_Control_m_LaserDelay = 0
            self.measconfig.m_Control_m_LaserWidth = 0
            self.measconfig.m_Control_m_LaserWaveLength = 0.0
            self.measconfig.m_Control_m_StoreToRam = 0
            
            ret = AVS_PrepareMeasure(self.handle, self.measconfig)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize spectrometer:\n{str(e)}")
            self.root.quit()
    
    def create_widgets(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        ttk.Label(control_frame, text="Measurement Mode:", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        
        self.live_btn = ttk.Button(control_frame, text="Start Live", command=self.toggle_live, width=20)
        self.live_btn.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.single_btn = ttk.Button(control_frame, text="Single Measurement", command=self.single_measurement, width=20)
        self.single_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.dark_btn = ttk.Button(control_frame, text="Capture Dark", command=self.capture_dark, width=20)
        self.dark_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.dark_subtract_var = tk.BooleanVar(value=False)
        dark_check = ttk.Checkbutton(control_frame, text="Enable Dark Subtraction", 
                                     variable=self.dark_subtract_var, command=self.toggle_dark_subtraction)
        dark_check.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        self.planck_btn = ttk.Button(control_frame, text="Fit Planck Curve", command=self.fit_planck, width=20)
        self.planck_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.planck_show_var = tk.BooleanVar(value=False)
        planck_check = ttk.Checkbutton(control_frame, text="Show Planck Fit", 
                                        variable=self.planck_show_var, command=self.toggle_planck_display)
        planck_check.grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        self.show_peak_var = tk.BooleanVar(value=True)
        peak_check = ttk.Checkbutton(control_frame, text="Show Peak Marker", 
                                     variable=self.show_peak_var, command=self.toggle_peak_display)
        peak_check.grid(row=7, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        self.show_crosshair_var = tk.BooleanVar(value=False)
        crosshair_check = ttk.Checkbutton(control_frame, text="Show Crosshair", 
                                          variable=self.show_crosshair_var, command=self.toggle_crosshair)
        crosshair_check.grid(row=8, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="10")
        settings_frame.grid(row=9, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(settings_frame, text="Integration Time (ms):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.integration_var = tk.DoubleVar(value=50.0)
        integration_spin = ttk.Spinbox(settings_frame, from_=1.0, to=60000.0, textvariable=self.integration_var, 
                                       width=15, command=self.update_integration_time)
        integration_spin.grid(row=0, column=1, pady=5, sticky=(tk.W, tk.E))
        integration_spin.bind('<Return>', lambda e: self.update_integration_time())
        
        ttk.Label(settings_frame, text="Averages:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.averages_var = tk.IntVar(value=1)
        averages_spin = ttk.Spinbox(settings_frame, from_=1, to=1000, textvariable=self.averages_var, 
                                    width=15, command=self.update_averages)
        averages_spin.grid(row=1, column=1, pady=5, sticky=(tk.W, tk.E))
        averages_spin.bind('<Return>', lambda e: self.update_averages())
        
        ttk.Label(settings_frame, text="Y-axis max (counts):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.y_axis_max_var = tk.DoubleVar(value=70000.0)
        y_axis_spin = ttk.Spinbox(settings_frame, from_=100.0, to=1000000.0,
                                  textvariable=self.y_axis_max_var, width=15, command=self.update_y_axis_max)
        y_axis_spin.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))
        y_axis_spin.bind('<Return>', lambda e: self.update_y_axis_max())

        ttk.Label(settings_frame, text="Sensitivity curve:").grid(row=3, column=0, sticky=tk.W, pady=5)
        load_sens_btn = ttk.Button(settings_frame, text="Load CSV", command=self.load_sensitivity_curve, width=12)
        load_sens_btn.grid(row=3, column=1, sticky=(tk.W), pady=5)

        self.sensitivity_apply_var = tk.BooleanVar(value=False)
        sens_check = ttk.Checkbutton(
            settings_frame,
            text="Apply sensitivity correction",
            variable=self.sensitivity_apply_var,
            command=self.update_plot,
        )
        sens_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=10, column=0, columnspan=2, pady=(20, 10), sticky=(tk.W, tk.E))
        self.save_btn = ttk.Button(control_frame, text="💾 Save Spectrum (CSV)", command=self.save_spectrum, width=22)
        self.save_btn.grid(row=11, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        self.export_png_btn = ttk.Button(control_frame, text="🖼 Export Plot (PNG)", command=self.export_plot_png, width=22)
        self.export_png_btn.grid(row=12, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="10")
        status_frame.grid(row=13, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.dark_status_label = ttk.Label(status_frame, text="Dark: Not captured", foreground="gray")
        self.dark_status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        self.planck_status_label = ttk.Label(status_frame, text="Planck: Not fitted", foreground="gray")
        self.planck_status_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        self.sensitivity_status_label = ttk.Label(status_frame, text="Sensitivity: none", foreground="gray")
        self.sensitivity_status_label.grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Wavelength (nm)", fontsize=12)
        self.ax.set_ylabel("Intensity (counts)", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.crosshair_vline = None
        self.crosshair_hline = None
        self.crosshair_text = None
        self.y_axis_max = 70000.0
        
        control_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(1, weight=1)
    
    def update_integration_time(self):
        
        if self.measconfig:
            self.measconfig.m_IntegrationTime = float(self.integration_var.get())
            AVS_PrepareMeasure(self.handle, self.measconfig)
    
    def update_averages(self):
        
        if self.measconfig:
            self.measconfig.m_NrAverages = int(self.averages_var.get())
            AVS_PrepareMeasure(self.handle, self.measconfig)
    
    def update_y_axis_max(self):
        """Apply user-defined y-axis max and refresh plot."""
        try:
            val = float(self.y_axis_max_var.get())
            if val < 1:
                val = 70000.0
            self.y_axis_max = val
            self.update_plot()
        except (ValueError, tk.TclError):
            self.y_axis_max_var.set(self.y_axis_max)
    
    def take_measurement(self):
        
        try:
            AVS_Measure(self.handle, 0, 1)
dataready = False
            timeout = time.time() + 10 
            poll_interval = max(0.01, self.measconfig.m_IntegrationTime / 1000.0 / 10) 
            
            while not dataready and time.time() < timeout:
                dataready = AVS_PollScan(self.handle)
                if not dataready:
                    time.sleep(poll_interval)
            
            if not dataready:
                return None
            
            timestamp, scopedata = AVS_GetScopeData(self.handle)
            spectrum = [scopedata[i] for i in range(self.pixels)]
            
            return np.array(spectrum)
        except Exception as e:
            if hasattr(self, 'status_label'):
                self.root.after(0, lambda: self.status_label.config(text=f"Error: {str(e)}", foreground="red"))
            return None
    
    def single_measurement(self):
        
        if self.is_live:
            self.toggle_live()
        
        self.status_label.config(text="Measuring...", foreground="orange")
        self.root.update()
        
        spectrum = self.take_measurement()
        
        if spectrum is not None:
            self.current_spectrum = spectrum
            self.update_plot()
            self.status_label.config(text="Measurement complete", foreground="green")
        else:
            self.status_label.config(text="Measurement failed", foreground="red")
    
    def capture_dark(self):
        
        if self.is_live:
            messagebox.showwarning("Warning", "Please stop live measurement before capturing dark spectrum.")
            return
        
        self.status_label.config(text="Capturing dark spectrum...", foreground="orange")
        self.root.update()
        
        spectrum = self.take_measurement()
        
        if spectrum is not None:
            self.dark_spectrum = spectrum
            self.dark_status_label.config(text="Dark: Captured", foreground="green")
            self.status_label.config(text="Dark spectrum captured", foreground="green")
            self.update_plot()
        else:
            self.status_label.config(text="Dark capture failed", foreground="red")
    
    def toggle_dark_subtraction(self):
        
        self.dark_subtraction_enabled = self.dark_subtract_var.get()
        self.update_plot()
    
    def toggle_live(self):
        
        if not self.is_live:
            self.is_live = True
            self.live_btn.config(text="Stop Live")
            self.status_label.config(text="Live measurement running...", foreground="blue")
            self.live_thread = threading.Thread(target=self.live_measurement_loop, daemon=True)
            self.live_thread.start()
        else:
            self.is_live = False
            self.live_btn.config(text="Start Live")
            self.status_label.config(text="Live measurement stopped", foreground="green")
    
    def live_measurement_loop(self):

        while self.is_live:
            try:
                spectrum = self.take_measurement()
                if spectrum is not None:
                    self.current_spectrum = spectrum
                    
                    self.root.after_idle(self.update_plot)
                
                time.sleep(0.05)
            except Exception as e:
                if self.is_live:  
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"Live error: {str(e)}", foreground="red"))
                break

    def _compute_display_spectrum(self):
        """
        Berechnet das Spektrum so, wie es angezeigt / gespeichert wird:
        - optional Dunkelkorrektur
        - optional Sensitivitätskorrektur
        Gibt (spectrum_array, label_text) zurück.
        """
        if self.current_spectrum is None:
            return None, "Spectrum"

        spectrum = self.current_spectrum.copy()
        label = "Spectrum"

            if self.dark_subtraction_enabled and self.dark_spectrum is not None:
            spectrum = spectrum - self.dark_spectrum
            spectrum = np.maximum(spectrum, 0)
            label = "Spectrum (dark corrected)"

            if (
            getattr(self, "sensitivity_apply_var", None) is not None
            and self.sensitivity_apply_var.get()
            and self.sensitivity_wavelengths is not None
            and self.sensitivity_values is not None
        ):
            try:
                sens_interp = np.interp(
                    np.array(self.wavelengths, dtype=float),
                    self.sensitivity_wavelengths,
                    self.sensitivity_values,
                    left=np.nan,
                    right=np.nan,
                )
                valid = np.isfinite(sens_interp) & (sens_interp > 0)
                if np.any(valid):
                    corrected = spectrum.copy()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corrected[valid] = spectrum[valid] / sens_interp[valid]
                    spectrum = corrected
                    label = (
                        "Spectrum (dark + sensitivity corrected)"
                        if self.dark_subtraction_enabled and self.dark_spectrum is not None
                        else "Spectrum (sensitivity corrected)"
                    )
            except Exception:
                pass

        return spectrum, label
    
    def update_plot(self):
    
        if len(self.wavelengths) == 0:
            return
    
        if self.crosshair_vline is not None:
            self.crosshair_vline.remove()
            self.crosshair_vline = None
        if self.crosshair_hline is not None:
            self.crosshair_hline.remove()
            self.crosshair_hline = None
        if self.crosshair_text is not None:
            self.crosshair_text.remove()
            self.crosshair_text = None
        
        self.ax.clear()
        self.ax.set_xlabel("Wellenlänge / nm", fontsize=12)
        self.ax.set_ylabel("Intensität / counts", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        spectrum_to_plot = None
        spectrum_label = "Spectrum"
        if self.current_spectrum is not None:
            spectrum_to_plot, spectrum_label = self._compute_display_spectrum()

            self.ax.plot(self.wavelengths, spectrum_to_plot, "k-", linewidth=0.8, label=spectrum_label)

            if self.dark_spectrum is not None:
                self.ax.plot(self.wavelengths, self.dark_spectrum, "r--", linewidth=1, alpha=0.5, label="Dark")
            
        if self.planck_show_var.get() and self.planck_fit is not None and self.planck_temperature is not None:

            if isinstance(self.planck_fit, np.ndarray) and len(self.planck_fit) == len(self.wavelengths):
                self.ax.plot(self.wavelengths, self.planck_fit, 'g--', linewidth=1.2, 
                            alpha=0.7, label=f'Planck fit (T={self.planck_temperature:.1f} K, R²={self.planck_r_squared:.3f})')
        
        if self.show_peak_var.get() and spectrum_to_plot is not None:
            peak_idx = np.argmax(spectrum_to_plot)
            self.peak_wavelength = self.wavelengths[peak_idx]
            self.peak_intensity = spectrum_to_plot[peak_idx]
            
            self.ax.plot(self.peak_wavelength, self.peak_intensity, 'ro', 
                       markersize=10, markeredgewidth=2, markeredgecolor='red', 
                       markerfacecolor='yellow', label=f'Peak: {self.peak_wavelength:.1f} nm')
            
            self.ax.annotate(f'{self.peak_wavelength:.1f} nm\n{self.peak_intensity:.0f} counts',
                           xy=(self.peak_wavelength, self.peak_intensity),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        self.ax.legend()
        self.ax.set_xlim([min(self.wavelengths), max(self.wavelengths)])
        
        try:
            y_max = float(self.y_axis_max_var.get())
            if y_max < 1:
                y_max = 70000.0
        except (ValueError, tk.TclError):
            y_max = self.y_axis_max
        self.ax.set_ylim([0, y_max])
        
        self.canvas.draw()
    
    def planck_function(self, wavelength_nm, temperature, scale_factor):

        is_scalar = np.isscalar(wavelength_nm)
        wavelength_nm = np.atleast_1d(wavelength_nm)

        wavelength_m = wavelength_nm * 1e-9
  
        h = constants.h
        c = constants.c
        k = constants.k
        
        exponent = h * c / (wavelength_m * k * temperature)
        
        result = np.zeros_like(wavelength_m, dtype=float)
        valid_mask = exponent <= 700
        
        if np.any(valid_mask):
            numerator = 2 * np.pi * h * c**2
            valid_wavelengths = wavelength_m[valid_mask]
            valid_exponents = exponent[valid_mask]
            denominator = valid_wavelengths**5 * (np.exp(valid_exponents) - 1)
            
            result[valid_mask] = scale_factor * numerator / denominator
        
        if is_scalar:
            return float(result[0])
        return result
    
    def fit_planck(self):

        if self.current_spectrum is None:
            messagebox.showwarning("Warning", "No spectrum data available. Please take a measurement first.")
            return
        
        spectrum_to_fit = self.current_spectrum.copy()
        if self.dark_subtraction_enabled and self.dark_spectrum is not None:
            spectrum_to_fit = spectrum_to_fit - self.dark_spectrum
            spectrum_to_fit = np.maximum(spectrum_to_fit, 0)
        
        valid_indices = spectrum_to_fit > 0
        if np.sum(valid_indices) < 10:
            messagebox.showerror("Error", "Not enough valid data points for fitting. The spectrum may be too weak.")
            return
        
        wavelengths_fit = np.array(self.wavelengths)[valid_indices]
        spectrum_fit = spectrum_to_fit[valid_indices]
        
        max_idx = np.argmax(spectrum_fit)
        peak_wavelength = wavelengths_fit[max_idx] * 1e-9  

        initial_temp = 2.898e-3 / peak_wavelength if peak_wavelength > 0 else 3000.0
        initial_temp = np.clip(initial_temp, 1000.0, 10000.0)  
        
       
        initial_scale = np.max(spectrum_fit) / self.planck_function(wavelengths_fit[max_idx], initial_temp, 1.0)
        initial_scale = np.clip(initial_scale, 1e-20, 1e20)
        
        try:
            self.status_label.config(text="Fitting Planck curve...", foreground="orange")
            self.root.update()
            
            def fit_func(wl, T, scale):
                return self.planck_function(wl, T, scale)
            
            popt, pcov = curve_fit(
                fit_func,
                wavelengths_fit,
                spectrum_fit,
                p0=[initial_temp, initial_scale],
                bounds=([1000.0, 1e-30], [10000.0, 1e30]),
                maxfev=5000
            )
            
            fitted_temp = popt[0]
            fitted_scale = popt[1]
            
            self.planck_fit = self.planck_function(np.array(self.wavelengths), fitted_temp, fitted_scale)
            
            if not isinstance(self.planck_fit, np.ndarray):
                self.planck_fit = np.array(self.planck_fit)
            if len(self.planck_fit) != len(self.wavelengths):
                self.planck_fit = self.planck_function(np.array(self.wavelengths), fitted_temp, fitted_scale)
   
            y_pred = self.planck_function(wavelengths_fit, fitted_temp, fitted_scale)
            ss_res = np.sum((spectrum_fit - y_pred) ** 2)
            ss_tot = np.sum((spectrum_fit - np.mean(spectrum_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            self.planck_temperature = fitted_temp
            self.planck_r_squared = r_squared
            self.planck_enabled = True

            if r_squared < 0.5:
                messagebox.showwarning(
                    "Poor Fit", 
                    f"Planck fit completed but quality is low (R² = {r_squared:.3f}).\n"
                    f"Temperature: {fitted_temp:.1f} K\n"
                    f"The spectrum may not be a blackbody or may have significant noise."
                )
            else:
                messagebox.showinfo(
                    "Fit Complete",
                    f"Planck curve fitted successfully!\n\n"
                    f"Temperature: {fitted_temp:.1f} K ({fitted_temp - 273.15:.1f} °C)\n"
                    f"Fit Quality (R²): {r_squared:.3f}\n\n"
                    f"The fitted curve is now displayed on the plot."
                )
            
            self.planck_status_label.config(
                text=f"Planck: T={fitted_temp:.1f} K, R²={r_squared:.3f}",
                foreground="green" if r_squared > 0.7 else "orange"
            )
            self.status_label.config(text="Planck fit complete", foreground="green")
            
            self.planck_show_var.set(True)
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Fit Error", f"Failed to fit Planck curve:\n{str(e)}")
            self.status_label.config(text="Planck fit failed", foreground="red")
            self.planck_fit = None
            self.planck_temperature = None
            self.planck_r_squared = None
    
    def toggle_planck_display(self):
        self.update_plot()
    
    def toggle_peak_display(self):
        self.show_peak = self.show_peak_var.get()
        self.update_plot()
    
    def toggle_crosshair(self):

        self.show_crosshair = self.show_crosshair_var.get()
        if not self.show_crosshair:
 
            if self.crosshair_vline is not None:
                self.crosshair_vline.remove()
                self.crosshair_vline = None
            if self.crosshair_hline is not None:
                self.crosshair_hline.remove()
                self.crosshair_hline = None
            if self.crosshair_text is not None:
                self.crosshair_text.remove()
                self.crosshair_text = None
            self.canvas.draw()
    
    def on_mouse_move(self, event):
 
        if not self.show_crosshair_var.get():
            return
        
        if event.inaxes != self.ax:

            if self.crosshair_vline is not None:
                self.crosshair_vline.remove()
                self.crosshair_vline = None
            if self.crosshair_hline is not None:
                self.crosshair_hline.remove()
                self.crosshair_hline = None
            if self.crosshair_text is not None:
                self.crosshair_text.remove()
                self.crosshair_text = None
            self.canvas.draw()
            return

        x = event.xdata
        y = event.ydata
        
        if x is None or y is None:
            return
        
        if self.crosshair_vline is not None:
            self.crosshair_vline.remove()
        if self.crosshair_hline is not None:
            self.crosshair_hline.remove()
        if self.crosshair_text is not None:
            self.crosshair_text.remove()
        
        self.crosshair_vline = self.ax.axvline(x, color='blue', linestyle='--', linewidth=1, alpha=0.7)

        self.crosshair_hline = self.ax.axhline(y, color='blue', linestyle='--', linewidth=1, alpha=0.7)
        
        intensity = 0
        if self.current_spectrum is not None and len(self.wavelengths) > 0:

            closest_idx = np.argmin(np.abs(np.array(self.wavelengths) - x))
            spectrum_disp, _ = self._compute_display_spectrum()
            if spectrum_disp is not None and 0 <= closest_idx < len(spectrum_disp):
                intensity = spectrum_disp[closest_idx]

        self.crosshair_text = self.ax.text(0.02, 0.98, 
                                          f'λ: {x:.2f} nm\nI: {intensity:.0f} counts',
                                          transform=self.ax.transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                          fontsize=9)
        
        self.canvas.draw()
    
    def save_spectrum(self):

        if self.current_spectrum is None:
            messagebox.showwarning("Warning", "No spectrum data to save. Please take a measurement first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                spectrum_to_save, _ = self._compute_display_spectrum()
                if spectrum_to_save is None:
                    messagebox.showwarning("Warning", "No spectrum data to save. Please take a measurement first.")
                    return

                with open(filename, 'w') as f:
                    f.write("Wavelength (nm),Intensity (counts)\n")
                    for wl, intensity in zip(self.wavelengths, spectrum_to_save):
                        f.write(f"{wl},{intensity}\n")
                
                messagebox.showinfo("Success", f"Spectrum saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save spectrum:\n{str(e)}")
    
    def export_plot_png(self):
        """Export the current plot as a PNG image."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save PNG:\n{str(e)}")

    def load_sensitivity_curve(self):
        """Load a sensitivity curve from CSV (nm, counts/response)."""
        filename = filedialog.askopenfilename(
            title="Select sensitivity CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            data = np.loadtxt(filename, delimiter=",", comments="#", skiprows=1)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("CSV must have at least two columns (wavelength, sensitivity).")

            wl = np.array(data[:, 0], dtype=float)
            sens = np.array(data[:, 1], dtype=float)

            order = np.argsort(wl)
            wl = wl[order]
            sens = sens[order]

            sens[sens <= 0] = np.nan

            self.sensitivity_wavelengths = wl
            self.sensitivity_values = sens
            self.sensitivity_loaded = True

            if hasattr(self, "sensitivity_status_label"):
                self.sensitivity_status_label.config(
                    text=f"Sensitivity: loaded ({len(wl)} pts)",
                    foreground="green",
                )

            if getattr(self, "sensitivity_apply_var", None) is not None and self.sensitivity_apply_var.get():
                self.update_plot()

        except Exception as e:
            self.sensitivity_wavelengths = None
            self.sensitivity_values = None
            self.sensitivity_loaded = False
            if hasattr(self, "sensitivity_status_label"):
                self.sensitivity_status_label.config(
                    text="Sensitivity: error loading",
                    foreground="red",
                )
            messagebox.showerror("Error", f"Failed to load sensitivity CSV:\n{str(e)}")
    
    def on_closing(self):
        
        self.is_live = False
        if self.handle is not None:
            try:
                AVS_StopMeasure(self.handle)
                AVS_Done()
            except:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrometerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
