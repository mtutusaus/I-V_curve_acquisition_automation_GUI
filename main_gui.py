
"""
Measurement Suite (I–V + Junction Temperature)
UI update:
 • Temperature tab: Linear/Quadratic model; A/B/C entries widened.
 • Heating period no longer writes TXT automatically; adds 'Export CSV' button.
 • Duration input in minutes (internally converted to seconds). Interval stays in seconds.
 • UI labels: 'VCE SMU...' → 'Bias SMU'; 'VCE source current (A)' → 'Bias voltage (mA)';
   'VCE compliance (V)' → 'Bias compliance (V)'. (Variable names unchanged in code.)
 • Single measurement: buttons to copy Voltage or Temperature separately to clipboard.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import json
from threading import Thread, Event
from time import sleep
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# External instrument drivers (as in your project)
import pyvisa
from pymeasure.instruments.keithley import Keithley2400
from tek371 import Tek371

APP_VERSION = "1.2"

# ----------------------- Utility -----------------------

def safe_float(entry: ttk.Entry, default: float) -> float:
    try:
        return float(entry.get())
    except Exception:
        return default

# ------------------ I–V Tab (unchanged from previous feature set) ------------------
class IVTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.root = master.winfo_toplevel()
        self.tek371 = None
        self.keithley = None
        self.measurement_running = False
        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        title = ttk.Label(main_frame, text="I–V Measurement System", font=('Helvetica', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        left_frame = ttk.Frame(main_frame)
        right_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=tk.N+tk.W, padx=(0, 10))
        right_frame.grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ===== Device Connection =====
        conn_frame = ttk.LabelFrame(left_frame, text="Device Connection", padding="10")
        conn_frame.grid(row=0, column=0, sticky=tk.W+tk.E, pady=5)

        ttk.Button(conn_frame, text="Scan GPIB Bus", command=self.scan_gpib).grid(row=0, column=0, pady=5, sticky=tk.W)
        ttk.Label(conn_frame, text="Available devices:").grid(row=0, column=1, sticky=tk.W)
        self.gpib_text = scrolledtext.ScrolledText(conn_frame, height=3, width=50, state='disabled')
        self.gpib_text.grid(row=1, column=0, columnspan=4, pady=5, sticky=tk.W)

        addrs_label = ttk.Label(conn_frame, text="Instrument Addresses", font=('Helvetica', 10, 'bold'))
        addrs_label.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(8, 2))
        ttk.Label(conn_frame, text="Tek371:").grid(row=3, column=0, sticky=tk.W)
        self.tek_addr = ttk.Entry(conn_frame, width=25)
        self.tek_addr.insert(0, "GPIB::23")
        self.tek_addr.grid(row=3, column=1, sticky=tk.W)
        ttk.Label(conn_frame, text="Keithley 2400:").grid(row=3, column=2, sticky=tk.W)
        self.keithley_addr = ttk.Entry(conn_frame, width=20)
        self.keithley_addr.insert(0, "GPIB::24")
        self.keithley_addr.grid(row=3, column=3, sticky=tk.W)

        ttk.Label(conn_frame, text="Tek371:").grid(row=4, column=0, sticky=tk.W, pady=(8, 5))
        ttk.Button(conn_frame, text="Connect", command=self.connect_tek).grid(row=4, column=1, padx=5, sticky=tk.W)
        self.tek_status = ttk.Label(conn_frame, text="Not connected", foreground='gray', width=12)
        self.tek_status.grid(row=4, column=2, sticky=tk.W)

        ttk.Label(conn_frame, text="Keithley 2400:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Button(conn_frame, text="Connect", command=self.connect_keithley).grid(row=5, column=1, padx=5, sticky=tk.W)
        self.keithley_status = ttk.Label(conn_frame, text="Not connected", foreground='gray', width=12)
        self.keithley_status.grid(row=5, column=2, sticky=tk.W)

        # ===== Measurement Parameters =====
        param_frame = ttk.LabelFrame(left_frame, text="Measurement Parameters", padding="10")
        param_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        params = [
            ("SMU Source Voltage (V):", "20"),
            ("SMU Compliance Current (A):", "1e-3"),
            ("Tracer Horizontal Scale (V/div):", "200e-3"),
            ("Tracer Vertical Scale (A/div):", "5"),
            ("Tracer VCE Percentage (%):", "100"),
            ("Tracer Peak Power (W):", "300"),
        ]
        self.param_entries = {}
        for i, (label, default) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(param_frame, width=15)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.param_entries[label] = entry

        # ===== File Settings =====
        file_frame = ttk.LabelFrame(left_frame, text="File Settings", padding="10")
        file_frame.grid(row=2, column=0, sticky=tk.W, pady=5)
        self.banner_label = tk.Label(left_frame,
            text="Connect Tek371 and Keithley 2400 to enable Start measurement button",
            bg="#FFC107", fg="black")
        self.banner_label.grid(row=3, column=0, sticky=tk.E+tk.W, pady=(5, 5))

        ttk.Label(file_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.folder_entry = ttk.Entry(file_frame, width=35)
        self.folder_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)

        file_params = [
            ("DUT Name:", "H40ER5S"),
            ("Device ID:", "dev10"),
            ("Vge Applied (V):", "20"),
            ("Temperature (°C):", "25"),
            ("Number of Curves:", "10"),
        ]
        self.file_entries = {}
        for i, (label, default) in enumerate(file_params, start=1):
            ttk.Label(file_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(file_frame, width=15)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.file_entries[label] = entry

        # ===== Control Buttons =====
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=4, column=0, pady=10, sticky=tk.W+tk.E)
        self.start_btn = ttk.Button(btn_frame, text="Start Measurement", command=self.start_measurement)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_measurement, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Clear Plot", command=self.clear_plot).grid(row=0, column=2, padx=5)

        # ===== Status =====
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding="10")
        status_frame.grid(row=5, column=0, sticky=tk.W+tk.E, pady=5)
        status_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor='w', justify='left')
        self.status_label.grid(row=0, column=0, sticky=tk.E+tk.W)
        self.progress = ttk.Progressbar(status_frame, length=400, mode='determinate')
        self.progress.grid(row=1, column=0, sticky=tk.E+tk.W, pady=(6, 0))
        status_frame.bind('<Configure>', self._on_status_resize)

        # ===== Right Plot =====
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        plot_frame = ttk.Frame(right_frame)
        plot_frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        self.ax.set_title('I–V Measurement Data')
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.fig.tight_layout()

        self.update_banner_visibility()

    def _on_status_resize(self, event):
        try:
            self.status_label.configure(wraplength=max(event.width - 20, 100))
        except Exception:
            pass

    def collect_settings(self):
        addrs = {
            "tek371": self.tek_addr.get(),
            "keithley2400": self.keithley_addr.get(),
        }
        meas = {label: self.param_entries[label].get() for label in self.param_entries}
        file_set = {
            "output_folder": self.folder_entry.get(),
            "DUT Name:": self.file_entries["DUT Name:"].get(),
            "Device ID:": self.file_entries["Device ID:"].get(),
            "Vge Applied (V):": self.file_entries["Vge Applied (V):"].get(),
            "Temperature (°C):": self.file_entries["Temperature (°C):"].get(),
            "Number of Curves:": self.file_entries["Number of Curves:"].get(),
        }
        base_filename = f"{file_set['DUT Name:']}_{file_set['Device ID:']}_{file_set['Vge Applied (V):']}V_{file_set['Temperature (°C):']}C"
        return {
            "addresses": addrs,
            "measurement": meas,
            "file": file_set,
            "base_filename_example": base_filename,
        }

    def apply_settings(self, settings: dict):
        try:
            if "addresses" in settings:
                addrs = settings["addresses"]
                if "tek371" in addrs:
                    self.tek_addr.delete(0, tk.END)
                    self.tek_addr.insert(0, addrs["tek371"])
                if "keithley2400" in addrs:
                    self.keithley_addr.delete(0, tk.END)
                    self.keithley_addr.insert(0, addrs["keithley2400"])
            if "measurement" in settings:
                meas = settings["measurement"]
                for label, entry in self.param_entries.items():
                    if label in meas:
                        entry.delete(0, tk.END)
                        entry.insert(0, str(meas[label]))
            if "file" in settings:
                fs = settings["file"]
                if "output_folder" in fs:
                    self.folder_entry.delete(0, tk.END)
                    self.folder_entry.insert(0, fs["output_folder"])
                for label, entry in self.file_entries.items():
                    if label in fs:
                        entry.delete(0, tk.END)
                        entry.insert(0, str(fs[label]))
            self.update_status("Settings applied to I–V tab")
        except Exception as e:
            messagebox.showerror("Apply Settings Error", str(e))

    def update_banner_visibility(self):
        try:
            both_connected = (self.tek_status.cget('text') == 'Connected' and self.keithley_status.cget('text') == 'Connected')
            if both_connected:
                self.banner_label.grid_remove()
            else:
                self.banner_label.grid(row=3, column=0, sticky=tk.E+tk.W, pady=(5, 5))
        except Exception:
            pass

    def scan_gpib(self):
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            gpib_devices = [r for r in resources if 'GPIB' in r]
            self.gpib_text.config(state='normal')
            self.gpib_text.delete(1.0, tk.END)
            if gpib_devices:
                self.gpib_text.insert(tk.END, '\n'.join(gpib_devices))
                self.update_status(f"Found {len(gpib_devices)} GPIB device(s)")
            else:
                self.gpib_text.insert(tk.END, "No GPIB devices found")
                self.update_status("No GPIB devices found")
            self.gpib_text.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error scanning GPIB", str(e))

    def connect_tek(self):
        try:
            self.update_status("Connecting to Tek371...")
            addr = self.tek_addr.get()
            self.tek371 = Tek371(addr)
            try:
                idn = self.tek371.id_string()
            except Exception:
                idn = None
            if not idn:
                raise RuntimeError("Tek371 did not respond to ID query")
            self.tek_status.config(text="Connected", foreground='green')
            self.update_status(f"Tek371 connected: {idn}")
            self.update_banner_visibility()
        except Exception as e:
            self.tek_status.config(text="Error", foreground='red')
            messagebox.showerror("Tek371 Connection Error", str(e))
            self.update_status("Tek371 connection failed")
            self.update_banner_visibility()

    def connect_keithley(self):
        try:
            self.update_status("Connecting to Keithley 2400...")
            addr = self.keithley_addr.get()
            self.keithley = Keithley2400(addr)
            idn = None
            try:
                idn = getattr(self.keithley, 'idn', None) or getattr(self.keithley, 'id', None)
                if not idn:
                    idn = self.keithley.ask("*IDN?")
            except Exception:
                idn = None
            if not idn:
                raise RuntimeError("Keithley 2400 did not respond to *IDN? or idn")
            self.keithley_status.config(text="Connected", foreground='green')
            self.update_status(f"Keithley connected: {idn}")
            self.update_banner_visibility()
        except Exception as e:
            self.keithley_status.config(text="Error", foreground='red')
            messagebox.showerror("Keithley 2400 Connection Error", str(e))
            self.update_status("Keithley connection failed")
            self.update_banner_visibility()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def _update_progress(self, percent: float):
        self.progress['value'] = percent
        self.root.update_idletasks()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        self.ax.set_title('I–V Measurement Data')
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()
        self.update_status("Plot cleared")
        self._update_progress(0)

    def start_measurement(self):
        if self.tek371 is None or self.keithley is None:
            messagebox.showerror("Error", "Please connect both devices before starting!")
            return
        folder = self.folder_entry.get()
        if not folder:
            messagebox.showerror("Error", "Please select an output folder!")
            return
        self.measurement_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        thread = Thread(target=self.perform_measurement, daemon=True)
        thread.start()

    def stop_measurement(self):
        self.measurement_running = False
        self.update_status("Stopping measurement...")

    def perform_measurement(self):
        try:
            smu_voltage = float(self.param_entries["SMU Source Voltage (V):"].get())
            compliance = float(self.param_entries["SMU Compliance Current (A):"].get())
            h_scale = float(self.param_entries["Tracer Horizontal Scale (V/div):"].get())
            v_scale = float(self.param_entries["Tracer Vertical Scale (A/div):"].get())
            vce_pct = float(self.param_entries["Tracer VCE Percentage (%):"].get())
            peak_power = int(self.param_entries["Tracer Peak Power (W):"].get())
            folder = self.folder_entry.get()
            dut = self.file_entries["DUT Name:"].get()
            dev_id = self.file_entries["Device ID:"].get()
            vge = self.file_entries["Vge Applied (V):"].get()
            temp = self.file_entries["Temperature (°C):"].get()
            num_curves = int(self.file_entries["Number of Curves:"].get())
            base_filename = f"{dut}_{dev_id}_{vge}V_{temp}C"
            os.makedirs(folder, exist_ok=True)

            # Configure SMU
            self.update_status("Configuring SMU...")
            self.keithley.reset()
            self.keithley.write("*CLS")
            self.keithley.write("*SRE 0")
            self.keithley.use_front_terminals()
            self.keithley.source_mode = "voltage"
            self.keithley.source_voltage = smu_voltage
            self.keithley.compliance_current = compliance
            sleep(0.5)

            # Configure Tek371
            self.update_status("Configuring Tek371...")
            self.tek371.initialize()
            self.tek371.set_peak_power(peak_power)
            self.tek371.set_step_number(0)
            self.tek371.set_step_voltage(200e-3)
            self.tek371.set_step_offset(0)
            self.tek371.enable_srq_event()
            self.tek371.set_horizontal("COL", h_scale)
            self.tek371.set_vertical(v_scale)
            self.tek371.set_display_mode("STO")
            sleep(0.5)

            self.update_status("Starting measurements...")
            self.keithley.enable_source()

            for i in range(1, num_curves + 1):
                if not self.measurement_running:
                    self.update_status("Measurement stopped by user")
                    break
                self._update_progress(((i - 1) / num_curves) * 100)
                self.update_status(f"Measuring curve {i}/{num_curves}...")

                self.tek371.set_collector_supply(vce_pct)
                self.tek371.set_measurement_mode("SWE")
                if self.tek371.wait_for_srq(timeout_s=60.0):
                    self.update_status(f"Sweep {i}/{num_curves} complete, reading data...")
                else:
                    raise TimeoutError(f"Sweep {i}/{num_curves} timeout")

                filename = os.path.join(folder, f"{base_filename}_{i}.csv")
                self.tek371.read_curve(filename)

                try:
                    data = pd.read_csv(filename)
                    self.ax.plot(data.iloc[:, 0], data.iloc[:, 1], alpha=0.5, linewidth=1, color='blue')
                    self.fig.tight_layout()
                    self.root.after(0, self.canvas.draw)
                except Exception as e:
                    print(f"Plot error: {e}")

                self.tek371.discard_and_disable_all_events()
                self.tek371.enable_srq_event()
                self._update_progress((i / num_curves) * 100)

            # Cleanup
            self.keithley.disable_source()
            self.keithley.beep(4000, 2)
            self.tek371.disable_srq_event()

            if self.measurement_running:
                self.update_status("Measurement complete!")
                self._update_progress(100)
        except Exception as e:
            self.update_status("Measurement error")
            messagebox.showerror("Measurement Error", str(e))
            try:
                if self.keithley:
                    self.keithley.disable_source()
            except Exception:
                pass
        finally:
            self.measurement_running = False
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')

# ------------------ Junction Temperature Tab ------------------
class TemperatureTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.root = master.winfo_toplevel()
        # Two Keithley 2400s for this tab
        self.smu_vce = None  # current source + voltage measurement (Bias SMU in UI)
        self.smu_vge = None  # gate voltage source
        self.measure_thread = None
        self.stop_event = Event()
        self.time_data = []
        self.tj_data = []
        self.last_voltage = None
        self.last_tj = None
        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        title = ttk.Label(main_frame, text="Junction Temperature Measurement", font=('Helvetica', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        left = ttk.Frame(main_frame)
        right = ttk.Frame(main_frame)
        left.grid(row=1, column=0, sticky=tk.N+tk.W, padx=(0, 10))
        right.grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # --- Connection section ---
        conn = ttk.LabelFrame(left, text="Device Connection (Two Keithley 2400)", padding="10")
        conn.grid(row=0, column=0, sticky=tk.W+tk.E)
        ttk.Button(conn, text="Scan GPIB Bus", command=self.scan_gpib).grid(row=0, column=0, pady=5, sticky=tk.W)
        self.gpib_text = scrolledtext.ScrolledText(conn, height=3, width=50, state='disabled')
        self.gpib_text.grid(row=1, column=0, columnspan=4, pady=5, sticky=tk.W)

        ttk.Label(conn, text="Bias SMU:").grid(row=2, column=0, sticky=tk.W)
        self.vce_addr = ttk.Entry(conn, width=18)
        self.vce_addr.insert(0, "GPIB::25")
        self.vce_addr.grid(row=2, column=1, sticky=tk.W)
        ttk.Button(conn, text="Connect", command=self.connect_vce).grid(row=2, column=2, padx=5)
        self.vce_status = ttk.Label(conn, text="Not connected", foreground='gray')
        self.vce_status.grid(row=2, column=3, sticky=tk.W)

        ttk.Label(conn, text="VGE SMU:").grid(row=3, column=0, sticky=tk.W)
        self.vge_addr = ttk.Entry(conn, width=18)
        self.vge_addr.insert(0, "GPIB::24")
        self.vge_addr.grid(row=3, column=1, sticky=tk.W)
        ttk.Button(conn, text="Connect", command=self.connect_vge).grid(row=3, column=2, padx=5)
        self.vge_status = ttk.Label(conn, text="Not connected", foreground='gray')
        self.vge_status.grid(row=3, column=3, sticky=tk.W)

        # --- Heating period (upper) ---
        hp = ttk.LabelFrame(left, text="Heating Period (15 min typical)", padding="10")
        hp.grid(row=1, column=0, sticky=tk.W+tk.E, pady=(8, 4))

        # Export/Import + Export CSV row
        btnrow = ttk.Frame(hp)
        btnrow.grid(row=0, column=0, columnspan=3, sticky=tk.W)
        ttk.Button(btnrow, text="Export Settings", command=self._export_all).grid(row=0, column=0, padx=(0,6))
        ttk.Button(btnrow, text="Import Settings", command=self._import_all).grid(row=0, column=1, padx=(0,12))
        ttk.Button(btnrow, text="Export CSV", command=self.export_csv).grid(row=0, column=2)

        ttk.Label(hp, text="Output Folder:").grid(row=1, column=0, sticky=tk.W)
        self.hp_folder = ttk.Entry(hp, width=35)
        self.hp_folder.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(hp, text="Browse", command=lambda: self._browse_to(self.hp_folder)).grid(row=1, column=2)

        ttk.Label(hp, text="File Prefix:").grid(row=2, column=0, sticky=tk.W)
        self.hp_prefix = ttk.Entry(hp, width=20)
        self.hp_prefix.insert(0, "temp_log")
        self.hp_prefix.grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="Hot-plate Setpoint (°C):").grid(row=3, column=0, sticky=tk.W)
        self.hp_setpoint = ttk.Entry(hp, width=10)
        self.hp_setpoint.insert(0, "130")
        self.hp_setpoint.grid(row=3, column=1, sticky=tk.W, padx=5)

        # Model selection + coefficients (wide entries)
        ttk.Label(hp, text="Tj Model:").grid(row=4, column=0, sticky=tk.W)
        self.hp_model = ttk.Combobox(hp, width=24, state='readonly', values=["Linear (A + B·V)", "Quadratic (A + B·V + C·V²)"])
        self.hp_model.set("Linear (A + B·V)")
        self.hp_model.grid(row=4, column=1, sticky=tk.W, padx=5)
        self.hp_model.bind('<<ComboboxSelected>>', self._on_model_change)

        # Coefficients row with larger width
        ttk.Label(hp, text="A (intercept):").grid(row=5, column=0, sticky=tk.W)
        self.hp_A = ttk.Entry(hp, width=30)
        self.hp_A.insert(0, "357.090847511229")
        self.hp_A.grid(row=5, column=1, sticky=tk.E+tk.W, padx=5)

        ttk.Label(hp, text="B (slope vs V):").grid(row=6, column=0, sticky=tk.W)
        self.hp_B = ttk.Entry(hp, width=30)
        self.hp_B.insert(0, "-532.214573058354")
        self.hp_B.grid(row=6, column=1, sticky=tk.E+tk.W, padx=5)

        self.row_c_label = ttk.Label(hp, text="C (V² term):")
        self.row_c_entry = ttk.Entry(hp, width=30)
        self.row_c_entry.insert(0, "0.0")
        # initially hidden for Linear; will place at row 7 when needed

        hp.columnconfigure(1, weight=1)  # allow column with entries to expand

        # Timing & instrument params (with UI label changes)
        ttk.Label(hp, text="Duration (min):").grid(row=8, column=0, sticky=tk.W)
        self.hp_duration = ttk.Entry(hp, width=8)
        self.hp_duration.insert(0, "15")
        self.hp_duration.grid(row=8, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="Interval (s):").grid(row=9, column=0, sticky=tk.W)
        self.hp_interval = ttk.Entry(hp, width=8)
        self.hp_interval.insert(0, "1")
        self.hp_interval.grid(row=9, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="Bias voltage (mA):").grid(row=10, column=0, sticky=tk.W)
        self.hp_Ic = ttk.Entry(hp, width=10)
        self.hp_Ic.insert(0, "150")
        self.hp_Ic.grid(row=10, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="Bias compliance (V):").grid(row=11, column=0, sticky=tk.W)
        self.hp_Vcomp = ttk.Entry(hp, width=10)
        self.hp_Vcomp.insert(0, "2")
        self.hp_Vcomp.grid(row=11, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="NPLC:").grid(row=12, column=0, sticky=tk.W)
        self.hp_nplc = ttk.Entry(hp, width=10)
        self.hp_nplc.insert(0, "10")
        self.hp_nplc.grid(row=12, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="Voltage range (V):").grid(row=13, column=0, sticky=tk.W)
        self.hp_vrange = ttk.Entry(hp, width=10)
        self.hp_vrange.insert(0, "2")
        self.hp_vrange.grid(row=13, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="VGE source voltage (V):").grid(row=14, column=0, sticky=tk.W)
        self.hp_Vge = ttk.Entry(hp, width=10)
        self.hp_Vge.insert(0, "15")
        self.hp_Vge.grid(row=14, column=1, sticky=tk.W, padx=5)

        ttk.Label(hp, text="VGE compliance current (A):").grid(row=15, column=0, sticky=tk.W)
        self.hp_Icomp = ttk.Entry(hp, width=10)
        self.hp_Icomp.insert(0, "1e-3")
        self.hp_Icomp.grid(row=15, column=1, sticky=tk.W, padx=5)

        btns = ttk.Frame(hp)
        btns.grid(row=16, column=0, columnspan=3, pady=(8, 0), sticky=tk.W)
        self.hp_start = ttk.Button(btns, text="Start Heating Measurement", command=self.start_heating)
        self.hp_start.grid(row=0, column=0, padx=5)
        self.hp_stop = ttk.Button(btns, text="Stop", state='disabled', command=self.stop_heating)
        self.hp_stop.grid(row=0, column=1, padx=5)

        self.hp_status = ttk.Label(hp, text="Ready", relief=tk.SUNKEN, anchor='w')
        self.hp_status.grid(row=17, column=0, columnspan=3, sticky=tk.E+tk.W, pady=(6, 0))
        self.hp_progress = ttk.Progressbar(hp, length=350, mode='determinate')
        self.hp_progress.grid(row=18, column=0, columnspan=3, sticky=tk.E+tk.W)

        # --- Single measurement (lower) ---
        sm = ttk.LabelFrame(left, text="Single Measurement (stable)", padding="10")
        sm.grid(row=2, column=0, sticky=tk.W+tk.E)

        ttk.Label(sm, text="Readings in buffer (N):").grid(row=0, column=0, sticky=tk.W)
        self.sm_N = ttk.Entry(sm, width=8)
        self.sm_N.insert(0, "10")
        self.sm_N.grid(row=0, column=1, sticky=tk.W, padx=5)

        self.sm_button = ttk.Button(sm, text="Measure Once", command=self.single_measurement)
        self.sm_button.grid(row=1, column=0, padx=5, pady=(5, 0))

        # Result + copy buttons
        res_row = ttk.Frame(sm)
        res_row.grid(row=1, column=1, columnspan=2, sticky=tk.W)
        self.sm_result = ttk.Label(res_row, text="Result: —", anchor='w')
        self.sm_result.grid(row=0, column=0, sticky=tk.W)
        ttk.Button(res_row, text="Copy Voltage", command=self.copy_voltage).grid(row=0, column=1, padx=(8,4))
        ttk.Button(res_row, text="Copy Temperature", command=self.copy_temperature).grid(row=0, column=2)

        # --- Right plot ---
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        plot = ttk.Frame(right)
        plot.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)

        self.tfig, self.tax = plt.subplots(figsize=(8, 5), dpi=100)
        self.tax.set_xlabel('Time (s)')
        self.tax.set_ylabel('Tj (°C)')
        self.tax.set_title('Junction Temperature vs Time')
        self.tax.grid(True)
        self.tcanvas = FigureCanvasTkAgg(self.tfig, master=plot)
        self.tcanvas.get_tk_widget().grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tfig.tight_layout()

        # Ensure C visibility reflects model
        self._on_model_change()

    # ---------- unified settings triggers ----------
    def _export_all(self):
        if hasattr(self.root, 'export_all_settings'):
            self.root.export_all_settings()

    def _import_all(self):
        if hasattr(self.root, 'import_all_settings'):
            self.root.import_all_settings()

    # ---------- Model UI handling ----------
    def _on_model_change(self, event=None):
        mode = self.hp_model.get()
        if mode.startswith("Quadratic"):
            # show C row at row=7
            self.row_c_label.grid(row=7, column=0, sticky=tk.W)
            self.row_c_entry.grid(row=7, column=1, sticky=tk.E+tk.W, padx=5)
        else:
            # hide C row
            self.row_c_label.grid_remove()
            self.row_c_entry.grid_remove()

    # ---------- Connection helpers ----------
    def scan_gpib(self):
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            gpib = [r for r in resources if 'GPIB' in r]
            self.gpib_text.config(state='normal')
            self.gpib_text.delete(1.0, tk.END)
            if gpib:
                self.gpib_text.insert(tk.END, '\n'.join(gpib))
            else:
                self.gpib_text.insert(tk.END, "No GPIB devices found")
            self.gpib_text.config(state='disabled')
        except Exception as e:
            messagebox.showerror("GPIB Scan Error", str(e))

    def connect_vce(self):
        try:
            addr = self.vce_addr.get()
            self.smu_vce = Keithley2400(addr)
            self.smu_vce.reset()
            self.smu_vce.use_front_terminals()
            self.vce_status.config(text="Connected", foreground='green')
        except Exception as e:
            self.vce_status.config(text="Error", foreground='red')
            messagebox.showerror("Bias SMU Connection Error", str(e))

    def connect_vge(self):
        try:
            addr = self.vge_addr.get()
            self.smu_vge = Keithley2400(addr)
            self.smu_vge.reset()
            self.smu_vge.use_front_terminals()
            self.vge_status.config(text="Connected", foreground='green')
        except Exception as e:
            self.vge_status.config(text="Error", foreground='red')
            messagebox.showerror("VGE SMU Connection Error", str(e))

    # ---------- Heating Period ----------
    def start_heating(self):
        if self.smu_vce is None or self.smu_vge is None:
            messagebox.showerror("Error", "Connect both Keithley 2400 instruments first.")
            return
        # no need to select output folder to run; only needed if exporting CSV
        self.time_data.clear()
        self.tj_data.clear()
        self.hp_progress['value'] = 0
        self.stop_event.clear()
        self.hp_start.config(state='disabled')
        self.hp_stop.config(state='normal')
        self.hp_status.config(text="Configuring instruments…")
        self.measure_thread = Thread(target=self._run_heating_measurement, daemon=True)
        self.measure_thread.start()

    def stop_heating(self):
        self.stop_event.set()
        self.hp_status.config(text="Stopping…")

    def _compute_tj(self, V: float) -> float:
        A = safe_float(self.hp_A, 0.0)
        B = safe_float(self.hp_B, 0.0)
        mode = self.hp_model.get()
        if mode.startswith("Quadratic"):
            C = safe_float(self.row_c_entry, 0.0)
            return A + B * V + C * (V ** 2)
        else:
            return A + B * V

    def _run_heating_measurement(self):
        # Read UI parameters
        duration_min = safe_float(self.hp_duration, 15.0)
        duration = int(max(0.0, duration_min) * 60.0)  # convert minutes → seconds
        interval = max(0.2, safe_float(self.hp_interval, 1.0))
        setpoint = self.hp_setpoint.get().strip() or "NA"
        # instrument params
        Ic_mA = safe_float(self.hp_Ic, 150.0)  # UI in mA
        Ic = Ic_mA / 1000.0                    # internal in A
        Vcomp = safe_float(self.hp_Vcomp, 2.0)
        nplc = safe_float(self.hp_nplc, 10)
        vrange = safe_float(self.hp_vrange, 2.0)
        Vge = safe_float(self.hp_Vge, 15.0)
        Icomp = safe_float(self.hp_Icomp, 1e-3)

        try:
            # Configure Bias SMU (current source, voltage measure)
            self.smu_vce.apply_current(Ic, Vcomp)
            self.smu_vce.measure_voltage(nplc, vrange)
            self.smu_vce.wires = 4
            # Configure VGE (voltage source)
            self.smu_vge.source_mode = "voltage"
            self.smu_vge.source_voltage = Vge
            self.smu_vge.compliance_current = Icomp
            sleep(0.5)
            # Enable sources: VGE then Bias
            self.smu_vge.enable_source()
            self.smu_vce.enable_source()

            start = datetime.now().timestamp()
            remaining = duration

            count_total = max(1, int(duration // interval))
            count = 0
            self.root.after(0, lambda: self.hp_status.config(text="Measuring… (data kept in memory)"))

            while remaining > 0 and not self.stop_event.is_set():
                self.smu_vce.source_current = Ic
                voltage = self.smu_vce.voltage
                Tj = self._compute_tj(voltage)
                tstamp = datetime.now().timestamp() - start
                # keep data for plot/export
                self.time_data.append(tstamp)
                self.tj_data.append(Tj)
                count += 1
                # update UI
                self.root.after(0, self._update_heating_ui, count, count_total)
                # wait for next interval
                slept = 0.0
                while slept < interval and not self.stop_event.is_set():
                    sleep(0.1)
                    slept += 0.1
                remaining -= interval
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Heating Measurement Error", str(e)))
        finally:
            try:
                self.smu_vce.beep(4000, 2)
                self.smu_vce.disable_source()
                self.smu_vge.disable_source()
                self.smu_vce.write("*CLS"); self.smu_vce.write("*SRE 0")
                self.smu_vge.write("*CLS"); self.smu_vge.write("*SRE 0")
            except Exception:
                pass
            self.root.after(0, self._heating_done)

    def _update_heating_ui(self, count, total):
        pct = 0 if total == 0 else min(100.0, (count / total) * 100)
        self.hp_progress['value'] = pct
        # redraw plot
        self.tax.clear()
        self.tax.set_xlabel('Time (s)')
        self.tax.set_ylabel('Tj (°C)')
        self.tax.set_title('Junction Temperature vs Time')
        self.tax.grid(True)
        if self.time_data:
            self.tax.plot(self.time_data, self.tj_data, color='tab:red')
        self.tfig.tight_layout()
        self.tcanvas.draw()

    def _heating_done(self):
        self.hp_status.config(text="Heating period finished (data ready to export).")
        self.hp_start.config(state='normal')
        self.hp_stop.config(state='disabled')

    # ---------- Export CSV on demand ----------
    def export_csv(self):
        if not self.time_data:
            messagebox.showinfo("Export CSV", "No data to export yet. Run the heating period first.")
            return
        folder_default = self.hp_folder.get().strip() or os.getcwd()
        prefix = self.hp_prefix.get().strip() or "temp_log"
        setpoint = self.hp_setpoint.get().strip() or "NA"
        default_name = f"{prefix}_{setpoint}.csv"
        path = filedialog.asksaveasfilename(
            title="Export Heating Period to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=folder_default,
            initialfile=default_name,
        )
        if not path:
            return
        try:
            df = pd.DataFrame({
                'time_s': self.time_data,
                'Tj_C': self.tj_data,
                'setpoint_C': [setpoint]*len(self.time_data)
            })
            df.to_csv(path, index=False)
            messagebox.showinfo("Export CSV", f"Heating period data exported to: {path}")
        except Exception as e:
            messagebox.showerror("Export CSV Error", str(e))

    # ---------- Single Measurement ----------
    def single_measurement(self):
        if self.smu_vce is None or self.smu_vge is None:
            messagebox.showerror("Error", "Connect both Keithley 2400 instruments first.")
            return
        N = int(safe_float(self.sm_N, 10))
        Ic_mA = safe_float(self.hp_Ic, 150.0)
        Ic = Ic_mA / 1000.0
        Vcomp = safe_float(self.hp_Vcomp, 2.0)
        nplc = safe_float(self.hp_nplc, 10)
        vrange = safe_float(self.hp_vrange, 2.0)
        Vge = safe_float(self.hp_Vge, 15.0)
        Icomp = safe_float(self.hp_Icomp, 1e-3)

        def worker():
            try:
                # configure identical to heating, then use buffer averaging
                self.smu_vce.apply_current(Ic, Vcomp)
                self.smu_vce.measure_voltage(nplc, vrange)
                self.smu_vce.wires = 4
                self.smu_vge.source_mode = "voltage"; self.smu_vge.source_voltage = Vge; self.smu_vge.compliance_current = Icomp
                sleep(0.5)
                self.smu_vge.enable_source(); self.smu_vce.enable_source()
                # buffer
                self.smu_vce.config_buffer(N)
                self.smu_vce.source_current = Ic
                self.smu_vce.start_buffer(); self.smu_vce.wait_for_buffer()
                voltage = self.smu_vce.mean_voltage
                Tj = self._compute_tj(voltage)
                self.smu_vce.beep(4000, 2)
                # update UI
                self.root.after(0, lambda: self._single_done(voltage, Tj))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Single Measurement Error", str(e)))
            finally:
                try:
                    self.smu_vce.disable_source(); self.smu_vge.disable_source()
                    self.smu_vce.write("*CLS"); self.smu_vce.write("*SRE 0")
                    self.smu_vge.write("*CLS"); self.smu_vge.write("*SRE 0")
                except Exception:
                    pass
        Thread(target=worker, daemon=True).start()

    def _single_done(self, voltage, Tj):
        self.last_voltage = voltage
        self.last_tj = Tj
        self.sm_result.config(text=f"Result: Vce(mean) = {voltage:.6f} V, Tj = {Tj:.2f} °C")
        if self.time_data:
            self.tax.plot([self.time_data[-1]], [Tj], marker='o', color='black', label='Single')
            self.tax.legend()
            self.tcanvas.draw()

    # ---------- Copy helpers ----------
    def copy_voltage(self):
        if self.last_voltage is None:
            messagebox.showinfo("Copy Voltage", "No single measurement yet.")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(f"{self.last_voltage:.6f}")
            self.root.update()  # keep clipboard after app closes
            messagebox.showinfo("Copy Voltage", "Voltage copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Copy Voltage Error", str(e))

    def copy_temperature(self):
        if self.last_tj is None:
            messagebox.showinfo("Copy Temperature", "No single measurement yet.")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(f"{self.last_tj:.2f}")
            self.root.update()
            messagebox.showinfo("Copy Temperature", "Temperature copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Copy Temperature Error", str(e))

    # ---------- settings API for unified export/import ----------
    def collect_settings(self) -> dict:
        return {
            "addresses": {
                "bias": self.vce_addr.get(),
                "vge": self.vge_addr.get(),
            },
            "heating": {
                "output_folder": self.hp_folder.get(),
                "file_prefix": self.hp_prefix.get(),
                "setpoint": self.hp_setpoint.get(),
                "model": self.hp_model.get(),
                "A": self.hp_A.get(),
                "B": self.hp_B.get(),
                "C": self.row_c_entry.get(),
                "duration_min": self.hp_duration.get(),
                "interval_s": self.hp_interval.get(),
                # instrument params (UI only)
                "Ic_mA": self.hp_Ic.get(),
                "Vcomp_V": self.hp_Vcomp.get(),
                "nplc": self.hp_nplc.get(),
                "vrange_V": self.hp_vrange.get(),
                "Vge_V": self.hp_Vge.get(),
                "Icomp_A": self.hp_Icomp.get(),
            },
            "single": {
                "buffer_N": self.sm_N.get(),
            },
        }

    def apply_settings(self, settings: dict):
        try:
            if "addresses" in settings:
                addrs = settings["addresses"]
                if "bias" in addrs:
                    self.vce_addr.delete(0, tk.END); self.vce_addr.insert(0, addrs["bias"])
                if "vge" in addrs:
                    self.vge_addr.delete(0, tk.END); self.vge_addr.insert(0, addrs["vge"])
            if "heating" in settings:
                h = settings["heating"]
                def put(widget, key):
                    if key in h:
                        widget.delete(0, tk.END); widget.insert(0, str(h[key]))
                put(self.hp_folder, "output_folder")
                put(self.hp_prefix, "file_prefix")
                put(self.hp_setpoint, "setpoint")
                if "model" in h:
                    self.hp_model.set(h["model"])
                    self._on_model_change()
                put(self.hp_A, "A")
                put(self.hp_B, "B")
                put(self.row_c_entry, "C")
                put(self.hp_duration, "duration_min")
                put(self.hp_interval, "interval_s")
                put(self.hp_Ic, "Ic_mA")
                put(self.hp_Vcomp, "Vcomp_V")
                put(self.hp_nplc, "nplc")
                put(self.hp_vrange, "vrange_V")
                put(self.hp_Vge, "Vge_V")
                put(self.hp_Icomp, "Icomp_A")
            if "single" in settings and "buffer_N" in settings["single"]:
                self.sm_N.delete(0, tk.END); self.sm_N.insert(0, str(settings["single"]["buffer_N"]))
            self.hp_status.config(text="Temperature settings applied")
        except Exception as e:
            messagebox.showerror("Apply Temp Settings Error", str(e))

    def _browse_to(self, entry: ttk.Entry):
        folder = filedialog.askdirectory()
        if folder:
            entry.delete(0, tk.END)
            entry.insert(0, folder)

# ------------------ Main App with Notebook ------------------
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Measurement Suite")
        try:
            self.state('zoomed')  # Windows
        except Exception:
            self.attributes('-zoomed', True)
        self.minsize(1200, 850)

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True)

        self.iv_tab = IVTab(nb)
        self.tj_tab = TemperatureTab(nb)
        nb.add(self.iv_tab, text="I–V Curves")
        nb.add(self.tj_tab, text="Junction Temperature")

        # expose unified handlers so tabs can call them
        self.export_all_settings = self._export_all_settings
        self.import_all_settings = self._import_all_settings

    # -------- Unified settings handlers --------
    def _collect_all_settings(self) -> dict:
        return {
            "app": {
                "version": APP_VERSION,
                "timestamp": datetime.now().isoformat(timespec='seconds'),
            },
            "iv_tab": self.iv_tab.collect_settings(),
            "temp_tab": self.tj_tab.collect_settings(),
        }

    def _apply_all_settings(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError("Invalid settings file format")
        if "iv_tab" in data:
            self.iv_tab.apply_settings(data["iv_tab"])
        if "temp_tab" in data:
            self.tj_tab.apply_settings(data["temp_tab"])

    def _export_all_settings(self):
        try:
            data = self._collect_all_settings()
            iv_file = data.get("iv_tab", {}).get("file", {})
            dut = iv_file.get("DUT Name:", "DUT")
            devid = iv_file.get("Device ID:", "dev")
            default_name = f"settings_{dut}_{devid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = filedialog.asksaveasfilename(
                title="Export All Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile=default_name,
            )
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Settings Export", f"Settings exported to: {path}")
        except Exception as e:
            messagebox.showerror("Export Settings Error", str(e))

    def _import_all_settings(self):
        try:
            path = filedialog.askopenfilename(
                title="Import All Settings",
                filetypes=[("JSON files", "*.json")]
            )
            if path:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._apply_all_settings(data)
                messagebox.showinfo("Settings Import", f"Settings imported from: {path}")
        except Exception as e:
            messagebox.showerror("Import Settings Error", str(e))

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
