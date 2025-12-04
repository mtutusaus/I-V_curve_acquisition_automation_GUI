import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import csv
import warnings
from time import sleep
from threading import Thread
from tek371 import Tek371
from pymeasure.instruments.keithley import Keithley2400

# Suppress PyVISA warning
warnings.filterwarnings("ignore", message="read string doesn't end with termination characters")


def compute_mean_file(folder_path: str, base_name: str, N: int):
    """Compute per-row mean of Voltage and Current across N files"""
    filepaths = [os.path.join(folder_path, f"{base_name}_{i}.csv") for i in range(1, N + 1)]
    rows_list = []

    for path in filepaths:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows_list.append(list(reader))

    header = rows_list[0][0]
    num_rows = len(rows_list[0])
    mean_rows = [header]

    for r in range(1, num_rows):
        values_v = [float(rows_list[i][r][0]) for i in range(N)]
        values_i = [float(rows_list[i][r][1]) for i in range(N)]
        mean_v = sum(values_v) / N
        mean_i = sum(values_i) / N
        mean_rows.append([mean_v, mean_i])

    mean_folder = os.path.join(folder_path, "mean")
    os.makedirs(mean_folder, exist_ok=True)
    out_path = os.path.join(mean_folder, f"{base_name}_MEAN.csv")

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(mean_rows)

    return out_path


class MeasurementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPIB I-V Measurement System")
        self.root.geometry("1100x900")

        # Device connections
        self.tek371 = None
        self.keithley = None
        self.measurement_running = False

        self.create_widgets()

    def create_widgets(self):
        # Main container with scrollbar
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="GPIB I-V Measurement System",
                          font=('Helvetica', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=4, pady=10)

        # Device Connection Frame
        conn_frame = ttk.LabelFrame(main_frame, text="Device Connection", padding="10")
        conn_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(conn_frame, text="Scan GPIB Bus",
                   command=self.scan_gpib).grid(row=0, column=0, pady=5)
        ttk.Label(conn_frame, text="Available devices:").grid(row=0, column=1, sticky=tk.W)

        self.gpib_text = scrolledtext.ScrolledText(conn_frame, height=3, width=70,
                                                   state='disabled')
        self.gpib_text.grid(row=1, column=0, columnspan=4, pady=5)

        # Tek371
        ttk.Label(conn_frame, text="Tek371 Address:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.tek_addr = ttk.Entry(conn_frame, width=25)
        self.tek_addr.insert(0, "GPIB0::23::INSTR")
        self.tek_addr.grid(row=2, column=1, sticky=tk.W)
        ttk.Button(conn_frame, text="Connect",
                   command=self.connect_tek).grid(row=2, column=2, padx=5)
        self.tek_status = ttk.Label(conn_frame, text="Not connected", foreground='gray')
        self.tek_status.grid(row=2, column=3, sticky=tk.W)

        # Keithley
        ttk.Label(conn_frame, text="Keithley 2400:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.keithley_addr = ttk.Entry(conn_frame, width=25)
        self.keithley_addr.insert(0, "GPIB::24")
        self.keithley_addr.grid(row=3, column=1, sticky=tk.W)
        ttk.Button(conn_frame, text="Connect",
                   command=self.connect_keithley).grid(row=3, column=2, padx=5)
        self.keithley_status = ttk.Label(conn_frame, text="Not connected", foreground='gray')
        self.keithley_status.grid(row=3, column=3, sticky=tk.W)

        # Measurement Parameters Frame
        param_frame = ttk.LabelFrame(main_frame, text="Measurement Parameters", padding="10")
        param_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)

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
            entry = ttk.Entry(param_frame, width=20)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.param_entries[label] = entry

        # File Settings Frame
        file_frame = ttk.LabelFrame(main_frame, text="File Settings", padding="10")
        file_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(file_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.folder_entry = ttk.Entry(file_frame, width=50)
        self.folder_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="Browse",
                   command=self.browse_folder).grid(row=0, column=2)

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
            entry = ttk.Entry(file_frame, width=20)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.file_entries[label] = entry

        # Control Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=4, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start Measurement",
                                    command=self.start_measurement)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop",
                                   command=self.stop_measurement, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)

        ttk.Button(btn_frame, text="Clear Plot",
                   command=self.clear_plot).grid(row=0, column=2, padx=5)

        # Status
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        self.progress = ttk.Progressbar(status_frame, length=600, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=6, column=0, columnspan=4, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        self.ax.set_title('I-V Measurement Data')
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

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
            messagebox.showerror("Error", f"Failed to scan GPIB: {str(e)}")

    def connect_tek(self):
        try:
            self.update_status("Connecting to Tek371...")
            addr = self.tek_addr.get()
            self.tek371 = Tek371(addr)
            idn = self.tek371.id_string()
            self.tek_status.config(text=f"✓ Connected: {idn}", foreground='green')
            self.update_status("Tek371 connected successfully")
        except Exception as e:
            self.tek_status.config(text=f"✗ Error: {str(e)}", foreground='red')
            self.update_status(f"Failed to connect Tek371: {str(e)}")

    def connect_keithley(self):
        try:
            self.update_status("Connecting to Keithley 2400...")
            addr = self.keithley_addr.get()
            self.keithley = Keithley2400(addr)
            idn = self.keithley.id
            self.keithley_status.config(text=f"✓ Connected: {idn}", foreground='green')
            self.update_status("Keithley 2400 connected successfully")
        except Exception as e:
            self.keithley_status.config(text=f"✗ Error: {str(e)}", foreground='red')
            self.update_status(f"Failed to connect Keithley: {str(e)}")

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        self.ax.set_title('I-V Measurement Data')
        self.ax.grid(True)
        self.canvas.draw()
        self.update_status("Plot cleared")

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

        # Run measurement in separate thread to keep GUI responsive
        thread = Thread(target=self.perform_measurement, daemon=True)
        thread.start()

    def stop_measurement(self):
        self.measurement_running = False
        self.update_status("Stopping measurement...")

    def perform_measurement(self):
        try:
            # Get parameters
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

            # Configure Tracer
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

            # Perform measurements
            for i in range(1, num_curves + 1):
                if not self.measurement_running:
                    self.update_status("Measurement stopped by user")
                    break

                self.update_status(f"Measuring curve {i}/{num_curves}...")
                self.progress['value'] = (i / num_curves) * 100

                # Set collector supply and perform sweep
                self.tek371.set_collector_supply(vce_pct)
                self.tek371.set_measurement_mode("SWE")

                if self.tek371.wait_for_srq(timeout_s=60.0):
                    self.update_status(f"Sweep {i}/{num_curves} complete, reading data...")
                else:
                    raise TimeoutError(f"Sweep {i}/{num_curves} timeout")

                # Save curve
                filename = os.path.join(folder, f"{base_filename}_{i}.csv")
                self.tek371.read_curve(filename)

                # Plot the curve
                try:
                    data = pd.read_csv(filename)
                    self.ax.plot(data.iloc[:, 0], data.iloc[:, 1],
                                 alpha=0.5, linewidth=1, color='blue')
                    self.canvas.draw()
                except Exception as e:
                    print(f"Plot error: {e}")

                # Reset SRQ
                self.tek371.discard_and_disable_all_events()
                self.tek371.enable_srq_event()

            # Cleanup
            self.keithley.disable_source()
            self.keithley.beep(4000, 2)
            self.tek371.disable_srq_event()

            if self.measurement_running:
                # Compute mean
                self.update_status("Computing mean curve...")
                mean_path = compute_mean_file(folder, base_filename, num_curves)

                # Plot mean
                mean_data = pd.read_csv(mean_path)
                self.ax.plot(mean_data.iloc[:, 0], mean_data.iloc[:, 1],
                             'r-', linewidth=2.5, label='Mean')
                self.ax.legend()
                self.canvas.draw()

                self.update_status(f"Measurement complete! Data saved to {folder}")
                self.progress['value'] = 100

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Measurement Error", str(e))
            if self.keithley:
                try:
                    self.keithley.disable_source()
                except:
                    pass
        finally:
            self.measurement_running = False
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementGUI(root)
    root.mainloop()