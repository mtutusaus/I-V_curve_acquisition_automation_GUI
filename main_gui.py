
# basic_gui.py
# Single-file, refactored and thread-safe I-V measurement GUI

from __future__ import annotations

import os
import csv
import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from time import sleep
from typing import Callable, Dict, Optional, Tuple

# --- Third-party / instruments ---
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tek371 import Tek371
from pymeasure.instruments.keithley import Keithley2400

# --- Global warnings ---
warnings.filterwarnings("ignore", message="read string doesn't end with termination characters")


# =========================
# 1) Constants & Config
# =========================
class UI:
    TITLE = "I-V Measurement System"

    # Parameter labels (kept stable for export/import compatibility)
    L_SMU_V = "SMU Source Voltage (V):"
    L_SMU_I_COMP = "SMU Compliance Current (A):"
    L_TR_H = "Tracer Horizontal Scale (V/div):"
    L_TR_V = "Tracer Vertical Scale (A/div):"
    L_TR_VCE = "Tracer VCE Percentage (%):"
    L_TR_PK_PWR = "Tracer Peak Power (W):"

    # File labels
    L_DUT = "DUT Name:"
    L_DEV = "Device ID:"
    L_VGE = "Vge Applied (V):"
    L_TEMP = "Temperature (°C):"
    L_NCURVES = "Number of Curves:"

    # Address keys (for JSON)
    K_TEK = "tek371"
    K_K24 = "keithley2400"


@dataclass
class Defaults:
    tek_address: str = "GPIB::23"
    k24_address: str = "GPIB::24"

    smu_v: float = 20.0
    smu_i_comp: float = 1e-3
    tr_h: float = 200e-3
    tr_v: float = 5.0
    tr_vce_pct: float = 100.0
    tr_peak_power: int = 300

    dut: str = "H40ER5S"
    dev: str = "dev10"
    vge: float = 20.0
    temp_c: float = 25.0
    ncurves: int = 10


# =========================
# 2) Utilities
# =========================
def try_zoomed(root: tk.Tk) -> None:
    """Maximize window if supported."""
    try:
        root.state("zoomed")
    except Exception:
        # Fallback: set a reasonable size if zoomed is unsupported
        root.geometry("1200x850")


def require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def ensure_folder_writable(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    test_file = path / ".write_test"
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
    finally:
        try:
            test_file.unlink(missing_ok=True)
        except Exception:
            pass


def compute_mean_file(folder_path: Path, base_name: str, N: int) -> Path:
    """
    Compute per-row mean of the first two columns across N CSV files.
    Output is written to <folder>/mean/<base_name>_MEAN.csv
    """
    filepaths = [folder_path / f"{base_name}_{i}.csv" for i in range(1, N + 1)]
    if not all(p.exists() for p in filepaths):
        missing = [str(p.name) for p in filepaths if not p.exists()]
        raise FileNotFoundError(f"Missing curve files: {missing}")

    dfs = []
    for p in filepaths:
        df = pd.read_csv(p)
        if df.shape[1] < 2:
            raise ValueError(f"{p.name} must have at least two columns (Voltage, Current)")
        dfs.append(df.iloc[:, :2].copy())

    # Sanity check: equal number of rows
    nrows = {len(df) for df in dfs}
    if len(nrows) != 1:
        raise ValueError("Curve CSVs have different row counts; cannot compute row-wise mean.")

    # Concatenate on columns and take mean pairwise
    concat_v = pd.concat([df.iloc[:, 0] for df in dfs], axis=1)
    concat_i = pd.concat([df.iloc[:, 1] for df in dfs], axis=1)
    mean_v = concat_v.mean(axis=1)
    mean_i = concat_i.mean(axis=1)

    mean_df = pd.DataFrame({"Voltage (V)": mean_v, "Current (A)": mean_i})

    out_dir = folder_path / "mean"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{base_name}_MEAN.csv"
    mean_df.to_csv(out_path, index=False)
    return out_path


# =========================================
# 3) Settings models (+ parsing & validation)
# =========================================
@dataclass
class AddressSettings:
    tek371: str
    keithley2400: str


@dataclass
class MeasurementParams:
    smu_v: float
    smu_i_comp: float
    tr_h: float
    tr_v: float
    tr_vce_pct: float
    tr_peak_power: int

    def validate(self) -> None:
        require_positive(UI.L_SMU_V, self.smu_v)
        require_positive(UI.L_SMU_I_COMP, self.smu_i_comp)
        require_positive(UI.L_TR_H, self.tr_h)
        require_positive(UI.L_TR_V, self.tr_v)
        require_positive(UI.L_TR_PK_PWR, float(self.tr_peak_power))
        if not (0 < self.tr_vce_pct <= 1000):
            raise ValueError(f"{UI.L_TR_VCE} must be in (0, 1000] (got {self.tr_vce_pct}).")


@dataclass
class FileParams:
    output_folder: Path
    dut: str
    dev: str
    vge: float
    temp_c: float
    ncurves: int

    @property
    def base_filename(self) -> str:
        # Preserve original naming semantics
        return f"{self.dut}_{self.dev}_{self.vge}V_{self.temp_c}C"

    def validate(self) -> None:
        if not self.dut:
            raise ValueError(f"{UI.L_DUT} cannot be empty")
        if not self.dev:
            raise ValueError(f"{UI.L_DEV} cannot be empty")
        require_positive(UI.L_VGE, self.vge)
        # Temperature may be <= 0, so only numeric check is implied by float conversion.
        require_positive(UI.L_NCURVES, float(self.ncurves))
        ensure_folder_writable(self.output_folder)


@dataclass
class RunSettings:
    addresses: AddressSettings
    measurement: MeasurementParams
    file: FileParams

    def validate(self) -> None:
        self.measurement.validate()
        self.file.validate()


# =========================================
# 4) Measurement Controller (no Tk calls)
# =========================================
class MeasurementController:
    """
    Encapsulates the measurement sequence, calling provided callbacks
    for status/progress/plot actions. No direct Tkinter access here.
    """

    def __init__(self, tek: Tek371, k24: Keithley2400) -> None:
        self.tek = tek
        self.k24 = k24
        self._stop_event = Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(
        self,
        settings: RunSettings,
        on_status: Callable[[str], None],
        on_progress: Callable[[float], None],
        on_plot_csv: Callable[[Path], None],
        on_plot_mean_csv: Callable[[Path], None],
    ) -> None:
        settings.validate()
        folder = settings.file.output_folder
        base = settings.file.base_filename
        N = settings.file.ncurves

        # Configure instruments
        on_status("Configuring Keithley 2400...")
        self._configure_keithley(settings)
        sleep(0.2)

        on_status("Configuring Tek371...")
        self._configure_tek(settings)
        sleep(0.2)

        on_status("Starting measurements...")
        self.k24.enable_source()

        try:
            for i in range(1, N + 1):
                if self._stop_event.is_set():
                    on_status("Measurement stopped by user")
                    break

                on_progress(((i - 1) / N) * 100.0)
                on_status(f"Measuring curve {i}/{N}...")

                self.tek.set_collector_supply(settings.measurement.tr_vce_pct)
                self.tek.set_measurement_mode("SWE")

                if not self.tek.wait_for_srq(timeout_s=60.0):
                    raise TimeoutError(f"Sweep {i}/{N} timed out")

                filename = folder / f"{base}_{i}.csv"
                self.tek.read_curve(str(filename))
                on_plot_csv(filename)

                # Reset SRQ for next iteration
                self.tek.discard_and_disable_all_events()
                self.tek.enable_srq_event()

                on_progress((i / N) * 100.0)

            # Compute and plot mean if not stopped
            if not self._stop_event.is_set():
                on_status("Computing mean curve...")
                mean_path = compute_mean_file(folder, base, N)
                on_plot_mean_csv(mean_path)
                on_status(f"Measurement complete! Data saved to {folder}")
                on_progress(100.0)

        finally:
            # Cleanup regardless of success/failure/stop
            try:
                self.k24.disable_source()
            except Exception:
                pass
            try:
                self.k24.beep(4000, 2)
            except Exception:
                pass
            try:
                self.tek.disable_srq_event()
            except Exception:
                pass

    # ----- Internal helpers (instrument config) -----
    def _configure_keithley(self, settings: RunSettings) -> None:
        p = settings.measurement
        self.k24.reset()
        self.k24.write("*CLS")
        self.k24.write("*SRE 0")
        self.k24.use_front_terminals()
        self.k24.source_mode = "voltage"
        self.k24.source_voltage = p.smu_v
        self.k24.compliance_current = p.smu_i_comp

    def _configure_tek(self, settings: RunSettings) -> None:
        p = settings.measurement
        self.tek.initialize()
        self.tek.set_peak_power(p.tr_peak_power)
        self.tek.set_step_number(0)
        self.tek.set_step_voltage(200e-3)
        self.tek.set_step_offset(0)
        self.tek.enable_srq_event()
        self.tek.set_horizontal("COL", p.tr_h)
        self.tek.set_vertical(p.tr_v)
        self.tek.set_display_mode("STO")


# =========================
# 5) GUI
# =========================
class MeasurementGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(UI.TITLE)
        try_zoomed(self.root)
        self.root.minsize(1200, 850)

        # Instruments
        self.tek371: Optional[Tek371] = None
        self.keithley: Optional[Keithley2400] = None

        # Controller / thread
        self.controller: Optional[MeasurementController] = None
        self.worker_thread: Optional[Thread] = None

        self._build_widgets()

    # ----- UI construction -----
    def _build_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        title = ttk.Label(main_frame, text=UI.TITLE, font=("Helvetica", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        left = ttk.Frame(main_frame)
        right = ttk.Frame(main_frame)
        left.grid(row=1, column=0, sticky=tk.N + tk.W, padx=(0, 10))
        right.grid(row=1, column=1, sticky=tk.N + tk.S + tk.E + tk.W)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ----- Connection frame -----
        conn = ttk.LabelFrame(left, text="Device Connection", padding="10")
        conn.grid(row=0, column=0, sticky=tk.W + tk.E, pady=5)

        ttk.Button(conn, text="Scan GPIB Bus", command=self.scan_gpib).grid(row=0, column=0, pady=5, sticky=tk.W)
        ttk.Label(conn, text="Available devices:").grid(row=0, column=1, sticky=tk.W)
        self.gpib_text = scrolledtext.ScrolledText(conn, height=3, width=50, state="disabled")
        self.gpib_text.grid(row=1, column=0, columnspan=4, pady=5, sticky=tk.W)

        addrs_label = ttk.Label(conn, text="Instrument Addresses", font=("Helvetica", 10, "bold"))
        addrs_label.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(8, 2))

        ttk.Label(conn, text="Tek371:").grid(row=3, column=0, sticky=tk.W)
        self.tek_addr = ttk.Entry(conn, width=25)
        self.tek_addr.insert(0, Defaults().tek_address)
        self.tek_addr.grid(row=3, column=1, sticky=tk.W)

        ttk.Label(conn, text="Keithley 2400:").grid(row=3, column=2, sticky=tk.W)
        self.keithley_addr = ttk.Entry(conn, width=20)
        self.keithley_addr.insert(0, Defaults().k24_address)
        self.keithley_addr.grid(row=3, column=3, sticky=tk.W)

        ttk.Label(conn, text="Tek371:").grid(row=4, column=0, sticky=tk.W, pady=(8, 5))
        ttk.Button(conn, text="Connect", command=self.connect_tek).grid(row=4, column=1, padx=5, sticky=tk.W)
        self.tek_status = ttk.Label(conn, text="Not connected", foreground="gray", width=12)
        self.tek_status.grid(row=4, column=2, sticky=tk.W)

        ttk.Label(conn, text="Keithley 2400:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Button(conn, text="Connect", command=self.connect_keithley).grid(row=5, column=1, padx=5, sticky=tk.W)
        self.keithley_status = ttk.Label(conn, text="Not connected", foreground="gray", width=12)
        self.keithley_status.grid(row=5, column=2, sticky=tk.W)

        # ----- Measurement params -----
        param = ttk.LabelFrame(left, text="Measurement Parameters", padding="10")
        param.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.param_entries: Dict[str, ttk.Entry] = {}
        for i, (label, default) in enumerate(
            [
                (UI.L_SMU_V, str(Defaults().smu_v)),
                (UI.L_SMU_I_COMP, str(Defaults().smu_i_comp)),
                (UI.L_TR_H, str(Defaults().tr_h)),
                (UI.L_TR_V, str(Defaults().tr_v)),
                (UI.L_TR_VCE, str(Defaults().tr_vce_pct)),
                (UI.L_TR_PK_PWR, str(Defaults().tr_peak_power)),
            ]
        ):
            ttk.Label(param, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            e = ttk.Entry(param, width=15)
            e.insert(0, default)
            e.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.param_entries[label] = e

        # ----- File settings -----
        filef = ttk.LabelFrame(left, text="File Settings", padding="10")
        filef.grid(row=2, column=0, sticky=tk.W, pady=5)

        ttk.Label(filef, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.folder_entry = ttk.Entry(filef, width=35)
        self.folder_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(filef, text="Browse", command=self.browse_folder).grid(row=0, column=2)

        self.file_entries: Dict[str, ttk.Entry] = {}
        for i, (label, default) in enumerate(
            [
                (UI.L_DUT, Defaults().dut),
                (UI.L_DEV, Defaults().dev),
                (UI.L_VGE, str(Defaults().vge)),
                (UI.L_TEMP, str(Defaults().temp_c)),
                (UI.L_NCURVES, str(Defaults().ncurves)),
            ],
            start=1,
        ):
            ttk.Label(filef, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            e = ttk.Entry(filef, width=15)
            e.insert(0, default)
            e.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.file_entries[label] = e

        ttk.Button(filef, text="Export Settings", command=self.export_settings).grid(
            row=i + 1, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Button(filef, text="Import Settings", command=self.import_settings).grid(
            row=i + 1, column=1, sticky=tk.W, pady=(8, 0)
        )

        # ----- Buttons -----
        btns = ttk.Frame(left)
        btns.grid(row=3, column=0, pady=10, sticky=tk.W + tk.E)
        self.start_btn = ttk.Button(btns, text="Start Measurement", command=self.start_measurement)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_measurement, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Clear Plot", command=self.clear_plot).grid(row=0, column=2, padx=5)

        # ----- Status -----
        status = ttk.LabelFrame(left, text="Status", padding="10")
        status.grid(row=4, column=0, sticky=tk.W + tk.E, pady=5)
        status.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status, text="Ready", relief=tk.SUNKEN, anchor="w", justify="left")
        self.status_label.grid(row=0, column=0, sticky=tk.E + tk.W)
        self.progress = ttk.Progressbar(status, length=400, mode="determinate")
        self.progress.grid(row=1, column=0, sticky=tk.E + tk.W, pady=(6, 0))
        status.bind("<Configure>", self._on_status_resize)

        # ----- Right side plot -----
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        plot_frame = ttk.Frame(right)
        plot_frame.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.fig.tight_layout()

    # ----- UI helpers -----
    def _style_axes(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("Voltage (V)")
        self.ax.set_ylabel("Current (A)")
        self.ax.set_title("I-V Measurement Data")
        self.ax.grid(True)

    def _on_status_resize(self, event) -> None:
        try:
            self.status_label.configure(wraplength=max(event.width - 20, 100))
        except Exception:
            pass

    def _set_status(self, message: str) -> None:
        # Runs on UI thread
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def _set_progress(self, percent: float) -> None:
        # Runs on UI thread
        self.progress["value"] = percent
        self.root.update_idletasks()

    def _post(self, fn: Callable[[], None]) -> None:
        # Schedule on main thread
        self.root.after(0, fn)

    # ----- Settings import/export -----
    def collect_settings(self) -> Dict:
        addrs = {
            UI.K_TEK: self.tek_addr.get(),
            UI.K_K24: self.keithley_addr.get(),
        }
        meas = {label: self.param_entries[label].get() for label in self.param_entries}
        file_set = {
            "output_folder": self.folder_entry.get(),
            UI.L_DUT: self.file_entries[UI.L_DUT].get(),
            UI.L_DEV: self.file_entries[UI.L_DEV].get(),
            UI.L_VGE: self.file_entries[UI.L_VGE].get(),
            UI.L_TEMP: self.file_entries[UI.L_TEMP].get(),
            UI.L_NCURVES: self.file_entries[UI.L_NCURVES].get(),
        }
        base_filename = f"{file_set[UI.L_DUT]}_{file_set[UI.L_DEV]}_{file_set[UI.L_VGE]}V_{file_set[UI.L_TEMP]}C"
        return {
            "addresses": addrs,
            "measurement": meas,
            "file": file_set,
            "base_filename_example": base_filename,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def apply_settings(self, settings: dict) -> None:
        try:
            if "addresses" in settings:
                addrs = settings["addresses"]
                if UI.K_TEK in addrs:
                    self.tek_addr.delete(0, tk.END)
                    self.tek_addr.insert(0, addrs[UI.K_TEK])
                if UI.K_K24 in addrs:
                    self.keithley_addr.delete(0, tk.END)
                    self.keithley_addr.insert(0, addrs[UI.K_K24])

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

            self._set_status("Settings applied successfully")
        except Exception as e:
            messagebox.showerror("Apply Settings Error", str(e))

    def export_settings(self) -> None:
        try:
            data = self.collect_settings()
            default_name = f"settings_{data['file'][UI.L_DUT]}_{data['file'][UI.L_DEV]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = filedialog.asksaveasfilename(
                title="Export Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile=default_name,
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self._set_status(f"Settings exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Settings Error", str(e))

    def import_settings(self) -> None:
        try:
            path = filedialog.askopenfilename(title="Import Settings", filetypes=[("JSON files", "*.json")])
            if path:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.apply_settings(data)
                self._set_status(f"Settings imported from {path}")
        except Exception as e:
            messagebox.showerror("Import Settings Error", str(e))

    # ----- Device ops -----
    def scan_gpib(self) -> None:
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            gpib_devices = [r for r in resources if "GPIB" in r]
            self.gpib_text.config(state="normal")
            self.gpib_text.delete(1.0, tk.END)
            if gpib_devices:
                self.gpib_text.insert(tk.END, "\n".join(gpib_devices))
                self._set_status(f"Found {len(gpib_devices)} GPIB device(s)")
            else:
                self.gpib_text.insert(tk.END, "No GPIB devices found")
                self._set_status("No GPIB devices found")
            self.gpib_text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error scanning GPIB", str(e))

    def connect_tek(self) -> None:
        try:
            self._set_status("Connecting to Tek371...")
            addr = self.tek_addr.get()
            self.tek371 = Tek371(addr)
            idn = None
            try:
                idn = self.tek371.id_string()
            except Exception:
                idn = None
            if not idn:
                raise RuntimeError("Tek371 did not respond to ID query")
            self.tek_status.config(text="Connected", foreground="green")
            self._set_status(f"Tek371 connected successfully: {idn}")
        except Exception as e:
            self.tek_status.config(text="Error", foreground="red")
            messagebox.showerror("Tek371 Connection Error", str(e))
            self._set_status("Tek371 connection failed")

    def connect_keithley(self) -> None:
        try:
            self._set_status("Connecting to Keithley 2400...")
            addr = self.keithley_addr.get()
            self.keithley = Keithley2400(addr)
            idn = None
            try:
                idn = getattr(self.keithley, "idn", None) or getattr(self.keithley, "id", None)
                if not idn:
                    idn = self.keithley.ask("*IDN?")
            except Exception:
                idn = None
            if not idn:
                raise RuntimeError("Keithley 2400 did not respond to *IDN? or idn")
            self.keithley_status.config(text="Connected", foreground="green")
            self._set_status(f"Keithley connected successfully: {idn}")
        except Exception as e:
            self.keithley_status.config(text="Error", foreground="red")
            messagebox.showerror("Keithley 2400 Connection Error", str(e))
            self._set_status("Keithley connection failed")

    # ----- Plot ops -----
    def clear_plot(self) -> None:
        self._style_axes()
        self.fig.tight_layout()
        self.canvas.draw()
        self._set_status("Plot cleared")
        self._set_progress(0.0)

    def _plot_csv(self, path: Path) -> None:
        # Runs on UI thread
        try:
            data = pd.read_csv(path)
            self.ax.plot(data.iloc[:, 0], data.iloc[:, 1], alpha=0.5, linewidth=1, color="blue")
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Plot error ({path.name}): {e}")

    def _plot_mean_csv(self, path: Path) -> None:
        # Runs on UI thread
        try:
            data = pd.read_csv(path)
            self.ax.plot(data.iloc[:, 0], data.iloc[:, 1], "r-", linewidth=1.5, label="Mean")
            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Plot mean error ({path.name}): {e}")

    # ----- Run / Stop -----
    def _parse_settings(self) -> RunSettings:
        # Addresses
        addrs = AddressSettings(
            tek371=self.tek_addr.get(),
            keithley2400=self.keithley_addr.get(),
        )

        # Measurement params
        p = MeasurementParams(
            smu_v=float(self.param_entries[UI.L_SMU_V].get()),
            smu_i_comp=float(self.param_entries[UI.L_SMU_I_COMP].get()),
            tr_h=float(self.param_entries[UI.L_TR_H].get()),
            tr_v=float(self.param_entries[UI.L_TR_V].get()),
            tr_vce_pct=float(self.param_entries[UI.L_TR_VCE].get()),
            tr_peak_power=int(self.param_entries[UI.L_TR_PK_PWR].get()),
        )

        # File params
        folder = self.folder_entry.get()
        if not folder:
            raise ValueError("Please select an output folder")
        f = FileParams(
            output_folder=Path(folder),
            dut=self.file_entries[UI.L_DUT].get(),
            dev=self.file_entries[UI.L_DEV].get(),
            vge=float(self.file_entries[UI.L_VGE].get()),
            temp_c=float(self.file_entries[UI.L_TEMP].get()),
            ncurves=int(self.file_entries[UI.L_NCURVES].get()),
        )
        return RunSettings(addresses=addrs, measurement=p, file=f)

    def start_measurement(self) -> None:
        # Basic connection checks
        if self.tek371 is None or self.keithley is None:
            messagebox.showerror("Error", "Please connect both devices before starting!")
            return

        try:
            settings = self._parse_settings()
            settings.validate()
        except Exception as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return

        self.controller = MeasurementController(self.tek371, self.keithley)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # Wrap controller callbacks to main thread via ._post(...)
        def on_status(msg: str) -> None:
            self._post(lambda: self._set_status(msg))

        def on_progress(pct: float) -> None:
            self._post(lambda: self._set_progress(pct))

        def on_plot_csv(path: Path) -> None:
            self._post(lambda: self._plot_csv(path))

        def on_plot_mean_csv(path: Path) -> None:
            self._post(lambda: self._plot_mean_csv(path))

        def work():
            try:
                self.controller.run(settings, on_status, on_progress, on_plot_csv, on_plot_mean_csv)
            except Exception as e:
                # Report errors on UI thread
                self._post(lambda: (self._set_status("Measurement error"), messagebox.showerror("Measurement Error", str(e))))
            finally:
                # Restore buttons on UI thread
                self._post(lambda: (self.start_btn.config(state="normal"), self.stop_btn.config(state="disabled")))

        self.worker_thread = Thread(target=work, daemon=True)
        self.worker_thread.start()

    def stop_measurement(self) -> None:
        if self.controller:
            self.controller.stop()
        self._set_status("Stopping measurement...")

    # ----- File dialogs -----
    def browse_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)


# =========================
# 6) Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementGUI(root)
    root.mainloop()
