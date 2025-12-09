from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from time import sleep
from typing import Callable, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tek371 import Tek371
from pymeasure.instruments.keithley import Keithley2400

warnings.filterwarnings("ignore", message="read string doesn't end with termination characters")


class UI:
    """User-facing strings & keys consolidated here."""

    TITLE = "I-V Measurement System"

    # Labels
    L_SMU_V = "Gate bias voltage (V):"
    L_SMU_I_COMP = "Gate bias compliance current (mA):"
    L_TR_H = "Tracer Horizontal Scale (V/div):"
    L_TR_V = "Tracer Vertical Scale (A/div):"
    L_TR_VCE = "Tracer VCE Percentage (%):"
    L_TR_PK_PWR = "Tracer Peak Power (W):"  # fixed value shown only

    L_DUT = "DUT Name:"
    L_DEV = "Device ID:"
    L_VGE = "Gate bias applied (V):"
    L_TEMP = "Temperature (°C):"
    L_NCURVES = "Number of Curves:"

    # Address keys (for JSON)
    K_TEK = "tek371"
    K_K24 = "keithley2400"

    # Banner
    WARN_CONNECT = "Must connect both equipments to enable start measurement button"

    # Status messages (centralized)
    STATUS_CONFIG_SMU = "Configuring Keithley 2400..."
    STATUS_CONFIG_TEK = "Configuring Tek371..."
    STATUS_START = "Starting measurements..."
    STATUS_MEAS = "Measuring curve {i}/{n}..."
    STATUS_STOPPED = "Measurement stopped by user"
    STATUS_MEAN = "Computing mean curve..."
    STATUS_DONE = "Measurement complete! Data saved to {folder}"
    STATUS_ERROR = "Measurement error"


@dataclass
class Defaults:
    tek_address: str = "GPIB::23"
    k24_address: str = "GPIB::24"

    smu_v: float = 15.0
    smu_i_comp_mA: float = 1.0  # max 1 mA
    tr_h: float = 0.2  # default 0.2 V/div
    tr_v: float = 1.0  # default 1 A/div
    tr_vce_pct: float = 100.0
    tr_peak_power: int = 300  # fixed 300 W

    dut: str = "dut"
    dev: str = "dev0"
    temp_c: float = 25.0
    ncurves: int = 10


# UI choices (strings for Comboboxes)
H_CHOICES = ("0.1", "0.2", "0.5", "1", "2", "5")
V_CHOICES = ("0.5", "1", "2", "5")

# Numeric allowed sets for robust validation
ALLOWED_H = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
ALLOWED_V = {0.5, 1.0, 2.0, 5.0}

def _is_allowed(value: float, allowed: set[float], eps: float = 1e-9) -> bool:
    return any(abs(value - a) <= eps for a in allowed)


def try_zoomed(root: tk.Tk) -> None:
    """Maximize window if supported."""
    try:
        root.state("zoomed")
    except Exception:
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
    """Compute per-row mean of the first two columns across N CSV files."""
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

    nrows = {len(df) for df in dfs}
    if len(nrows) != 1:
        raise ValueError("Curve CSVs have different row counts; cannot compute row-wise mean.")

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
# Data models & controller
# =========================================
@dataclass
class AddressSettings:
    tek371: str
    keithley2400: str


@dataclass
class MeasurementParams:
    smu_v: float
    smu_i_comp_mA: float  # UI in mA
    tr_h: float
    tr_v: float
    tr_vce_pct: float
    tr_peak_power: int  # fixed to 300

    def validate(self) -> None:
        require_positive(UI.L_SMU_V, self.smu_v)
        require_positive(UI.L_SMU_I_COMP, self.smu_i_comp_mA)
        if self.smu_i_comp_mA > 1.0:
            raise ValueError(f"{UI.L_SMU_I_COMP} must be ≤ 1.0 mA (got {self.smu_i_comp_mA} mA).")
        if not _is_allowed(self.tr_h, ALLOWED_H):
            raise ValueError(f"{UI.L_TR_H} must be one of {sorted(ALLOWED_H)} V/div (got {self.tr_h}).")
        if not _is_allowed(self.tr_v, ALLOWED_V):
            raise ValueError(f"{UI.L_TR_V} must be one of {sorted(ALLOWED_V)} A/div (got {self.tr_v}).")
        if not (0.0 <= self.tr_vce_pct <= 100.0):
            raise ValueError(f"{UI.L_TR_VCE} must be between 0 and 100 (got {self.tr_vce_pct}).")
        if self.tr_peak_power != 300:
            raise ValueError("Peak power is fixed to 300 W for this GUI.")

    @property
    def smu_i_comp_A(self) -> float:
        return self.smu_i_comp_mA / 1000.0


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
        return f"{self.dut}_{self.dev}_{self.vge}V_{self.temp_c}C"

    def validate(self) -> None:
        if not self.dut:
            raise ValueError(f"{UI.L_DUT} cannot be empty")
        if not self.dev:
            raise ValueError(f"{UI.L_DEV} cannot be empty")
        require_positive(UI.L_VGE, self.vge)
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


class MeasurementController:
    """Encapsulates the measurement sequence (no direct Tk calls)."""

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
        on_status(UI.STATUS_CONFIG_SMU)
        self._configure_keithley(settings)
        sleep(0.1)

        on_status(UI.STATUS_CONFIG_TEK)
        self._configure_tek(settings)
        sleep(0.1)

        on_status(UI.STATUS_START)
        self.k24.enable_source()

        try:
            for i in range(1, N + 1):
                if self._stop_event.is_set():
                    on_status(UI.STATUS_STOPPED)
                    break

                on_progress(((i - 1) / N) * 100.0)
                on_status(UI.STATUS_MEAS.format(i=i, n=N))

                # Trigger sweep & wait
                self.tek.set_collector_supply(settings.measurement.tr_vce_pct)
                self.tek.set_measurement_mode("SWE")
                if not self.tek.wait_for_srq(timeout_s=60.0):
                    raise TimeoutError(f"Sweep {i}/{N} timed out")

                # Save CSV & plot
                filename = folder / f"{base}_{i}.csv"
                self.tek.read_curve(str(filename))
                on_plot_csv(filename)

                # Reset SRQ for next iteration
                self.tek.discard_and_disable_all_events()
                self.tek.enable_srq_event()

                on_progress((i / N) * 100.0)

            # Compute and plot mean if not stopped
            if not self._stop_event.is_set():
                on_status(UI.STATUS_MEAN)
                mean_path = compute_mean_file(folder, base, N)
                on_plot_mean_csv(mean_path)
                on_status(UI.STATUS_DONE.format(folder=folder))
                on_progress(100.0)

        finally:
            # Cleanup regardless
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
        self.k24.compliance_current = p.smu_i_comp_A

    def _configure_tek(self, settings: RunSettings) -> None:
        p = settings.measurement
        self.tek.initialize()
        self.tek.set_peak_power(300)  # fixed
        self.tek.set_step_number(0)
        self.tek.set_step_voltage(200e-3)
        self.tek.set_step_offset(0)
        self.tek.enable_srq_event()
        self.tek.set_horizontal("COL", p.tr_h)
        self.tek.set_vertical(p.tr_v)
        self.tek.set_display_mode("STO")


# =========================================
# GUI
# =========================================
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

        # Run state flag (explicit, clearer than widget state checks)
        self.is_running: bool = False

        # Parameter variables (Spinboxes/Comboboxes)
        self.var_smu_v = tk.StringVar(value=str(Defaults().smu_v))
        self.var_i_comp = tk.StringVar(value=str(Defaults().smu_i_comp_mA))
        self.var_tr_h = tk.StringVar(value=str(Defaults().tr_h))
        self.var_tr_v = tk.StringVar(value=str(Defaults().tr_v))
        self.var_tr_vce = tk.StringVar(value=str(Defaults().tr_vce_pct))
        # File vars
        self.var_vge = tk.StringVar(value=self.var_smu_v.get())
        self.var_temp = tk.StringVar(value=str(Defaults().temp_c))
        self.var_ncurves = tk.StringVar(value=str(Defaults().ncurves))

        self._build_widgets()
        self._update_connect_state()

    def _build_widgets(self) -> None:
        """Create and layout all widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

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

        self.warn_label = ttk.Label(
            conn,
            text=UI.WARN_CONNECT,
            foreground="#8a6d3b",
            background="#fcf8e3",
            padding=5,
        )
        self.warn_label.grid(row=3, column=0, columnspan=4, sticky=tk.W + tk.E, pady=(2, 8))

        ttk.Label(conn, text="Tek371:").grid(row=4, column=0, sticky=tk.W)
        self.tek_addr = ttk.Entry(conn, width=25)
        self.tek_addr.insert(0, Defaults().tek_address)
        self.tek_addr.grid(row=4, column=1, sticky=tk.W)

        ttk.Label(conn, text="Keithley 2400:").grid(row=4, column=2, sticky=tk.W)
        self.keithley_addr = ttk.Entry(conn, width=20)
        self.keithley_addr.insert(0, Defaults().k24_address)
        self.keithley_addr.grid(row=4, column=3, sticky=tk.W)

        ttk.Label(conn, text="Tek371:").grid(row=5, column=0, sticky=tk.W, pady=(8, 5))
        ttk.Button(conn, text="Connect", command=self.connect_tek).grid(row=5, column=1, padx=5, sticky=tk.W)
        self.tek_status = ttk.Label(conn, text="Not connected", foreground="gray", width=12)
        self.tek_status.grid(row=5, column=2, sticky=tk.W)

        ttk.Label(conn, text="Keithley 2400:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Button(conn, text="Connect", command=self.connect_keithley).grid(row=6, column=1, padx=5, sticky=tk.W)
        self.keithley_status = ttk.Label(conn, text="Not connected", foreground="gray", width=12)
        self.keithley_status.grid(row=6, column=2, sticky=tk.W)

        # ----- Measurement params -----
        param = ttk.LabelFrame(left, text="Measurement Parameters", padding="10")
        param.grid(row=1, column=0, sticky=tk.W, pady=5)

        ttk.Label(param, text=UI.L_SMU_V).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sb_smu_v = tk.Spinbox(param, from_=0.0, to=100.0, increment=0.1, width=12, textvariable=self.var_smu_v)
        self.sb_smu_v.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(param, text=UI.L_SMU_I_COMP).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sb_i_comp = tk.Spinbox(param, from_=0.01, to=1.0, increment=0.01, width=12, textvariable=self.var_i_comp)
        self.sb_i_comp.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(param, text=UI.L_TR_H).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.cb_tr_h = ttk.Combobox(param, values=H_CHOICES, width=11, state="readonly", textvariable=self.var_tr_h)
        self.cb_tr_h.set(str(Defaults().tr_h))
        self.cb_tr_h.grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(param, text=UI.L_TR_V).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.cb_tr_v = ttk.Combobox(param, values=V_CHOICES, width=11, state="readonly", textvariable=self.var_tr_v)
        self.cb_tr_v.set(str(Defaults().tr_v))
        self.cb_tr_v.grid(row=3, column=1, sticky=tk.W, padx=5)

        ttk.Label(param, text=UI.L_TR_VCE).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.sb_vce = tk.Spinbox(param, from_=0, to=100, increment=1, width=12, textvariable=self.var_tr_vce)
        self.sb_vce.grid(row=4, column=1, sticky=tk.W, padx=5)

        ttk.Label(param, text=UI.L_TR_PK_PWR).grid(row=5, column=0, sticky=tk.W, pady=2)
        self.lbl_pp = ttk.Label(param, text="300", foreground="#333")
        self.lbl_pp.grid(row=5, column=1, sticky=tk.W, padx=5)

        # ----- File settings -----
        filef = ttk.LabelFrame(left, text="File Settings", padding="10")
        filef.grid(row=2, column=0, sticky=tk.W, pady=5)

        ttk.Label(filef, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.folder_entry = ttk.Entry(filef, width=35)
        self.folder_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(filef, text="Browse", command=self.browse_folder).grid(row=0, column=2)

        self.file_entries: Dict[str, tk.Widget] = {}

        ttk.Label(filef, text=UI.L_DUT).grid(row=1, column=0, sticky=tk.W, pady=2)
        e_dut = ttk.Entry(filef, width=15); e_dut.insert(0, Defaults().dut)
        e_dut.grid(row=1, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_DUT] = e_dut

        ttk.Label(filef, text=UI.L_DEV).grid(row=2, column=0, sticky=tk.W, pady=2)
        e_dev = ttk.Entry(filef, width=15); e_dev.insert(0, Defaults().dev)
        e_dev.grid(row=2, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_DEV] = e_dev

        ttk.Label(filef, text=UI.L_VGE).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.var_vge.set(self.var_smu_v.get())
        e_vge = ttk.Entry(filef, width=15, textvariable=self.var_vge, state="readonly")
        e_vge.grid(row=3, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_VGE] = e_vge

        ttk.Label(filef, text=UI.L_TEMP).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.sb_temp = tk.Spinbox(filef, from_=-55, to=250, increment=1, width=15, textvariable=self.var_temp)
        self.sb_temp.grid(row=4, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_TEMP] = self.sb_temp

        ttk.Label(filef, text=UI.L_NCURVES).grid(row=5, column=0, sticky=tk.W, pady=2)
        self.sb_ncurves = tk.Spinbox(filef, from_=1, to=200, increment=1, width=15, textvariable=self.var_ncurves)
        self.sb_ncurves.grid(row=5, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_NCURVES] = self.sb_ncurves

        ttk.Button(filef, text="Export Settings", command=self.export_settings).grid(row=6, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Button(filef, text="Import Settings", command=self.import_settings).grid(row=6, column=1, sticky=tk.W, pady=(8, 0))

        # ----- Control Buttons -----
        btns = ttk.Frame(left); btns.grid(row=3, column=0, pady=10, sticky=tk.W + tk.E)
        self.start_btn = ttk.Button(btns, text="Start Measurement", command=self.start_measurement, state="disabled")
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_measurement, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.clear_btn = ttk.Button(btns, text="Clear Plot", command=self.clear_plot)
        self.clear_btn.grid(row=0, column=2, padx=5)

        # ----- Status -----
        status = ttk.LabelFrame(left, text="Status", padding="10")
        status.grid(row=4, column=0, sticky=tk.W + tk.E, pady=5); status.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status, text="Ready", relief=tk.SUNKEN, anchor="w", justify="left")
        self.status_label.grid(row=0, column=0, sticky=tk.E + tk.W)
        self.progress = ttk.Progressbar(status, length=400, mode="determinate")
        self.progress.grid(row=1, column=0, sticky=tk.E + tk.W, pady=(6, 0))
        status.bind("<Configure>", self._on_status_resize)

        # ----- Plot area -----
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        plot_frame = ttk.Frame(right)
        plot_frame.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        plot_frame.columnconfigure(0, weight=1); plot_frame.rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.fig.tight_layout()

        def _sync_vge(*_):
            self.var_vge.set(self.var_smu_v.get())
        self.var_smu_v.trace_add("write", _sync_vge)

    # ----- UI helpers -----
    def _style_axes(self) -> None:
        """Default axes style and labels."""
        self.ax.clear(); self.ax.set_xlabel("Voltage (V)"); self.ax.set_ylabel("Current (A)")
        self.ax.set_title("I-V Measurement Data"); self.ax.grid(True)

    def _clear_plot_only(self) -> None:
        """Clear axes without changing Status or Progress."""
        self._style_axes(); self.fig.tight_layout(); self.canvas.draw()

    def _on_status_resize(self, event) -> None:
        try:
            self.status_label.configure(wraplength=max(event.width - 20, 100))
        except Exception:
            pass

    def _set_status(self, message: str) -> None:
        """Update status text on UI thread."""
        self.status_label.config(text=message); self.root.update_idletasks()

    def _set_progress(self, percent: float) -> None:
        """Update progress bar value on UI thread."""
        self.progress["value"] = percent; self.root.update_idletasks()

    def _post(self, fn: Callable[[], None]) -> None:
        """Schedule `fn` to run on the Tk main thread."""
        self.root.after(0, fn)

    def _show_error(self, title: str, err: Exception) -> None:
        """Unified error dialog helper."""
        try:
            messagebox.showerror(title, str(err))
        except Exception:
            pass

    def _update_connect_state(self) -> None:
        both_connected = (self.tek371 is not None) and (self.keithley is not None)
        self.start_btn.config(state="normal" if both_connected else "disabled")
        self.warn_label.grid_remove() if both_connected else self.warn_label.grid()

    # ----- Settings import/export -----
    def collect_settings(self) -> Dict:
        """Collect current UI settings for JSON export."""
        addrs = {UI.K_TEK: self.tek_addr.get(), UI.K_K24: self.keithley_addr.get()}
        meas = {
            UI.L_SMU_V: self.var_smu_v.get(),
            UI.L_SMU_I_COMP: self.var_i_comp.get(),
            UI.L_TR_H: self.var_tr_h.get(),
            UI.L_TR_V: self.var_tr_v.get(),
            UI.L_TR_VCE: self.var_tr_vce.get(),
            UI.L_TR_PK_PWR: "300",
        }
        file_set = {
            "output_folder": self.folder_entry.get(),
            UI.L_DUT: self.file_entries[UI.L_DUT].get(),
            UI.L_DEV: self.file_entries[UI.L_DEV].get(),
            UI.L_VGE: self.var_vge.get(),
            UI.L_TEMP: self.var_temp.get(),
            UI.L_NCURVES: self.var_ncurves.get(),
        }
        base_filename = f"{file_set[UI.L_DUT]}_{file_set[UI.L_DEV]}_{self.var_vge.get()}V_{file_set[UI.L_TEMP]}C"
        return {
            "addresses": addrs,
            "measurement": meas,
            "file": file_set,
            "base_filename_example": base_filename,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def apply_settings(self, settings: dict) -> None:
        """Apply settings dict to UI controls."""
        try:
            if "addresses" in settings:
                addrs = settings["addresses"]
                if UI.K_TEK in addrs:
                    self.tek_addr.delete(0, tk.END); self.tek_addr.insert(0, addrs[UI.K_TEK])
                if UI.K_K24 in addrs:
                    self.keithley_addr.delete(0, tk.END); self.keithley_addr.insert(0, addrs[UI.K_K24])
            if "measurement" in settings:
                meas = settings["measurement"]
                if UI.L_SMU_V in meas: self.var_smu_v.set(str(meas[UI.L_SMU_V]))
                if UI.L_SMU_I_COMP in meas: self.var_i_comp.set(str(meas[UI.L_SMU_I_COMP]))
                if UI.L_TR_H in meas and str(meas[UI.L_TR_H]) in H_CHOICES: self.var_tr_h.set(str(meas[UI.L_TR_H]))
                if UI.L_TR_V in meas and str(meas[UI.L_TR_V]) in V_CHOICES: self.var_tr_v.set(str(meas[UI.L_TR_V]))
                if UI.L_TR_VCE in meas: self.var_tr_vce.set(str(meas[UI.L_TR_VCE]))
            if "file" in settings:
                fs = settings["file"]
                if "output_folder" in fs:
                    self.folder_entry.delete(0, tk.END); self.folder_entry.insert(0, fs["output_folder"])
                if UI.L_DUT in fs:
                    w = self.file_entries[UI.L_DUT]; w.delete(0, tk.END); w.insert(0, str(fs[UI.L_DUT]))
                if UI.L_DEV in fs:
                    w = self.file_entries[UI.L_DEV]; w.delete(0, tk.END); w.insert(0, str(fs[UI.L_DEV]))
                if UI.L_TEMP in fs: self.var_temp.set(str(fs[UI.L_TEMP]))
                if UI.L_NCURVES in fs: self.var_ncurves.set(str(fs[UI.L_NCURVES]))
                if UI.L_VGE in fs:
                    self.var_smu_v.set(str(fs[UI.L_VGE]))
            self._set_status("Settings applied successfully")
        except Exception as e:
            self._show_error("Apply Settings Error", e)

    def export_settings(self) -> None:
        try:
            data = self.collect_settings()
            default_name = f"settings_{data['file'][UI.L_DUT]}_{data['file'][UI.L_DEV]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = filedialog.asksaveasfilename(title="Export Settings", defaultextension=".json",
                                                filetypes=[("JSON files", "*.json")], initialfile=default_name)
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self._set_status(f"Settings exported to {path}")
        except Exception as e:
            self._show_error("Export Settings Error", e)

    def import_settings(self) -> None:
        try:
            path = filedialog.askopenfilename(title="Import Settings", filetypes=[("JSON files", "*.json")])
            if path:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.apply_settings(data)
                self._set_status(f"Settings imported from {path}")
        except Exception as e:
            self._show_error("Import Settings Error", e)

    # ----- Device ops -----
    def scan_gpib(self) -> None:
        try:
            rm = pyvisa.ResourceManager(); resources = rm.list_resources()
            gpib_devices = [r for r in resources if "GPIB" in r]
            self.gpib_text.config(state="normal"); self.gpib_text.delete(1.0, tk.END)
            if gpib_devices:
                self.gpib_text.insert(tk.END, "\n".join(gpib_devices)); self._set_status(f"Found {len(gpib_devices)} GPIB device(s)")
            else:
                self.gpib_text.insert(tk.END, "No GPIB devices found"); self._set_status("No GPIB devices found")
            self.gpib_text.config(state="disabled")
        except Exception as e:
            self._show_error("Error scanning GPIB", e)

    def connect_tek(self) -> None:
        try:
            self._set_status("Connecting to Tek371...")
            addr = self.tek_addr.get(); self.tek371 = Tek371(addr)
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
            self._show_error("Tek371 Connection Error", e)
            self._set_status("Tek371 connection failed")
        finally:
            self._update_connect_state()

    def connect_keithley(self) -> None:
        try:
            self._set_status("Connecting to Keithley 2400...")
            addr = self.keithley_addr.get(); self.keithley = Keithley2400(addr)
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
            self._show_error("Keithley 2400 Connection Error", e)
            self._set_status("Keithley connection failed")
        finally:
            self._update_connect_state()

    # ----- Plot ops -----
    def clear_plot(self) -> None:
        """Clear plot when idle; ignore during an active run."""
        if self.is_running:
            return
        self._style_axes(); self.fig.tight_layout(); self.canvas.draw();
        self._set_status("Plot cleared"); self._set_progress(0.0)

    def _plot_csv(self, path: Path) -> None:
        try:
            data = pd.read_csv(path)
            self.ax.plot(data.iloc[:, 0], data.iloc[:, 1], alpha=0.5, linewidth=1, color="blue")
            self.fig.tight_layout(); self.canvas.draw()
        except Exception as e:
            print(f"Plot error ({path.name}): {e}")

    def _plot_mean_csv(self, path: Path) -> None:
        try:
            data = pd.read_csv(path)
            self.ax.plot(data.iloc[:, 0], data.iloc[:, 1], "r-", linewidth=1.5, label="Mean")
            self.ax.legend(); self.fig.tight_layout(); self.canvas.draw()
        except Exception as e:
            print(f"Plot mean error ({path.name}): {e}")

    # ----- Run / Stop -----
    def build_settings(self) -> RunSettings:
        """Collect UI values and create a RunSettings object."""
        addrs = AddressSettings(tek371=self.tek_addr.get(), keithley2400=self.keithley_addr.get())
        p = MeasurementParams(
            smu_v=float(self.var_smu_v.get()),
            smu_i_comp_mA=float(self.var_i_comp.get()),
            tr_h=float(self.var_tr_h.get()),
            tr_v=float(self.var_tr_v.get()),
            tr_vce_pct=float(self.var_tr_vce.get()),
            tr_peak_power=300,
        )
        folder = self.folder_entry.get()
        if not folder:
            raise ValueError("Please select an output folder")
        f = FileParams(
            output_folder=Path(folder),
            dut=self.file_entries[UI.L_DUT].get(),
            dev=self.file_entries[UI.L_DEV].get(),
            vge=float(self.var_vge.get()),
            temp_c=float(self.var_temp.get()),
            ncurves=int(self.var_ncurves.get()),
        )
        return RunSettings(addresses=addrs, measurement=p, file=f)

    def start_measurement(self) -> None:
        """Validate, clear plot, start worker thread, and update button states."""
        if self.tek371 is None or self.keithley is None:
            self._update_connect_state(); return
        try:
            settings = self.build_settings(); settings.validate()
        except Exception as e:
            self._show_error("Invalid Parameters", e); return

        # Clear plot immediately at start (without touching status/progress)
        self._clear_plot_only()

        self.controller = MeasurementController(self.tek371, self.keithley)
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.clear_btn.config(state="disabled")

        # Wrap controller callbacks to main thread
        def on_status(msg: str) -> None: self._post(lambda: self._set_status(msg))
        def on_progress(pct: float) -> None: self._post(lambda: self._set_progress(pct))
        def on_plot_csv(path: Path) -> None: self._post(lambda: self._plot_csv(path))
        def on_plot_mean_csv(path: Path) -> None: self._post(lambda: self._plot_mean_csv(path))

        def work():
            try:
                self.controller.run(settings, on_status, on_progress, on_plot_csv, on_plot_mean_csv)
            except Exception as e:
                self._post(lambda: (self._set_status(UI.STATUS_ERROR), self._show_error("Measurement Error", e)))
            finally:
                self._post(lambda: (
                    setattr(self, "is_running", False),
                    self.start_btn.config(state="normal"),
                    self.stop_btn.config(state="disabled"),
                    self.clear_btn.config(state="normal"),
                ))
        self.worker_thread = Thread(target=work, daemon=True); self.worker_thread.start()

    def stop_measurement(self) -> None:
        """Signal the controller to stop and update status."""
        if self.controller:
            self.controller.stop()
        self._set_status("Stopping measurement...")

    def browse_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)


# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementGUI(root)
    root.mainloop()
