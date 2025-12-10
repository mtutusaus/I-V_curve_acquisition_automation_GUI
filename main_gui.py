# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from time import sleep
import time
from typing import Callable, Dict, Optional, List, Literal

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# External packages (available in your lab environment)
import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from tek371 import Tek371
from pymeasure.instruments.keithley import Keithley2400

warnings.filterwarnings("ignore", message="read string doesn't end with termination characters")

# =========================================
# UI text
# =========================================
class UI:
    TITLE = "I-V Measurement System"

    # Labels (I-V)
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

    # Gate-bias UI labels
    L_GATE_SRC = "Gate Bias Source:"
    L_STEP_V = "Step Voltage (V):"
    L_STEP_OFF = "Step Offset (×):"
    L_VGE_COMPUTED = "Computed Gate Voltage (V):"

    # Address keys (for JSON)
    K_TEK = "tek371"
    K_K24 = "keithley2400"

    # Banner
    WARN_CONNECT = "Must connect both equipments to enable start measurement button"

    # Status messages (I-V)
    STATUS_CONFIG_SMU = "Configuring Keithley 2400..."
    STATUS_CONFIG_TEK = "Configuring Tek371..."
    STATUS_START = "Starting measurements..."
    STATUS_MEAS = "Measuring curve {i}/{n}..."
    STATUS_STOPPED = "Measurement stopped by user"
    STATUS_MEAN = "Computing mean curve..."
    STATUS_DONE = "Measurement complete! Data saved to {folder}"
    STATUS_ERROR = "Measurement error"

    # --- TSEP ---
    TAB_TSEP = "Junction Temperature (TSEP)"
    TSEP_SCAN = "Scan GPIB Bus"
    TSEP_ADDRS = "Instrument Addresses"
    TSEP_ADDR_VGE = "Keithley 2400 Gate bias:"
    TSEP_ADDR_VCE = "Keithley 2400 current bias:"
    TSEP_CONNECT_VGE = "Connect"
    TSEP_CONNECT_VCE = "Connect"

    TSEP_VGE_SECTION = "VGE Settings"
    TSEP_VCE_SECTION = "VCE Settings"
    TSEP_OUTPUTS = "Outputs"

    TSEP_VGE_V = "Gate bias voltage (V):"
    TSEP_VGE_COMP = "Gate bias compliance current (mA):"
    TSEP_VCE_I = "Biasing current (mA):"
    TSEP_VCE_COMP = "Biasing compliance voltage (V):"

    TSEP_EQ = "Conversion Equation"
    TSEP_EQ_LINEAR = "Linear (Tj = a + b·V)"
    TSEP_EQ_QUAD = "Quadratic (Tj = a + b·V + c·V²)"
    TSEP_A = "a:"
    TSEP_B = "b:"
    TSEP_C = "c:"

    TSEP_MEASURE = "Measure temperature"

    TSEP_MEAN_V = "Voltage (V):"
    TSEP_TJ = "Temperature (°C):"

    TSEP_COPY_V = "Copy voltage"
    TSEP_COPY_T = "Copy temperature"

    TSEP_STATUS_READY = "Ready"

# =========================================
# Defaults & allowed values
# =========================================
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

# UI choices (Comboboxes)
H_CHOICES = ("0.1", "0.2", "0.5", "1", "2", "5")
V_CHOICES = ("0.5", "1", "2", "5")

# Internal gate choices
GATE_SRC_EXTERNAL = "External (SMU)"
GATE_SRC_INTERNAL = "Internal (Tek371)"
STEP_V_CHOICES = ("0.2", "0.5", "1", "2", "5")

# Numeric allowed sets
ALLOWED_H = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
ALLOWED_V = {0.5, 1.0, 2.0, 5.0}
ALLOWED_STEP_V = {0.2, 0.5, 1.0, 2.0, 5.0}

# TSEP fixed measurement constants
@dataclass
class TSEPDefaults:
    vce_gpib_address: str = "GPIB::25"
    vce_source_current_A: float = 150e-3
    vce_compliance_voltage_V: float = 2.0
    vce_measure_nplc: float = 10.0
    vce_measure_voltage_range_V: float = 2.0
    vce_buffer_count: int = 10
    use_4_wires: bool = True
    settle_s: float = 0.5

EquationType = Literal["linear", "quadratic"]

# =========================================
# Helpers
# =========================================
def _is_allowed(value: float, allowed: set[float], eps: float = 1e-9) -> bool:
    return any(abs(value - a) <= eps for a in allowed)

def try_zoomed(root: tk.Tk) -> None:
    try:
        root.state("zoomed")
    except Exception:
        root.geometry("1200x900")

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
    filepaths = [folder_path / f"{base_name}_{i}.csv" for i in range(1, N + 1)]
    if not all(p.exists() for p in filepaths):
        missing = [str(p.name) for p in filepaths if not p.exists()]
        raise FileNotFoundError(f"Missing curve files: {missing}")
    dfs: List[pd.DataFrame] = []
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
# Data models
# =========================================
@dataclass
class AddressSettings:
    tek371: str
    keithley2400: str

@dataclass
class MeasurementParams:
    smu_v: float
    smu_i_comp_mA: float
    tr_h: float
    tr_v: float
    tr_vce_pct: float
    tr_peak_power: int
    gate_source: str
    step_voltage: float
    step_offset: float

    def validate(self) -> None:
        if self.gate_source == GATE_SRC_EXTERNAL:
            require_positive(UI.L_SMU_V, self.smu_v)
            require_positive(UI.L_SMU_I_COMP, self.smu_i_comp_mA)
            if self.smu_i_comp_mA > 1.0:
                raise ValueError(f"{UI.L_SMU_I_COMP} must be ≤ 1.0 mA (got {self.smu_i_comp_mA} mA).")
        elif self.gate_source == GATE_SRC_INTERNAL:
            if not _is_allowed(self.step_voltage, ALLOWED_STEP_V):
                raise ValueError(f"{UI.L_STEP_V} must be one of {sorted(ALLOWED_STEP_V)} V (got {self.step_voltage}).")
            if not (0.0 <= self.step_offset <= 5.0):
                raise ValueError(f"{UI.L_STEP_OFF} must be between 0.0 and 5.0 (got {self.step_offset}).")
            vge_internal = self.step_voltage * self.step_offset
            if not (0.0 <= vge_internal <= 25.0):
                raise ValueError(f"Computed VGE must be within 0–25 V (got {vge_internal}).")
        else:
            raise ValueError("Invalid Gate Bias Source selected.")
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
        if not (0.0 <= self.vge <= 25.0):
            raise ValueError(f"{UI.L_VGE} must be between 0 and 25 V (got {self.vge}).")
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
# Controller: I-V measurement
# =========================================
class MeasurementController:
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
        gate_source = settings.measurement.gate_source

        # Configure instruments
        if gate_source == GATE_SRC_EXTERNAL:
            on_status(UI.STATUS_CONFIG_SMU)
            self._configure_keithley(settings)
            sleep(0.1)
        on_status(UI.STATUS_CONFIG_TEK)
        self._configure_tek(settings)
        sleep(0.1)
        on_status(UI.STATUS_START)

        if gate_source == GATE_SRC_EXTERNAL:
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
                filename = folder / (f"{base}.csv" if N == 1 else f"{base}_{i}.csv")
                self.tek.read_curve(str(filename))
                on_plot_csv(filename)
                # Reset SRQ for next iteration
                self.tek.discard_and_disable_all_events()
                self.tek.enable_srq_event()
                on_progress((i / N) * 100.0)
            # Compute and plot mean if not stopped (only when N > 1)
            if not self._stop_event.is_set():
                if N > 1:
                    on_status(UI.STATUS_MEAN)
                    mean_path = compute_mean_file(folder, base, N)
                    on_plot_mean_csv(mean_path)
                on_status(UI.STATUS_DONE.format(folder=folder))
                on_progress(100.0)
        finally:
            # Cleanup
            try:
                if settings.measurement.gate_source == GATE_SRC_EXTERNAL:
                    self.k24.disable_source()
            except Exception:
                pass
            try:
                self.k24.beep(4000, 2)
            except Exception:
                pass
            try:
                if settings.measurement.gate_source == GATE_SRC_INTERNAL:
                    self.tek.set_step_voltage(200e-3)
                    self.tek.set_step_offset(0)
            except Exception:
                pass
            try:
                self.tek.disable_srq_event()
            except Exception:
                pass

    # --- Instrument config helpers ---
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
        if p.gate_source == GATE_SRC_INTERNAL:
            self.tek.set_step_voltage(p.step_voltage)
            self.tek.set_step_offset(p.step_offset)
        else:
            self.tek.set_step_voltage(200e-3)
            self.tek.set_step_offset(0)
        self.tek.enable_srq_event()
        self.tek.set_horizontal("COL", p.tr_h)
        self.tek.set_vertical(p.tr_v)
        self.tek.set_display_mode("STO")

# =========================================
# TSEP data & controller (independent tab)
# =========================================
@dataclass
class TSEPParams:
    vge_gpib: str
    vce_gpib: str
    vge_voltage_V: float
    vge_compliance_mA: float
    vce_source_mA: float
    vce_compliance_V: float
    equation_type: EquationType
    a: float
    b: float
    c: float  # used only for quadratic

    def validate(self) -> None:
        if self.vce_source_mA <= 0:
            raise ValueError("VCE source current must be > 0 mA")
        if self.vce_compliance_V <= 0:
            raise ValueError("VCE compliance voltage must be > 0 V")
        if not (self.equation_type in ("linear", "quadratic")):
            raise ValueError("Equation type must be 'linear' or 'quadratic'")
        if self.vge_compliance_mA <= 0 or self.vge_compliance_mA > 1.0:
            raise ValueError("VGE compliance current must be in (0, 1.0] mA")

@dataclass
class TSEPResult:
    mean_voltage_V: float
    tj_celsius: float
    timestamp_iso: str

class TSEPController:
    def __init__(self) -> None:
        self.smu_vge: Optional[Keithley2400] = None
        self.smu_vce: Optional[Keithley2400] = None
        self._const = TSEPDefaults()

    def _configure_vge(self, p: TSEPParams) -> None:
        smu = self.smu_vge
        assert smu is not None
        smu.reset()
        smu.use_front_terminals()
        smu.source_mode = "voltage"
        smu.source_voltage = p.vge_voltage_V
        smu.compliance_current = p.vge_compliance_mA / 1000.0

    def _configure_vce(self, p: TSEPParams) -> None:
        smu = self.smu_vce
        assert smu is not None
        smu.reset()
        smu.use_front_terminals()
        smu.apply_current(self._const.vce_source_current_A, p.vce_compliance_V)
        smu.measure_voltage(self._const.vce_measure_nplc, self._const.vce_measure_voltage_range_V)
        if self._const.use_4_wires:
            smu.wires = 4

    def _cleanup(self) -> None:
        for smu in (self.smu_vce, self.smu_vge):
            try:
                smu.disable_source()
            except Exception:
                pass
        for smu in (self.smu_vce, self.smu_vge):
            try:
                smu.write("*CLS"); smu.write("*SRE 0")
            except Exception:
                pass
        try:
            if self.smu_vce:
                self.smu_vce.beep(4000, 2)
        except Exception:
            pass

    def run(self,
            params: TSEPParams,
            on_status: Callable[[str], None],
            on_progress: Callable[[float], None]) -> TSEPResult:
        params.validate()
        c = self._const
        on_status("Connecting TSEP instruments...")
        self.smu_vge = Keithley2400(params.vge_gpib)
        self.smu_vce = Keithley2400(params.vce_gpib)
        # Try IDs and show them via status
        try:
            idn_vge = getattr(self.smu_vge, "idn", None) or self.smu_vge.ask("*IDN?")
            on_status(f"VGE connected: {idn_vge}")
        except Exception:
            pass
        try:
            idn_vce = getattr(self.smu_vce, "idn", None) or self.smu_vce.ask("*IDN?")
            on_status(f"VCE connected: {idn_vce}")
        except Exception:
            pass
        # Configure
        on_status("Configuring VGE SMU...")
        self._configure_vge(params)
        on_status("Configuring VCE SMU...")
        # Override the fixed source current with user-controlled mA
        c.vce_source_current_A = params.vce_source_mA / 1000.0
        c.vce_compliance_voltage_V = params.vce_compliance_V
        self._configure_vce(params)
        sleep(c.settle_s)
        # Enable sources: VGE then VCE
        on_status("Enabling VGE...")
        self.smu_vge.enable_source()
        on_status("Enabling VCE...")
        self.smu_vce.enable_source()
        # Buffer measurement (fixed count)
        self.smu_vce.config_buffer(c.vce_buffer_count)
        self.smu_vce.source_current = c.vce_source_current_A
        on_status("Measuring VCE buffer...")
        self.smu_vce.start_buffer()
        self.smu_vce.wait_for_buffer()
        # Read mean voltage
        try:
            V = float(self.smu_vce.mean_voltage)
        except Exception:
            V = float(self.smu_vce.voltage)
        # Compute Tj
        if params.equation_type == "linear":
            tj = params.a + params.b * V
        else:
            tj = params.a + params.b * V + params.c * (V ** 2)
        result = TSEPResult(
            mean_voltage_V=V,
            tj_celsius=tj,
            timestamp_iso=datetime.now().isoformat(timespec="seconds")
        )
        self._cleanup()
        on_status(f"VCE @ {c.vce_source_current_A:.3f} A = {V:.6f} V ; Tj = {tj:.3f} °C")
        on_progress(100.0)
        return result

# =========================================
# GUI
# =========================================
class MeasurementGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(UI.TITLE)
        try_zoomed(self.root)
        self.root.minsize(1200, 900)
        # Notebook (two tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.iv_tab = ttk.Frame(self.notebook)
        self.tsep_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.iv_tab, text="I-V Measurement")
        self.notebook.add(self.tsep_tab, text=UI.TAB_TSEP)

        # Instruments
        self.tek371: Optional[Tek371] = None
        self.keithley: Optional[Keithley2400] = None
        # TSEP instruments
        self.tsep_vge_smu: Optional[Keithley2400] = None
        self.tsep_vce_smu: Optional[Keithley2400] = None

        # Controller / thread
        self.controller: Optional[MeasurementController] = None
        self.worker_thread: Optional[Thread] = None
        # Run state flag
        self.is_running: bool = False

        # Parameter variables (I-V)
        self.var_smu_v = tk.StringVar(value=str(Defaults().smu_v))
        self.var_i_comp = tk.StringVar(value=str(Defaults().smu_i_comp_mA))
        self.var_tr_h = tk.StringVar(value=str(Defaults().tr_h))
        self.var_tr_v = tk.StringVar(value=str(Defaults().tr_v))
        self.var_tr_vce = tk.StringVar(value=str(Defaults().tr_vce_pct))
        # Gate bias selection
        self.var_gate_source = tk.StringVar(value=GATE_SRC_EXTERNAL)
        self.var_step_voltage = tk.StringVar(value="0.2")
        self.var_step_offset = tk.StringVar(value="0.00")
        # File vars
        self.var_vge = tk.StringVar(value=self.var_smu_v.get())
        self.var_temp = tk.StringVar(value=str(Defaults().temp_c))
        self.var_ncurves = tk.StringVar(value=str(Defaults().ncurves))
        self.file_entries: Dict[str, tk.Widget] = {}

        # TSEP vars
        self.var_tsep_vge_addr = tk.StringVar(value="(linked)")
        self.var_tsep_vce_addr = tk.StringVar(value=TSEPDefaults().vce_gpib_address)
        self.var_tsep_vge = tk.StringVar(value=str(Defaults().smu_v))
        self.var_tsep_vge_comp = tk.StringVar(value=str(Defaults().smu_i_comp_mA))
        self.var_tsep_ic = tk.StringVar(value=str(int(TSEPDefaults().vce_source_current_A * 1000)))
        self.var_tsep_vcomp = tk.StringVar(value=str(TSEPDefaults().vce_compliance_voltage_V))
        self.var_tsep_eq = tk.StringVar(value="linear")
        self.var_tsep_a = tk.StringVar(value="357.090847511229")
        self.var_tsep_b = tk.StringVar(value="-532.214573058354")
        self.var_tsep_c = tk.StringVar(value="0.0")
        self.var_tsep_v_read = tk.StringVar(value="—")
        self.var_tsep_tj = tk.StringVar(value="—")

        # Heating period vars
        self.var_heat_minutes = tk.StringVar(value='15')
        self.var_setpoint_c = tk.StringVar(value='130')
        self.heating_running: bool = False
        self.heating_thread: Optional[Thread] = None
        self._heat_stop = Event()
        self._heat_data: List[tuple] = []  # (t_s, Tj_C)
        self.heat_start_time: Optional[float] = None
        # arrays for live scatter plotting + time axis
        self._heat_times: List[float] = []
        self._heat_tj_values: List[float] = []
        self._heat_time_axis: Optional[np.ndarray] = None

        # Build tabs
        self._build_iv_widgets()
        self._build_tsep_widgets()
        self._update_connect_state()

    # -----------------------
    # Build I-V widgets
    def _build_iv_widgets(self) -> None:
        main_frame = ttk.Frame(self.iv_tab, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.iv_tab.columnconfigure(0, weight=1)
        self.iv_tab.rowconfigure(0, weight=1)

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
        self.gpib_text.grid(row=1, column=0, columnspan=6, pady=5, sticky=tk.W)

        addrs_label = ttk.Label(conn, text="Instrument Addresses", font=("Helvetica", 10, "bold"))
        addrs_label.grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=(8, 2))

        self.warn_label = ttk.Label(conn, text=UI.WARN_CONNECT, foreground="#8a6d3b", background="#fcf8e3", padding=5)
        self.warn_label.grid(row=3, column=0, columnspan=6, sticky=tk.W + tk.E, pady=(2, 8))

        # Tek371 row
        ttk.Label(conn, text="Tek371:").grid(row=4, column=0, sticky=tk.W)
        self.tek_addr = ttk.Entry(conn, width=25)
        self.tek_addr.insert(0, Defaults().tek_address)
        self.tek_addr.grid(row=4, column=1, sticky=tk.W)
        ttk.Button(conn, text="Connect", command=self.connect_tek).grid(row=4, column=2, padx=5, sticky=tk.W)
        self.tek_status = ttk.Label(conn, text="Not connected", foreground="gray", width=20)
        self.tek_status.grid(row=4, column=3, sticky=tk.W)

        # Keithley row
        ttk.Label(conn, text="Keithley 2400:").grid(row=5, column=0, sticky=tk.W)
        self.keithley_addr = ttk.Entry(conn, width=25)
        self.keithley_addr.insert(0, Defaults().k24_address)
        self.keithley_addr.grid(row=5, column=1, sticky=tk.W)
        ttk.Button(conn, text="Connect", command=self.connect_keithley).grid(row=5, column=2, padx=5, sticky=tk.W)
        self.keithley_status = ttk.Label(conn, text="Not connected", foreground="gray", width=20)
        self.keithley_status.grid(row=5, column=3, sticky=tk.W)

        # ----- Gate Bias Parameters -----
        gatef = ttk.LabelFrame(left, text="Gate Bias Parameters", padding="10")
        gatef.grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(gatef, text=UI.L_GATE_SRC).grid(row=0, column=0, sticky=tk.W, pady=(2, 2))
        self.cb_gate_src = ttk.Combobox(gatef, values=(GATE_SRC_EXTERNAL, GATE_SRC_INTERNAL), state="readonly", width=18, textvariable=self.var_gate_source)
        self.cb_gate_src.grid(row=0, column=1, sticky=tk.W, padx=5)
        # External (SMU) controls
        ttk.Label(gatef, text=UI.L_SMU_V).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sb_smu_v = tk.Spinbox(gatef, from_=-100.0, to=100.0, increment=0.1, width=12, textvariable=self.var_smu_v)
        self.sb_smu_v.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(gatef, text=UI.L_SMU_I_COMP).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.sb_i_comp = tk.Spinbox(gatef, from_=0.01, to=1.0, increment=0.01, width=12, textvariable=self.var_i_comp)
        self.sb_i_comp.grid(row=2, column=1, sticky=tk.W, padx=5)
        # Internal (Tek371) controls
        ttk.Label(gatef, text=UI.L_STEP_V).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.cb_step_v = ttk.Combobox(gatef, values=STEP_V_CHOICES, state="readonly", width=11, textvariable=self.var_step_voltage)
        self.cb_step_v.grid(row=3, column=1, sticky=tk.W, padx=5)
        ttk.Label(gatef, text=UI.L_STEP_OFF).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.sb_step_off = tk.Spinbox(gatef, from_=0.0, to=5.0, increment=0.01, width=12, textvariable=self.var_step_offset)
        self.sb_step_off.grid(row=4, column=1, sticky=tk.W, padx=5)
        ttk.Label(gatef, text=UI.L_VGE_COMPUTED).grid(row=5, column=0, sticky=tk.W, pady=2)
        self.entry_vge_display = ttk.Entry(gatef, textvariable=self.var_vge, state="readonly", width=12)
        self.entry_vge_display.grid(row=5, column=1, sticky=tk.W, padx=5)

        # ----- Tracer Parameters -----
        param = ttk.LabelFrame(left, text="Tracer Parameters", padding="10")
        param.grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(param, text=UI.L_TR_H).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cb_tr_h = ttk.Combobox(param, values=H_CHOICES, width=11, state="readonly", textvariable=self.var_tr_h)
        self.cb_tr_h.set(str(Defaults().tr_h))
        self.cb_tr_h.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(param, text=UI.L_TR_V).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cb_tr_v = ttk.Combobox(param, values=V_CHOICES, width=11, state="readonly", textvariable=self.var_tr_v)
        self.cb_tr_v.set(str(Defaults().tr_v))
        self.cb_tr_v.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(param, text=UI.L_TR_VCE).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.sb_vce = tk.Spinbox(param, from_=0, to=100, increment=1, width=12, textvariable=self.var_tr_vce)
        self.sb_vce.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(param, text=UI.L_TR_PK_PWR).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.lbl_pp = ttk.Label(param, text="300", foreground="#333")
        self.lbl_pp.grid(row=3, column=1, sticky=tk.W, padx=5)

        # ----- File Settings -----
        filef = ttk.LabelFrame(left, text="File Settings", padding="10")
        filef.grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Label(filef, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.folder_entry = ttk.Entry(filef, width=35)
        self.folder_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(filef, text="Browse", command=self.browse_folder).grid(row=0, column=2)
        ttk.Label(filef, text=UI.L_DUT).grid(row=1, column=0, sticky=tk.W, pady=2)
        e_dut = ttk.Entry(filef, width=15); e_dut.insert(0, Defaults().dut)
        e_dut.grid(row=1, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_DUT] = e_dut
        ttk.Label(filef, text=UI.L_DEV).grid(row=2, column=0, sticky=tk.W, pady=2)
        e_dev = ttk.Entry(filef, width=15); e_dev.insert(0, Defaults().dev)
        e_dev.grid(row=2, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_DEV] = e_dev
        ttk.Label(filef, text=UI.L_TEMP).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.sb_temp = tk.Spinbox(filef, from_=-55, to=250, increment=1, width=15, textvariable=self.var_temp)
        self.sb_temp.grid(row=3, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_TEMP] = self.sb_temp
        ttk.Label(filef, text=UI.L_NCURVES).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.sb_ncurves = tk.Spinbox(filef, from_=1, to=200, increment=1, width=15, textvariable=self.var_ncurves)
        self.sb_ncurves.grid(row=4, column=1, sticky=tk.W, padx=5); self.file_entries[UI.L_NCURVES] = self.sb_ncurves
        ttk.Button(filef, text="Export Settings", command=self.export_settings).grid(row=5, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Button(filef, text="Import Settings", command=self.import_settings).grid(row=5, column=1, sticky=tk.W, pady=(8, 0))

        # ----- Control Buttons -----
        btns = ttk.Frame(left); btns.grid(row=4, column=0, pady=10, sticky=tk.W + tk.E)
        self.start_btn = ttk.Button(btns, text="Start Measurement", command=self.start_measurement, state="disabled")
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_measurement, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.clear_btn = ttk.Button(btns, text="Clear Plot", command=self.clear_plot)
        self.clear_btn.grid(row=0, column=2, padx=5)

        # ----- Status -----
        status = ttk.LabelFrame(left, text="Status", padding="10")
        status.grid(row=5, column=0, sticky=tk.W + tk.E, pady=5); status.columnconfigure(0, weight=1)
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

        # Traces & initial state
        def _sync_vge(*_):
            self._sync_vge_source()
        self.var_smu_v.trace_add("write", _sync_vge)
        self.var_gate_source.trace_add("write", lambda *_: self._update_gate_controls())
        self.var_step_voltage.trace_add("write", lambda *_: self._sync_vge_source())
        self.var_step_offset.trace_add("write", lambda *_: self._sync_vge_source())
        self._update_gate_controls()
        self._sync_vge_source()

    # -----------------------
    # Build TSEP tab (with heating plot on right)
    def _build_tsep_widgets(self) -> None:
        main = ttk.Frame(self.tsep_tab, padding="10")
        main.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.tsep_tab.columnconfigure(0, weight=1)
        self.tsep_tab.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.grid(row=0, column=0, sticky=tk.N + tk.W, padx=(0, 10))
        right.grid(row=0, column=1, sticky=tk.N + tk.S + tk.E + tk.W)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # --- Connection & addresses ---
        conn = ttk.LabelFrame(left, text=UI.TSEP_ADDRS, padding="10")
        conn.grid(row=0, column=0, sticky=tk.W + tk.E, pady=5)
        ttk.Button(conn, text=UI.TSEP_SCAN, command=self.tsep_scan_gpib).grid(row=0, column=0, pady=5, sticky=tk.W)
        ttk.Label(conn, text="Available devices:").grid(row=0, column=1, sticky=tk.W)
        self.tsep_gpib_text = scrolledtext.ScrolledText(conn, height=3, width=50, state="disabled")
        self.tsep_gpib_text.grid(row=1, column=0, columnspan=6, pady=5, sticky=tk.W)

        ttk.Label(conn, text=UI.TSEP_ADDR_VGE).grid(row=2, column=0, sticky=tk.W)
        self.tsep_addr_vge_entry = ttk.Entry(conn, width=20, state="disabled")  # disabled, mirrors I-V
        self.tsep_addr_vge_entry.insert(0, self.keithley_addr.get())
        self.tsep_addr_vge_entry.grid(row=2, column=1, sticky=tk.W)
        ttk.Button(conn, text=UI.TSEP_CONNECT_VGE, command=self.tsep_connect_vge).grid(row=2, column=2, padx=5, sticky=tk.W)
        self.tsep_vge_status = ttk.Label(conn, text="Not connected", foreground="gray", width=24)
        self.tsep_vge_status.grid(row=2, column=3, sticky=tk.W)

        ttk.Label(conn, text=UI.TSEP_ADDR_VCE).grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        self.tsep_addr_vce_entry = ttk.Entry(conn, width=20)
        self.tsep_addr_vce_entry.insert(0, self.var_tsep_vce_addr.get())
        self.tsep_addr_vce_entry.grid(row=3, column=1, sticky=tk.W)
        ttk.Button(conn, text=UI.TSEP_CONNECT_VCE, command=self.tsep_connect_vce).grid(row=3, column=2, padx=5, sticky=tk.W)
        self.tsep_vce_status = ttk.Label(conn, text="Not connected", foreground="gray", width=24)
        self.tsep_vce_status.grid(row=3, column=3, sticky=tk.W)

        # --- VGE settings ---
        vgef = ttk.LabelFrame(left, text=UI.TSEP_VGE_SECTION, padding="10")
        vgef.grid(row=1, column=0, sticky=tk.W + tk.E, pady=5)
        ttk.Label(vgef, text=UI.TSEP_VGE_V).grid(row=0, column=0, sticky=tk.W)
        tk.Spinbox(vgef, from_=-25.0, to=25.0, increment=0.1, width=10, textvariable=self.var_tsep_vge).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(vgef, text=UI.TSEP_VGE_COMP).grid(row=1, column=0, sticky=tk.W)
        tk.Spinbox(vgef, from_=0.01, to=1.0, increment=0.01, width=10, textvariable=self.var_tsep_vge_comp).grid(row=1, column=1, sticky=tk.W)

        # --- VCE settings ---
        vcef = ttk.LabelFrame(left, text=UI.TSEP_VCE_SECTION, padding="10")
        vcef.grid(row=2, column=0, sticky=tk.W + tk.E, pady=5)
        ttk.Label(vcef, text=UI.TSEP_VCE_I).grid(row=0, column=0, sticky=tk.W)
        tk.Spinbox(vcef, from_=1.0, to=1000.0, increment=1.0, width=10, textvariable=self.var_tsep_ic).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(vcef, text=UI.TSEP_VCE_COMP).grid(row=1, column=0, sticky=tk.W)
        tk.Spinbox(vcef, from_=0.1, to=10.0, increment=0.1, width=10, textvariable=self.var_tsep_vcomp).grid(row=1, column=1, sticky=tk.W)

        # --- Equation ---
        eqf = ttk.LabelFrame(left, text=UI.TSEP_EQ, padding="10")
        eqf.grid(row=3, column=0, sticky=tk.W + tk.E, pady=5)
        self.var_tsep_eq.trace_add("write", lambda *_: self._toggle_c_entry())
        rb_lin = ttk.Radiobutton(eqf, text=UI.TSEP_EQ_LINEAR, variable=self.var_tsep_eq, value="linear")
        rb_quad = ttk.Radiobutton(eqf, text=UI.TSEP_EQ_QUAD, variable=self.var_tsep_eq, value="quadratic")
        rb_lin.grid(row=0, column=0, sticky=tk.W)
        rb_quad.grid(row=0, column=1, sticky=tk.W)
        a_frame = ttk.Frame(eqf); a_frame.grid(row=1, column=0, sticky=tk.W, padx=(0,0))
        ttk.Label(a_frame, text=UI.TSEP_A).pack(side=tk.LEFT, padx=(0,2))
        ttk.Entry(a_frame, textvariable=self.var_tsep_a, width=18).pack(side=tk.LEFT)
        b_frame = ttk.Frame(eqf); b_frame.grid(row=1, column=1, sticky=tk.W, padx=(12,0))
        ttk.Label(b_frame, text=UI.TSEP_B).pack(side=tk.LEFT, padx=(0,2))
        ttk.Entry(b_frame, textvariable=self.var_tsep_b, width=18).pack(side=tk.LEFT)
        c_frame = ttk.Frame(eqf); c_frame.grid(row=1, column=2, sticky=tk.W, padx=(12,0))
        ttk.Label(c_frame, text=UI.TSEP_C).pack(side=tk.LEFT, padx=(0,2))
        self.entry_tsep_c = ttk.Entry(c_frame, textvariable=self.var_tsep_c, width=18)
        self.entry_tsep_c.pack(side=tk.LEFT)
        self._toggle_c_entry()

        # --- Actions ---
        actions = ttk.Frame(left)
        actions.grid(row=4, column=0, sticky=tk.W + tk.E, pady=(10, 0))
        self.btn_tsep_run = ttk.Button(actions, text=UI.TSEP_MEASURE, command=self.run_tsep)
        self.btn_tsep_run.grid(row=0, column=0, padx=5, sticky=tk.W)

        # --- Outputs ---
        outputs = ttk.LabelFrame(left, text=UI.TSEP_OUTPUTS, padding="10")
        outputs.grid(row=5, column=0, sticky=tk.W + tk.E, pady=5)
        ttk.Label(outputs, text=UI.TSEP_MEAN_V).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(outputs, textvariable=self.var_tsep_v_read, width=18, state="readonly").grid(row=0, column=1, sticky=tk.W)
        self.btn_tsep_copy_v = ttk.Button(outputs, text=UI.TSEP_COPY_V, command=self.copy_tsep_voltage)
        self.btn_tsep_copy_v.grid(row=1, column=1, sticky=tk.W, pady=(2, 10))
        ttk.Label(outputs, text=UI.TSEP_TJ).grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(outputs, textvariable=self.var_tsep_tj, width=18, state="readonly").grid(row=2, column=1, sticky=tk.W)
        self.btn_tsep_copy_t = ttk.Button(outputs, text=UI.TSEP_COPY_T, command=self.copy_tsep_temperature)
        self.btn_tsep_copy_t.grid(row=3, column=1, sticky=tk.W, pady=(2, 0))
        self.btn_tsep_clear = ttk.Button(outputs, text="Clear outputs", command=self.clear_tsep_outputs)
        self.btn_tsep_clear.grid(row=4, column=1, sticky=tk.W, pady=(8, 0))

        # --- Heating period controls ---
        heatf = ttk.LabelFrame(left, text="Heating period", padding="10")
        heatf.grid(row=6, column=0, sticky=tk.W + tk.E, pady=5)
        ttk.Label(heatf, text="Heating time (minutes):").grid(row=0, column=0, sticky=tk.W)
        tk.Spinbox(heatf, from_=1, to=240, increment=1, width=8, textvariable=self.var_heat_minutes).grid(row=0, column=1, sticky=tk.W, padx=(4,0))
        ttk.Label(heatf, text="Setpoint (°C):").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        tk.Spinbox(heatf, from_=-55, to=300, increment=1, width=8, textvariable=self.var_setpoint_c).grid(row=0, column=3, sticky=tk.W, padx=(4,0))
        self.btn_heat_start = ttk.Button(heatf, text="Start heating period measurement", command=self.start_heating)
        self.btn_heat_start.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(8,2))
        self.btn_heat_stop = ttk.Button(heatf, text="Stop", command=self.stop_heating, state="disabled")
        self.btn_heat_stop.grid(row=1, column=2, sticky=tk.W, pady=(8,2), padx=(10,0))
        self.btn_heat_export = ttk.Button(heatf, text="Export to CSV", command=self.export_heating_csv, state="disabled")
        self.btn_heat_export.grid(row=2, column=0, sticky=tk.W, pady=(4,0))
        self.btn_heat_clearplot = ttk.Button(heatf, text="Clear plot", command=self.clear_heating_plot)
        self.btn_heat_clearplot.grid(row=2, column=1, sticky=tk.W, pady=(4,0))

        # --- Status + Progress ---
        statf = ttk.LabelFrame(left, text="Status", padding="10")
        statf.grid(row=7, column=0, sticky=tk.W + tk.E, pady=5)
        self.tsep_status = ttk.Label(statf, text=UI.TSEP_STATUS_READY, relief=tk.SUNKEN, anchor="w", justify="left")
        self.tsep_status.grid(row=0, column=0, sticky=tk.E + tk.W)
        self.tsep_progress = ttk.Progressbar(statf, mode="determinate", length=400)
        self.tsep_progress.grid(row=1, column=0, sticky=tk.E + tk.W, pady=(6, 0))
        statf.columnconfigure(0, weight=1)
        statf.bind("<Configure>", self._on_tsep_status_resize)

        # --- Right plot area ---
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        plotf = ttk.Frame(right)
        plotf.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        plotf.columnconfigure(0, weight=1); plotf.rowconfigure(0, weight=1)
        self.fig_heat, self.ax_heat = plt.subplots(figsize=(7, 5), dpi=100)
        self._style_heat_axes()
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=plotf)
        self.canvas_heat.get_tk_widget().grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.fig_heat.tight_layout()
        # Single scatter artist for live Tj points
        self.heat_scatter = self.ax_heat.scatter([], [], s=12, c='red', marker='o', label='Tj')
        self.ax_heat.legend(loc='upper right')
        self.var_setpoint_c.trace_add("write", lambda *_: self._draw_setpoint_line())
        self._draw_setpoint_line()

    # -----------------------
    # Common UI helpers
    def _style_axes(self) -> None:
        self.ax.clear(); self.ax.set_xlabel("Voltage (V)"); self.ax.set_ylabel("Current (A)")
        self.ax.set_title("I-V Measurement Data"); self.ax.grid(True)

    def _style_heat_axes(self) -> None:
        self.ax_heat.clear()
        self.ax_heat.set_xlabel("Time (s)")
        self.ax_heat.set_ylabel("Junction temperature (°C)")
        self.ax_heat.set_title("Heating period: Tj vs time")
        self.ax_heat.grid(True)
        self.fig_heat.tight_layout()
        if hasattr(self, 'canvas_heat'):
            self.canvas_heat.draw()

    def _draw_setpoint_line(self) -> None:
        try:
            sp = float(self.var_setpoint_c.get())
        except Exception:
            sp = None
        if not hasattr(self, 'ax_heat'):
            return
        # remove existing setpoint line if present
        if hasattr(self, '_heat_setpoint_line') and self._heat_setpoint_line in self.ax_heat.lines:
            try:
                self._heat_setpoint_line.remove()
            except Exception:
                pass
        if sp is not None:
            self._heat_setpoint_line = self.ax_heat.axhline(sp, color='orange', linestyle='--', linewidth=1.3, label='Setpoint')
            # Make sure the setpoint line does NOT affect autoscaling
            self._heat_setpoint_line.set_zorder(0)
            self._heat_setpoint_line.set_alpha(0.9)
            handles, labels = self.ax_heat.get_legend_handles_labels()
            if 'Setpoint' in labels:
                self.ax_heat.legend(loc='upper right')
        if hasattr(self, 'canvas_heat'):
            self.canvas_heat.draw_idle()

    def _clear_plot_only(self) -> None:
        self._style_axes(); self.fig.tight_layout(); self.canvas.draw()

    def _on_status_resize(self, event) -> None:
        try:
            self.status_label.configure(wraplength=max(event.width - 20, 100))
        except Exception:
            pass

    def _on_tsep_status_resize(self, event) -> None:
        try:
            self.tsep_status.configure(wraplength=max(event.width - 20, 100))
        except Exception:
            pass

    def _set_status(self, message: str) -> None:
        self.status_label.config(text=message); self.root.update_idletasks()

    def _set_tsep_status(self, msg: str) -> None:
        self.tsep_status.config(text=msg); self.root.update_idletasks()

    def _set_progress(self, percent: float) -> None:
        self.progress["value"] = percent; self.root.update_idletasks()

    def _set_tsep_progress(self, pct: float) -> None:
        self.tsep_progress["value"] = pct; self.root.update_idletasks()

    def _post(self, fn: Callable[[], None]) -> None:
        self.root.after(0, fn)

    def _show_error(self, title: str, err: Exception) -> None:
        try:
            messagebox.showerror(title, str(err))
        except Exception:
            pass

    def _update_connect_state(self) -> None:
        # I-V tab start button gating
        src_mode = self.var_gate_source.get() if hasattr(self, "var_gate_source") else GATE_SRC_EXTERNAL
        need_k = (src_mode == GATE_SRC_EXTERNAL)
        ok = (self.tek371 is not None) and ((self.keithley is not None) if need_k else True)
        self.start_btn.config(state="normal" if ok else "disabled")
        try:
            if need_k:
                self.warn_label.config(text=UI.WARN_CONNECT)
            else:
                self.warn_label.config(text="Must connect Tek371 to enable start measurement button")
        except Exception:
            pass
        self.warn_label.grid_remove() if ok else self.warn_label.grid()
        # Sync TSEP VGE address entry (disabled) with I-V address
        try:
            self.tsep_addr_vge_entry.config(state="normal")
            self.tsep_addr_vge_entry.delete(0, tk.END)
            self.tsep_addr_vge_entry.insert(0, self.keithley_addr.get())
            self.tsep_addr_vge_entry.config(state="disabled")
        except Exception:
            pass

    # -----------------------
    # Settings import/export (includes TSEP)
    def collect_settings(self) -> Dict:
        addrs = {UI.K_TEK: self.tek_addr.get(), UI.K_K24: self.keithley_addr.get()}
        meas = {
            UI.L_SMU_V: self.var_smu_v.get(),
            UI.L_SMU_I_COMP: self.var_i_comp.get(),
            UI.L_TR_H: self.var_tr_h.get(),
            UI.L_TR_V: self.var_tr_v.get(),
            UI.L_TR_VCE: self.var_tr_vce.get(),
            UI.L_TR_PK_PWR: "300",
            "Gate Bias Source": self.var_gate_source.get(),
            "Step Voltage (V)": self.var_step_voltage.get(),
            "Step Offset": self.var_step_offset.get(),
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
        # TSEP settings
        tsep = {
            "vge_gpib": self.keithley_addr.get(),  # linked to I-V tab
            "vce_gpib": self.tsep_addr_vce_entry.get(),
            "vge_voltage": self.var_tsep_vge.get(),
            "vge_compliance_mA": self.var_tsep_vge_comp.get(),
            "vce_source_mA": self.var_tsep_ic.get(),
            "vce_compliance_V": self.var_tsep_vcomp.get(),
            "equation_type": self.var_tsep_eq.get(),
            "a": self.var_tsep_a.get(),
            "b": self.var_tsep_b.get(),
            "c": self.var_tsep_c.get(),
        }
        return {
            "addresses": addrs,
            "measurement": meas,
            "file": file_set,
            "tsep": tsep,
            "base_filename_example": base_filename,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def apply_settings(self, settings: dict) -> None:
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
                if "Gate Bias Source" in meas: self.var_gate_source.set(str(meas["Gate Bias Source"]))
                if "Step Voltage (V)" in meas: self.var_step_voltage.set(str(meas["Step Voltage (V)"]))
                if "Step Offset" in meas: self.var_step_offset.set(str(meas["Step Offset"]))
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
            # TSEP block
            if "tsep" in settings:
                ts = settings["tsep"]
                if "vce_gpib" in ts:
                    self.tsep_addr_vce_entry.delete(0, tk.END); self.tsep_addr_vce_entry.insert(0, str(ts["vce_gpib"]))
                if "vge_voltage" in ts: self.var_tsep_vge.set(str(ts["vge_voltage"]))
                if "vge_compliance_mA" in ts: self.var_tsep_vge_comp.set(str(ts["vge_compliance_mA"]))
                if "vce_source_mA" in ts: self.var_tsep_ic.set(str(ts["vce_source_mA"]))
                if "vce_compliance_V" in ts: self.var_tsep_vcomp.set(str(ts["vce_compliance_V"]))
                if "equation_type" in ts: self.var_tsep_eq.set(str(ts["equation_type"]))
                if "a" in ts: self.var_tsep_a.set(str(ts["a"]))
                if "b" in ts: self.var_tsep_b.set(str(ts["b"]))
                if "c" in ts: self.var_tsep_c.set(str(ts["c"]))
            # Derived & sync
            self._update_gate_controls(); self._sync_vge_source(); self._update_connect_state()
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

    # -----------------------
    # Device ops (I-V tab)
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

    # -----------------------
    # TSEP device ops
    def tsep_scan_gpib(self) -> None:
        try:
            rm = pyvisa.ResourceManager(); resources = rm.list_resources()
            gpib_devices = [r for r in resources if "GPIB" in r]
            self.tsep_gpib_text.config(state="normal"); self.tsep_gpib_text.delete(1.0, tk.END)
            if gpib_devices:
                self.tsep_gpib_text.insert(tk.END, "\n".join(gpib_devices)); self._set_tsep_status(f"Found {len(gpib_devices)} GPIB device(s)")
            else:
                self.tsep_gpib_text.insert(tk.END, "No GPIB devices found"); self._set_tsep_status("No GPIB devices found")
            self.tsep_gpib_text.config(state="disabled")
        except Exception as e:
            self._show_error("Error scanning GPIB (TSEP)", e)

    def tsep_connect_vge(self) -> None:
        try:
            addr = self.keithley_addr.get()
            self._set_tsep_status("Connecting VGE SMU...")
            self.tsep_vge_smu = Keithley2400(addr)
            try:
                idn = getattr(self.tsep_vge_smu, "idn", None) or self.tsep_vge_smu.ask("*IDN?")
            except Exception:
                idn = None
            if not idn:
                raise RuntimeError("VGE SMU did not respond to *IDN? or idn")
            # show 'Connected'
            self.tsep_vge_status.config(text="Connected", foreground="green")
            self._set_tsep_status(f"VGE connected successfully: {idn}")
        except Exception as e:
            self.tsep_vge_status.config(text="Error", foreground="red")
            self._show_error("VGE Connection Error", e)
            self._set_tsep_status("VGE SMU connection failed")

    def tsep_connect_vce(self) -> None:
        try:
            addr = self.tsep_addr_vce_entry.get()
            self._set_tsep_status("Connecting VCE SMU...")
            self.tsep_vce_smu = Keithley2400(addr)
            try:
                idn = getattr(self.tsep_vce_smu, "idn", None) or self.tsep_vce_smu.ask("*IDN?")
            except Exception:
                idn = None
            if not idn:
                raise RuntimeError("VCE SMU did not respond to *IDN? or idn")
            self.tsep_vce_status.config(text="Connected", foreground="green")
            self._set_tsep_status(f"VCE connected successfully: {idn}")
        except Exception as e:
            self.tsep_vce_status.config(text="Error", foreground="red")
            self._show_error("VCE Connection Error", e)
            self._set_tsep_status("VCE SMU connection failed")

    # -----------------------
    # Plot ops (I-V)
    def clear_plot(self) -> None:
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

    # -----------------------
    # Run / Stop (I-V)
    def build_settings(self) -> RunSettings:
        addrs = AddressSettings(tek371=self.tek_addr.get(), keithley2400=self.keithley_addr.get())
        p = MeasurementParams(
            smu_v=float(self.var_smu_v.get()),
            smu_i_comp_mA=float(self.var_i_comp.get()),
            tr_h=float(self.var_tr_h.get()),
            tr_v=float(self.var_tr_v.get()),
            tr_vce_pct=float(self.var_tr_vce.get()),
            tr_peak_power=300,
            gate_source=self.var_gate_source.get(),
            step_voltage=float(self.var_step_voltage.get()) if self.var_gate_source.get() == GATE_SRC_INTERNAL else 0.0,
            step_offset=float(self.var_step_offset.get()) if self.var_gate_source.get() == GATE_SRC_INTERNAL else 0.0,
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
        src_mode = self.var_gate_source.get()
        if self.tek371 is None or (src_mode == GATE_SRC_EXTERNAL and self.keithley is None):
            self._update_connect_state(); return
        try:
            settings = self.build_settings(); settings.validate()
        except Exception as e:
            self._show_error("Invalid Parameters", e); return
        self._clear_plot_only()
        self.controller = MeasurementController(self.tek371, self.keithley if self.keithley else Keithley2400("GPIB::0"))
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.clear_btn.config(state="disabled")
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
        if self.controller:
            self.controller.stop()
            self._set_status("Stopping measurement...")

    def browse_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)

    # Gate-bias helpers (I-V)
    def _update_gate_controls(self) -> None:
        internal = (self.var_gate_source.get() == GATE_SRC_INTERNAL)
        try:
            self.cb_step_v.configure(state=("normal" if internal else "disabled"))
            self.sb_step_off.configure(state=("normal" if internal else "disabled"))
            self.sb_smu_v.configure(state=("disabled" if internal else "normal"))
            self.sb_i_comp.configure(state=("disabled" if internal else "normal"))
        except Exception:
            pass
        self._update_connect_state(); self._sync_vge_source()

    def _sync_vge_source(self) -> None:
        if self.var_gate_source.get() == GATE_SRC_INTERNAL:
            try:
                sv = float(self.var_step_voltage.get())
                so = float(self.var_step_offset.get())
                vge = sv * so
                self.var_vge.set(f"{vge:.3f}")
            except Exception:
                self.var_vge.set("0.000")
        else:
            self.var_vge.set(self.var_smu_v.get())

    # -----------------------
    # TSEP actions
    def _toggle_c_entry(self) -> None:
        try:
            self.entry_tsep_c.configure(state=("normal" if self.var_tsep_eq.get() == "quadratic" else "disabled"))
        except Exception:
            pass

    def run_tsep(self) -> None:
        try:
            p = TSEPParams(
                vge_gpib=self.keithley_addr.get(),  # linked
                vce_gpib=self.tsep_addr_vce_entry.get(),
                vge_voltage_V=float(self.var_tsep_vge.get()),
                vge_compliance_mA=float(self.var_tsep_vge_comp.get()),
                vce_source_mA=float(self.var_tsep_ic.get()),
                vce_compliance_V=float(self.var_tsep_vcomp.get()),
                equation_type=self.var_tsep_eq.get(),
                a=float(self.var_tsep_a.get()),
                b=float(self.var_tsep_b.get()),
                c=float(self.var_tsep_c.get()),
            )
            p.validate()
        except Exception as e:
            self._show_error("Invalid TSEP Parameters", e); return

        def on_status(msg: str) -> None: self._post(lambda: self._set_tsep_status(msg))
        def on_progress(pct: float) -> None: self._post(lambda: self._set_tsep_progress(pct))

        def work():
            try:
                ctrl = TSEPController()
                res = ctrl.run(p, on_status, on_progress)
                def update_ui():
                    self.var_tsep_v_read.set(f"{res.mean_voltage_V:.6f}")
                    self.var_tsep_tj.set(f"{res.tj_celsius:.3f}")
                    self._set_tsep_status("Temperature measurement complete")
                self._post(update_ui)
                self._last_tsep_result = res
            except Exception as e:
                self._post(lambda: (self._set_tsep_status("TSEP error"), self._show_error("TSEP Error", e)))
        Thread(target=work, daemon=True).start()

    def clear_tsep_outputs(self) -> None:
        try:
            self.var_tsep_v_read.set("—")
            self.var_tsep_tj.set("—")
            self._set_tsep_status("Outputs cleared")
        except Exception as e:
            self._show_error("Clear Outputs Error", e)

    # Copy helpers
    def _copy_to_clipboard(self, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_tsep_status("Copied to clipboard")
        except Exception as e:
            self._show_error("Clipboard Error", e)

    def copy_tsep_voltage(self) -> None:
        val = self.var_tsep_v_read.get()
        if val and val != "—":
            self._copy_to_clipboard(val)
        else:
            self._show_error("Copy Voltage Error", Exception("No voltage value to copy"))

    def copy_tsep_temperature(self) -> None:
        val = self.var_tsep_tj.get()
        if val and val != "—":
            self._copy_to_clipboard(val)
        else:
            self._show_error("Copy Temperature Error", Exception("No temperature value to copy"))

    # --- Heating period ---
    def clear_heating_plot(self) -> None:
        if self.heating_running:
            return
        self._heat_data = []
        self._heat_times = []
        self._heat_tj_values = []
        self._style_heat_axes()
        self._draw_setpoint_line()
        self._set_tsep_status("Heating plot cleared")

    def _update_y_limits_from_data(self) -> None:
        """Set Y-axis limits based on current Tj values, ignoring the setpoint line."""
        if not self._heat_tj_values:
            return
        ymin = min(self._heat_tj_values)
        ymax = max(self._heat_tj_values)
        # Ensure non-zero span and add padding
        span = max(1e-6, ymax - ymin)
        pad = max(0.5, 0.08 * span)
        self.ax_heat.set_ylim(ymin - pad, ymax + pad)

    def _heating_measure_loop(self, params: TSEPParams, duration_s: int, on_status, on_progress):
        ctrl = TSEPController()
        try:
            on_status("Connecting TSEP instruments...")
            ctrl.smu_vge = Keithley2400(params.vge_gpib)
            ctrl.smu_vce = Keithley2400(params.vce_gpib)
            try:
                idn_vge = getattr(ctrl.smu_vge, 'idn', None) or ctrl.smu_vge.ask('*IDN?')
                on_status(f"VGE connected: {idn_vge}")
            except Exception:
                pass
            try:
                idn_vce = getattr(ctrl.smu_vce, 'idn', None) or ctrl.smu_vce.ask('*IDN?')
                on_status(f"VCE connected: {idn_vce}")
            except Exception:
                pass
            ctrl._configure_vge(params)
            ctrl._const.vce_source_current_A = params.vce_source_mA / 1000.0
            ctrl._const.vce_compliance_voltage_V = params.vce_compliance_V
            ctrl._configure_vce(params)
            sleep(ctrl._const.settle_s)
            on_status("Enabling VGE...")
            ctrl.smu_vge.enable_source()
            on_status("Enabling VCE...")
            ctrl.smu_vce.enable_source()
            on_status("Measuring heating period...")

            self.heat_start_time = t0 = time.time()
            self._heat_data = []
            # Pre-build time axis and fix X limits
            self._heat_time_axis = np.arange(0, duration_s + 1, 1, dtype=float)
            try:
                self.ax_heat.set_xlim(0, float(duration_s))
            except Exception:
                pass
            self._post(lambda: (self.ax_heat.relim(), self.ax_heat.autoscale_view(), self.canvas_heat.draw()))

            for i in range(duration_s):
                if self._heat_stop.is_set():
                    on_status("Heating measurement stopped by user")
                    break
                try:
                    ctrl.smu_vce.source_current = ctrl._const.vce_source_current_A
                except Exception:
                    pass
                try:
                    V = float(getattr(ctrl.smu_vce, 'voltage'))
                except Exception:
                    V = float(ctrl.smu_vce.voltage)
                if params.equation_type == 'linear':
                    tj = params.a + params.b * V
                else:
                    tj = params.a + params.b * V + params.c * (V ** 2)
                t_s = time.time() - t0
                self._heat_data.append((t_s, tj))

                def _upd():
                    self._heat_times.append(t_s)
                    self._heat_tj_values.append(tj)
                    offsets = np.column_stack((self._heat_times, self._heat_tj_values))
                    try:
                        self.heat_scatter.set_offsets(offsets)
                    except Exception:
                        self.heat_scatter = self.ax_heat.scatter(self._heat_times, self._heat_tj_values, s=12, c='red', marker='o', label='Tj')
                        self.ax_heat.legend(loc='upper right')
                    # Update Y limits from data (ignore setpoint line)
                    self._update_y_limits_from_data()
                    self.canvas_heat.draw()  # force draw for reliability
                    self.var_tsep_tj.set(f"{tj:.3f}")
                self._post(_upd)
                on_progress(min(100.0, (t_s / duration_s) * 100.0))
                target = t0 + (i + 1)
                while not self._heat_stop.is_set() and time.time() < target:
                    sleep(0.05)
            on_progress(100.0)
        finally:
            try:
                ctrl.smu_vce.disable_source()
            except Exception:
                pass
            try:
                ctrl.smu_vge.disable_source()
            except Exception:
                pass
            try:
                ctrl.smu_vce.write('*CLS'); ctrl.smu_vce.write('*SRE 0')
            except Exception:
                pass
            try:
                ctrl.smu_vge.write('*CLS'); ctrl.smu_vge.write('*SRE 0')
            except Exception:
                pass

    def start_heating(self) -> None:
        if self.heating_running:
            return
        try:
            duration_min = int(float(self.var_heat_minutes.get()))
            duration_s = max(1, duration_min * 60)
            p = TSEPParams(
                vge_gpib=self.keithley_addr.get(),
                vce_gpib=self.tsep_addr_vce_entry.get(),
                vge_voltage_V=float(self.var_tsep_vge.get()),
                vge_compliance_mA=float(self.var_tsep_vge_comp.get()),
                vce_source_mA=float(self.var_tsep_ic.get()),
                vce_compliance_V=float(self.var_tsep_vcomp.get()),
                equation_type=self.var_tsep_eq.get(),
                a=float(self.var_tsep_a.get()),
                b=float(self.var_tsep_b.get()),
                c=float(self.var_tsep_c.get()),
            )
            p.validate()
        except Exception as e:
            self._show_error("Invalid Heating Parameters", e); return

        self.heating_running = True
        self._heat_stop.clear()
        self._set_tsep_status("Preparing heating period...")
        self.btn_heat_start.config(state='disabled')
        self.btn_heat_stop.config(state='normal')
        self.btn_heat_export.config(state='disabled')
        # Pre-build x axis and set fixed x-limits; clear arrays & scatter
        self._heat_time_axis = np.arange(0, duration_s + 1, 1, dtype=float)
        try:
            self.ax_heat.set_xlim(0, float(duration_s))
        except Exception:
            pass
        self._heat_times = []
        self._heat_tj_values = []
        try:
            self.heat_scatter.set_offsets(np.empty((0, 2)))
        except Exception:
            pass
        self._draw_setpoint_line()

        def on_status(msg: str) -> None: self._post(lambda: self._set_tsep_status(msg))
        def on_progress(pct: float) -> None: self._post(lambda: self._set_tsep_progress(pct))

        def work():
            try:
                self._heating_measure_loop(p, duration_s, on_status, on_progress)
                self._post(lambda: self._set_tsep_status("Heating measurement complete"))
            except Exception as e:
                self._post(lambda: (self._set_tsep_status("Heating error"), self._show_error("Heating Error", e)))
            finally:
                self._post(lambda: (
                    setattr(self, 'heating_running', False),
                    self.btn_heat_start.config(state='normal'),
                    self.btn_heat_stop.config(state='disabled'),
                    self.btn_heat_export.config(state='normal' if len(self._heat_data)>0 else 'disabled')
                ))
        self.heating_thread = Thread(target=work, daemon=True); self.heating_thread.start()

    def stop_heating(self) -> None:
        if not self.heating_running:
            return
        self._heat_stop.set()
        self._set_tsep_status("Stopping heating measurement...")

    def export_heating_csv(self) -> None:
        if not self._heat_data:
            self._show_error("Export Error", Exception("No data to export")); return
        try:
            df = pd.DataFrame(self._heat_data, columns=['Time (s)', 'Tj (°C)'])
            try:
                setpoint = float(self.var_setpoint_c.get())
            except Exception:
                setpoint = None
            if setpoint is not None:
                df['Setpoint (°C)'] = setpoint
            sp_str = f"{int(setpoint)}C" if setpoint is not None else "SP"
            default_name = f"heating_Tj_{sp_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path = filedialog.asksaveasfilename(title="Export heating period to CSV", defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv")], initialfile=default_name)
            if path:
                df.to_csv(path, index=False)
                self._set_tsep_status(f"Heating data exported to {path}")
        except Exception as e:
            self._show_error("CSV Export Error", e)

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementGUI(root)
    root.mainloop()
