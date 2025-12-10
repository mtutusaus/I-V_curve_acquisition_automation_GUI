import pyvisa
from pyvisa.constants import (
    VI_EVENT_SERVICE_REQ,
    EventMechanism,
    VI_ALL_ENABLED_EVENTS,
)
from . import commands as cmd
import time
import threading
import csv


import logging
logger = logging.getLogger(__name__)

class Tek371:
    """Driver class for Tektronix 371 Curve Tracer using PyVISA."""

    def __init__(self, resource: str, timeout_ms: int = 5000):
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = timeout_ms
        self.inst.write_termination = '\n'
        self.inst.read_termination = '\n'

        # --- SRQ state ---
        self._srq_called = False
        self._srq_lock = threading.Lock()
        self._srq_handler_installed = False

    # -----------------------
    # Low-level I/O
    # -----------------------
    def write(self, command: str) -> None:
        self.inst.write(command)

    def query(self, command: str) -> str:
        return self.inst.query(command)

    def read_raw(self) -> bytes:
        return self.inst.read_raw()

    def close(self) -> None:
        try:
            # Remove SRQ handler if installed
            if self._srq_handler_installed:
                try:
                    self.inst.disable_event(VI_EVENT_SERVICE_REQ, EventMechanism.all)
                except Exception:
                    pass
            self.inst.close()
        finally:
            self.rm.close()

    # -----------------------------
    # Connection & Utilities
    # -----------------------------

    def initialize(self) -> None:
        """
        Initializes the 371 to its power-up settings.

        Initializes the instrument as:
            DIS STORE, CUR OFF, DIS CAL:OFF, DIS INV:OFF, STP CUR:1.0E-3,
            STP OFF:0.00, STP INV:OFF, PKP 300, CSP NPN, HOR COL:1.0E+0,
            OPC OFF, MEA REPEAT, STP NUM:2, STP MUL:OFF, VCS 0.0,
            VER COL:1.0E+0, RQS ON, DEB OFF.
        """
        self.write(cmd.INI_SET)

    def id_string(self) -> str:
        """
        Queries the 371 for its firmware version

        Returns:
            str: A response string in the format:
                "ID SONY_TEK/371,V81.1F <version>"
            where <version> is the current firmware version.
        """
        return self.query(cmd.ID_QUERY)

    def get_help(self) -> str:
        """
        Ask the 371 for a list of all valid command and query headers.

        Returns:
            str: A comma-separated list of valid headers in the format:
                "HELP READOUT, TEXT, LINE, DOT, WINDOW, CURSOR, DISPLAY, HORIZ, VERT,
                STEPGEN, MEASURE, ENTER, RECALL, SAVE, PLOT, PSTATUS, PKPOWER, CSPOL,
                CSOUT, VCSPPLY, OUTPUTS, WFMPRE, CURVE, WAVFRM, RQS, OPC, EVENT, TEST,
                INIT, ID, DEBUG, SET"
        """
        return self.query(cmd.HEL_QUERY)

    def get_settings(self) -> str:
        """
        Queries the 371 for its current front-panel settings

        Returns:
            str: A semicolon-separated string containing the instrument settings.
            Example format:
                OPC <mode>;
                RQS <mode>;
                PKPOWER <watts>;
                CSPOL <polarity>;
                HORIZ <source:size>;
                VERT COLLECT:<size>;
                STEPGEN OUT:<mode>,NUMBER:<number>,OFFSET:<offset>,INVERT:<mode>,MULT:<mode>,<source:size>;
                VCSPPLY <percent>;
                MEASURE <mode>;
                DISPLAY INVERT:<mode>,CAL:<mode>,<display mode>;
                CURSOR <mode>

            Notes:
                - If measurement mode is SWE (sweep) or SSW (slow sweep), the MEASURE field will be "SWEEP".
        """
        return self.query(cmd.SET_QUERY)

    def perform_test(self) -> str:
        """
        Perform diagnostic tests on the ROM and RAM.

        Returns:
            str: A response string in the format:
                "TEST ROM:<code>,RAM:<code>"
            where each <code> is an error code (0000 means no error).
            Refer to the Service Manual for details on codes other than 0000.
        """

        return self.query(cmd.TES_QUERY)

    # -----------------------------
    # Collector Supply
    # -----------------------------

    def set_collector_polarity(self, mode: str) -> None:
        """
        Set the collector supply polarity and mode.

        Args:
            mode (str): Polarity mode. Accepted values:
                - "NPN" or "POS" (positive)
                - "PNP" or "NEG" (negative)
        """
        self.write(cmd.CSP_SET.format(mode=mode))

    def get_collector_polarity(self) -> str:
        """
        Queries the 371 for the current setting of the Collector Supply polarity

        Returns:
            str: A response string in the format "CSPOL <mode>", where <mode> is:
                - "NPN" for positive polarity
                - "PNP" for negative polarity
        """
        return self.query(cmd.CSP_QUERY)

    def set_peak_power(self, watts: int) -> None:
        """
        Selects the Collector Supply Peak Power setting.

        Args:
            watts (int): Peak power in watts. Accepted values:
                - 3000, 300, 30, 3
        """
        self.write(cmd.PKP_SET.format(set=watts))

    def get_peak_power(self) -> str:
        """
        Queries the 371 for the current setting of the Collector Supply Peak Power.

        Returns:
            str: A response string in the format "PKPOWER <set>" where <set> is:
                - 3000, 300, 30 or 3, in watts
        """
        return self.query(cmd.PKP_QUERY)

    def set_collector_supply(self, percent: float) -> None:
        """
        Sets the Collector Supply output level.

        Args:
            percent (float): Collector supply level as a percentage, from 0.0 to 100.0
                in increments of 0.1%.
        Notes:
            - 100.0% is approximately 30V
        """
        self.write(cmd.VCS_SET.format(data=f"{percent:.1f}"))

    def get_collector_supply(self) -> str:
        """
        Queries the 371 for the current setting of the Collector Supply output level

        Returns:
            str: A response string in the format "VCSPPLY <data>" where <data> is:
                - Collector supply level as a percentage, from 0.0 to 100.0
                in increments of 0.1%.
        Notes:
            - 100.0% is approximately 30V
        """
        return self.query(cmd.VCS_QUERY)

    def get_breaker_status(self) -> str:
        """
        Queries the 371 for the current setting of the Collector supply HIGH VOLTAGE and HIGH CURRENT breakers.

        Returns:
            str: A response string in the format "CSO <mode>" where <mode> is:
                - BOTH: Both the HIGH VOLTAGE and HIGH CURRENT breakers are enabled
                - VOLTAGE: HIGH VOLTAGE breaker is enabled; HIGH CURRENT breaker is disabled.
                - CURRENT: HIGH CURRENT breaker is enabled; HIGH VOLTAGE breaker is disabled.
                - OFF: Both the HIGH VOLTAGE and HIGH CURRENT breakers are disabled.
        Notes:
            - The state of the breakers can only be queried, to set it, it must be done manually on the equipment.
        """
        return self.query(cmd.CSO_QUERY)

    # -----------------------------
    # Display
    # -----------------------------

    def set_display_mode(self, mode: str) -> None:
        """
        Set the display mode.

        Args:
            mode (str): Display mode. Accepted values:
                - "NST" (non-store mode)
                - "STO" (store mode)
        """
        self.write(cmd.DIS_MODE_SET.format(mode=mode))

    def view_curve(self, index: int) -> None:
        """
        Set view mode and display a curve from the specified location.

        Args:
            index (int): Bubble memory index. Accepted values:
                - 1 to 16
        """
        self.write(cmd.DIS_VIEW_SET.format(index=index))

    def compare_curve(self, index: int) -> None:
        """
        Set compare mode and display curve from the specified location

        Args:
            index (int): Bubble memory index. Accepted values:
                - 1 to 16
        """
        self.write(cmd.DIS_COMP_SET.format(index=index))

    def invert_display(self, status: str) -> None:
        """
        Set the display to invert mode

        Args:
            status (str): Display invert mode. Accepted values:
                - "ON"
                - "OFF" (default)
        """
        self.write(cmd.DIS_INV_SET.format(status=status))

    def set_calibration(self, status: str) -> None:
        """
        Set the CRT to calibration mode

        Args:
            status (str): CRT calibration mode. Accepted values:
                - "ZER"
                - "OFF"
                - "FUL"
        """
        self.write(cmd.DIS_CAL_SET.format(status=status))

    def get_display_settings(self) -> str:
        """
        Queries the 371 for the current settings of the display, polarity, and calibration mode.

        Returns:
            str: A response string in the format: "DISPLAY <mode1>,<mode2>,<mode3>" where:
                - mode1: NSTORE, STORE, VIEW:<index> (bubble memory index 1–16), or COMPARE:<index> (bubble memory index 1–16)
                - mode2: INVERT:OFF or INVERT:ON
                - mode3: CAL:ZERO, CAL:OFF, or CAL:FULL
        """
        return self.query(cmd.DIS_QUERY)

    def store_display(self, index: int) -> None:
        """
        Store the display in mass storage. Only valid in store or view mode

        Args:
            index (int): Bubble memory index. Accepted values:
                - 1 to 16
        """
        self.write(cmd.ENT_SET.format(index=index))

    # -----------------------------
    # Horizontal & Vertical
    # -----------------------------

    def set_horizontal(self, source: str, volt_div: float) -> None:
        """
        TODO: Check values with tek371, i think that it goes in a sequence of 1-2-5...
        Set the 371 horizontal display source and sensitivity.

        Args:
            source (str): Horizontal source. Accepted values:
                - "COL"
                - "STP"
            volt_div (float): Voltage per division. Accepted ranges:
                - For COL: 500.0E-3 to 5.0 (if peak watts is 3KW/300W)
                            50.0 to 500.0 (if peak watts is 30W/3W)
                - For STP: 100.0E-3 to 5.0
        """
        self.write(cmd.HOR_SET.format(source=source, volt=f"{volt_div:.2E}"))

    def get_horizontal(self) -> str:
        """
        Queries the 371 for the current horizontal source and sensitivity settings.

        Returns:
            str: A response string in the format: "HORIZ <source>:<volt>" where:
                - <source> is COLLECT or STPGEN
                - <volt> is sensitivity (volt/div) in scientific notation
        """
        return self.query(cmd.HOR_QUERY)

    def set_vertical(self, amp_div: float) -> None:
        """
        TODO: Check values with tek371, i think that it goes in a sequence of 1-2-5...
        Set the vertical sensitivity of the 371.

        Args:
            amp_div (float): Vertical sensitivity (amp/div). Accepted ranges:
                - 1.0 to 50.0 when peak watts is 3 kW
                - 500.0E-3 to 5.0 when peak watts is 300 W
                - 100.0E-6 to 5.0E-3 when peak watts is 30 W
                - 10.0E-6 to 500.0E-6 when peak watts is 3 W
        """
        self.write(cmd.VER_SET.format(amp=f"{amp_div:.2E}"))

    def get_vertical(self) -> str:
        """
        Queries the 371 for the vertical sensitivity settings

        Returns:
            str: A response string in the format: "VER COL:<amp>" where <amp> is:
                - sensitivity in A/div
        """
        return self.query(cmd.VER_QUERY)

    # -----------------------------
    # Step Generator
    # -----------------------------

    def enable_step_output(self, mode: str) -> None:
        """
        Enables the step generator source
        Args:
            mode (str): Step generator enabled. Accepted values:
                - "ON"
                - "OFF"
        """
        self.write(cmd.STP_OUT_SET.format(mode=mode))

    def set_step_current(self, val: float) -> None:
        """
        TODO: Not sure about this, i think it is a fixed sequence as well...
        Set the step generator to provide current steps and the step size in amperes.

        Args:
            val (float): Current step size (amp/div). Accepted ranges:
                - 1.0E-6 to 2.0E-3 when peak power is 30 W / 3 W
                - 1.0E-3 to 2.0 when peak power is 3 kW / 300 W
        """
        self.write(cmd.STP_CUR_SET.format(val=f"{val:.2E}"))

    def set_step_voltage(self, val: float) -> None:
        """
        Set the step generator to provide voltage steps and the step size in volts.

        Args:
            val (float): Voltage step size (volt/step). Accepted range:
                - 200.0E-3 through 5.0 in a 1-2-5 sequence
        """
        self.write(cmd.STP_VOL_SET.format(val=f"{val:.2E}"))

    def set_step_number(self, val: int) -> None:
        """
        Set number of steps to be generated
        Args:
            val (int): Number of steps. Accepted range:
                - 0 to 5
        """
        self.write(cmd.STP_NUM_SET.format(val=val))

    def invert_step(self, mode: str) -> None:
        """
        Set the step generator polarity
        Args:
            mode (str): Step generator polarity. Accepted values:
                - "ON"
                - "OFF"
        """
        self.write(cmd.STP_INV_SET.format(mode=mode))

    def set_step_multiplier(self, mode: str) -> None:
        """
        Set the step generator step multi 0.1X mode
        Args:
            mode (str): Step generator multi 0.1X mode. Accepted values:
                - "ON"
                - "OFF"
        """
        self.write(cmd.STP_MUL_SET.format(mode=mode))

    def enable_step_offset(self, mode: str) -> None:
        """
        Enable or disable the step generator offset.

        Args:
            mode (str): Offset mode. Accepted values:
                - "ON"
                - "OFF"
        """
        self.write(cmd.STP_OFF_MODE_SET.format(mode=mode))

    def set_step_offset(self, val: float) -> None:
        """
        Set the step generator offset value.

        Args:
            val (float): Offset value. Accepted ranges:
                - 0 to 500 when step multiplication is ON
                - 0 to 5 times the step/offset setting otherwise
        """
        self.write(cmd.STP_OFF_SET.format(val=f"{val:.2f}"))

    def get_step_settings(self) -> str:
        """
        Query the 371 instrument for the current step generator settings.

        Includes:
            - Source type: amps/step or volts/step
            - Number of steps
            - Offset
            - Polarity
            - Multiplier mode
            - Output mode

        Returns:
            str: Response in the format:
                "STPGEN NUMBER:<num>, OFFSET:<offset>, INVERT:<invert>, MULT:<mult>,
                <typ:size>, <mult>, <typ:size>, <output>"

                Where:
                    - num: Number of steps (0–5)
                    - offset: Offset value multiplier
                    - invert: Invert mode status ("ON" or "OFF")
                    - mult: Step multiplier status ("ON" or "OFF")
                    - typ:size: CURRENT:size (A/step) or VOLTAGE:size (V/step)
                    - output: Output status ("ON" or "OFF")
        """
        return self.query(cmd.STP_QUERY)

    # -----------------------------
    # Measurement
    # -----------------------------

    def set_measurement_mode(self, mode: str) -> None:
        """
        Select the measurement mode.

        Args:
            mode (str): Measurement mode command in the format: "MEA <mode>", where <mode> is:
                - REP: Repeat
                - SIN: Single
                - SWE: Sweep
                - SSW: Slow sweep
        """
        self.write(cmd.MEA_SET.format(mode=mode))

    def get_measurement_mode(self) -> str:
        """
        Queries the 371B for the current measurement mode setting

        Returns:
            str: Response in the format: "MEASURE <mode>", where <mode> is:
                - "REPEAT"
                - "SINGLE"
                - "SWEEP"
                - "SSWEEP"
        """
        return self.query(cmd.MEA_QUERY)

    # -----------------------------
    # Cursor & CRT
    # -----------------------------

    def cursor_off(self) -> None:
        """
        Sets the 371 cursor mode to off
        """
        self.write(cmd.CURS_SET)

    def set_dot_cursor(self, pos: int) -> None:
        """
        Sets the 371 dot cursor position to a specific point on the currently displayed curve.

        Args:
            pos (int): Curve point. Accepted range:
                - 0, 1, 2, ... 1024 (0 is the beginning of the curve and 1024 is the end).

        Notes:
            When measuring only one curve (without Step Generator enabled), the number of points ranges from 0 to 255.
        """
        self.write(cmd.DOT_SET.format(data=pos))

    def get_dot_cursor(self) -> str:
        """
        Queries the 371 for the location of the dot cursor on the currently displayed curve

        Returns:
            - str: Response in the format: DOT <NR1> where <NR1> is:
                - number between 0 and 1024 specifying the location of the dot cursor on the currently displayed curve
                (0 is the beginning of the curve and 1024 is the end)

        Notes:
            When measuring only one curve (without Step Generator enabled), the number of points ranges from 0 to 255.
        """
        return self.query(cmd.DOT_QUERY)

    def set_line_cursor(self, hor: int, ver: int) -> None:
        self.write(cmd.LIN_SET.format(hor=hor, ver=ver))

    def get_line_cursor(self) -> str:
        return self.query(cmd.LIN_QUERY)

    def set_window_cursor(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.write(cmd.WIN_SET.format(bott_l_hor=x1, bott_l_ver=y1, up_r_hor=x2, up_r_ver=y2))

    def get_window_cursor(self) -> str:
        return self.query(cmd.WIN_QUERY)

    def get_cursor_readout(self, fmt: str = "") -> str:
        return self.query(cmd.REA_QUERY.format(number_format=fmt).strip())

    def get_beta_gm(self) -> str:
        return self.query(cmd.BGM_QUERY)

    # -----------------------------
    # Text
    # -----------------------------

    def write_text(self, text: str) -> None:
        self.write(cmd.TEX_SET.format(string=f'"{text}"'))

    def read_text(self) -> str:
        return self.query(cmd.TEX_QUERY)

    # -----------------------------
    # Waveform & Curve
    # -----------------------------

    def read_curve(self, filename: str) -> None:
        """
        Reads the curve from the instrument, parses and scales points,
        and saves them to a CSV file.

        Args:
            filename (str): Path to the output CSV file.
        """

        # Read preamble and split into array
        preamble_array = self.read_preamble().split(",")

        # Extract values
        nr_pt = int(preamble_array[2].split(":")[1].strip())  # NR.PT (number of points, set by the 371)
        xmult = float(preamble_array[4].split(":")[1].strip())  # XMULT (x-axis gain, depends on horizontal sensitivity)
        ymult = float(preamble_array[8].split(":")[1].strip())  # YMULT (y-axis gain, depends on vertical sensitivity)
        xoff = int(preamble_array[6].split(":")[1].strip())  # XOFF (x-axis offset)
        yoff = int(preamble_array[10].split(":")[1].strip())  # YOFF (y-axis offset)

        # Calculate expected bytes (just used to check all the waveform was received correctly)

        # ---- From the TEK371 manual ----
        curve_head_len = 25
        bytes_for_data_len = 2
        bytes_for_checksum = 1
        bytes_per_x = 2
        bytes_per_y = 2
        # ---- From the TEK371 manual ----

        points_to_read = nr_pt * (bytes_per_x + bytes_per_y)
        expected_bytes = curve_head_len + bytes_for_data_len + points_to_read + bytes_for_checksum

        # Request curve and read exact number of bytes

        self.write(cmd.CUR_QUERY)

        raw_curve = self.inst.read_bytes(expected_bytes)

        # Parse and scale points
        start_idx = curve_head_len + bytes_for_data_len
        end_idx = len(raw_curve) - bytes_for_checksum
        points_bytes = raw_curve[start_idx:end_idx] # Just extract waveform data, without preamble nor checksum

        points = []
        for i in range(0, len(points_bytes), 4): # Data is arranged in pairs of 4 bytes, 2 for X and 2 for Y
            x_bytes = points_bytes[i:i + 2]
            y_bytes = points_bytes[i + 2:i + 4]
            if len(x_bytes) < 2 or len(y_bytes) < 2:
                break
            coord_x = int.from_bytes(x_bytes, byteorder="big", signed=False) - xoff # Adjust X-axis offset
            coord_y = int.from_bytes(y_bytes, byteorder="big", signed=False) - yoff # Adjust Y-axis offset
            voltage = coord_x * xmult # Apply X-axis gain
            current = coord_y * ymult # Apply Y-axis gain
            points.append((voltage, current))
            # Due to how the 371 measures, it returns first the higher current values. To simplify post-processing
            # we simply reorder it to ascending current
            points.sort(key=lambda p: p[1])

        # Check if we received and parsed correctly the measured waveform
        if len(points) != nr_pt:
            logger.error("Parsed points mismatch: got %s, expected %s", len(points), nr_pt)
            raise ValueError(f"Number of points parsed ({len(points)}) does not match expected ({nr_pt}).")

        # Save scaled points to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Voltage (V)", "Current (A)"])
            writer.writerows(points)

        logger.info(f"Curve saved to {filename}")

    def read_preamble(self) -> str:
        return self.query(cmd.WFM_QUERY)

    def read_waveform(self) -> str:
        return self.query(cmd.WAV_QUERY)

    def set_waveform_length(self, points: int) -> None:
        self.write(cmd.WFM_LENGTH_SET.format(points=points))

    def get_waveform_length(self) -> str:
        return self.query(cmd.WFM_LENGTH_QUERY)

    # -----------------------------
    # Miscellaneous
    # -----------------------------

    def save_settings(self, index: int) -> None:
        self.write(cmd.SAV_SET.format(index=index))

    def recall_settings(self, index: int) -> None:
        self.write(cmd.REC_SET.format(index=index))

    def get_output_status(self) -> str:
        return self.query(cmd.OUT_QUERY)

    def print_curve(self, mode: str) -> None:
        self.write(cmd.PLO_SET.format(mode=mode))

    def get_printer_status(self) -> str:
        return self.query(cmd.PST_QUERY)

    def debug_mode(self, status: str) -> None:
        self.write(cmd.DEB_SET.format(status=status))

    def get_debug_status(self) -> str:
        return self.query(cmd.DEB_QUERY)

    def get_event_code(self) -> str:
        return self.query(cmd.EVE_QUERY)

    def set_opc(self, status: str) -> None:
        self.write(cmd.OPC_SET.format(status=status))

    def get_opc_status(self) -> str:
        return self.query(cmd.OPC_QUERY)

    def set_rqs(self, status: str) -> None:
        self.write(cmd.RQS_SET.format(status=status))

    def get_rqs_status(self) -> str:
        return self.query(cmd.RQS_QUERY)

    # -----------------------------
    # SRQ handling
    #  -----------------------------
    def discard_and_disable_all_events(self) -> None:
        """
        Clear any pending VISA events and disable delivery.
        Useful before enabling SRQ to start clean.
        """
        try:
            self.inst.discard_events(VI_ALL_ENABLED_EVENTS, EventMechanism.all)
            self.inst.disable_event(VI_ALL_ENABLED_EVENTS, EventMechanism.all)
        except Exception:
            # Different VISA backends may behave slightly differently;
            # swallow harmless errors.
            pass

    def enable_srq_event(self) -> None:
        """
        Enable SRQ at instrument level (OPC & RQS) and at controller level (VISA events).
        Installs a handler that flips a flag when SRQ arrives.
        """
        # 1) Ask instrument to assert SRQ when operations complete
        #    (these map to your commands.py)
        self.write(cmd.OPC_SET.format(status="ON"))   # Operation Complete ON
        self.write(cmd.RQS_SET.format(status="ON"))   # Service Request ON

        # 2) Reset state and clean previous event configuration
        with self._srq_lock:
            self._srq_called = False
        self.discard_and_disable_all_events()

        # 3) Install handler and enable SRQ delivery in VISA
        def _srq_handler(resource, event, user_handle):
            # Flip the flag, and optionally you can query EVENT code here if you need to distinguish causes.
            with self._srq_lock:
                self._srq_called = True

        wrapped = self.inst.wrap_handler(_srq_handler)
        self.inst.install_handler(VI_EVENT_SERVICE_REQ, wrapped, None)
        self.inst.enable_event(VI_EVENT_SERVICE_REQ, EventMechanism.handler, None)
        self._srq_handler_installed = True

    def wait_for_srq(self, poll_interval: float = 0.1, timeout_s: float = 30.0) -> bool:
        """
        Block until SRQ arrives (sweep complete), or timeout.

        Returns:
            True if SRQ received before timeout, False otherwise.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._srq_lock:
                if self._srq_called:
                    # reset for next operation
                    self._srq_called = False
                    return True
            time.sleep(poll_interval)
        return False

    def disable_srq_event(self) -> None:
        """Disable SRQ generation in the instrument and stop VISA SRQ delivery."""
        # instrument side
        try:
            self.write(cmd.RQS_SET.format(status="OFF"))
        except Exception:
            pass
        try:
            self.write(cmd.OPC_SET.format(status="OFF"))
        except Exception:
            pass

        # controller side
        try:
            self.inst.disable_event(VI_EVENT_SERVICE_REQ, EventMechanism.all)
        except Exception:
            pass
        self._srq_handler_installed = False
        with self._srq_lock:
            self._srq_called = False
