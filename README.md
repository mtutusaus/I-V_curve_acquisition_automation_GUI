# I-V curve acquisition automation GUI

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyVISA](https://img.shields.io/badge/instrument-PyVISA-blue.svg)
![PyMeasure](https://img.shields.io/badge/instrument-PyMeasure-green.svg)
![AI-Assisted](https://img.shields.io/badge/Development-AI--Assisted-purple)
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Graphical interface for automated I-V curve acquisition of MOS gated power devices using a **Tektronix TEK371** curve tracer and a **Keitlhey2400 SMU**.

This GUI provides a user-friendly interface for the [curve-tracer](https://github.com/mtutusaus/curve-tracer) automation framework, enabling easy transistor characterization.

---
## **Usage**

1. Run the GUI application:
```bash
   python main_gui.py
```

2. In the GUI:
   - Connect to the instruments via GPIB by quering their IDs
   - Configure measurement parameters
   - Set the output directory for data files
   - Click "Start Measurement" to begin acquisition

3. The application will:
   - Acquire the specified number of I-V curves
   - Save individual measurements separately
   - Compute and save the mean curve in a dedicated subfolder named `mean`
   - Display real-time results
     
---
## **Current Implementation Status**

**Currently Implemented:**
- I-V curve measurement and acquisition
- Real-time data visualization
- Automatic data organization and export

**Planned Features:**
- Dual gate voltage source selection (Tek371 / Keithley 2400)
- Junction temperature monitoring:
    - Hot plate heating period monitoring
    - Instantaneous temperature measurement 
- Single I-V point measurements

These features are available in the command-line [curve-tracer](https://github.com/mtutusaus/curve-tracer) version and will be integrated into the GUI in future releases.

---
## **Related Projects**

This GUI is built on top of:
- [curve-tracer](https://github.com/mtutusaus/curve-tracer) - Core automation scripts and measurement logic
- [tek371-driver](https://github.com/mtutusaus/tek371-driver) - Low-level Tektronix 371 communication driver

---
## **Development**

This GUI is built entirely through AI-assisted development (M365 Copilot). The underlying curve tracer automation comes from my [curve-tracer](https://github.com/mtutusaus/curve-tracer) repository.

---
## **License**
This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — Derivatives must use the same license

See the [LICENSE](LICENSE) file for the full license text.

## Author

[Miquel Tutusaus](https://github.com/mtutusaus), 2025
