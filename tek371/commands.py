"""
commands.py
Stores TEK371 GPIB command strings grouped by functionality.
"""

""" Collector Supply Commands and Queries
The Collector Supply group of commands and queries sets and reports the status of the collector supply polarity, mode, 
peak power, and output level. They also query the 317 on the status of the collector supply HIGH CURRENT and HIGH
VOLTAGE breakers, which cannot be set via the GPIB.
"""
CSO_QUERY = "CSO?"
# Purpose:  Queries the 371 for the current setting of the Collector supply HIGH VOLTAGE and HIGH CURRENT breakers
# Response: CSO <mode> where mode = BOTH | VOLTAGE | CURRENT | OFF
# BOTH:     Both the HIGH VOLTAGE and HIGH CURRENT breakers are enabled
# VOLTAGE:  HIGH VOLTAGE breaker is enabled; HIGH CURRENT breaker is disabled.
# CURRENT:  HIGH CURRENT breaker is enabled; HIGH VOLTAGE breaker is disabled.
# OFF:      Both the HIGH VOLTAGE and HIGH CURRENT breakers are disabled.

CSP_SET = "CSP {mode}"
# Purpose:  Sets the collector supply polarity and mode
# Syntax:   CSP <mode> where mode = NPN | POS | PNP | NEG
# NPN:      Sets polarity for NPN devices
# POS:      Positive polarity
# PNP:      Sets polarity for PNP devices
# NEG:      Negative polarity

CSP_QUERY = "CSP?"
# Purpose:  Queries the 371 for the current setting of the Collector Supply polarity
# Response: CSPOL <mode> where mode = NPN | PNP
# NPN:      Polarity is set to NPN
# PNP:      Polarity is set to PNP

PKP_SET = "PKP {set}"
# Purpose:  Selects the Collector Supply Peak Power setting
# Syntax:   PKP <set> where set = 3000 or 300 or 30 or 3, in watts

PKP_QUERY = "PKP?"
# Purpose:  Queries the 371 for the current setting of the Collector Supply Peak Power
# Response: PKPOWER <set> where set = 3000 or 300 or 30 or 3, in watts

VCS_SET = "VCS {data}"
# Purpose:  Sets the Collector Supply output level
# Syntax:   VCS <data> where data = 0.0 or 0.1 ... 99.9 or 100.0 in percentage in increments of 0.1%

VCS_QUERY = "VCS?"
# Purpose:  Queries the 371 for the current setting of the Collector Supply output level
# Response: VCSPPLY <data> where data = 0.0 or 0.1 ... 99.9 or 100.0 in percentage in increments of 0.1%

""" CRT Readout Transfer Commands and Queries
The CRT Readout Transfer group allows the controller to read horizontal and vertical cursor parameters from the 371, and to read or write text in the text area of the display graticule
"""
BGM_QUERY = "BGM?"
# Purpose:  Queries the 371 for the Beta or gm/DIV readout
# Response: BGM <para> where para is NR1 (integer)

REA_QUERY = "REA? {number_format}"
# Purpose:  Queries the 371 for the vertical and horizontal cursor parameter readouts.
# Argument: STR (default) or SCI. When queried with the SCI argument, the responses are NR3 (scientific notation)
# Response: READOUT <xread>,<yread> where <xread> is horizontal reading in volts and <yread> vertical reading in amperes (ohm or siemens for f line cursor)
# Notes:    Window cursor values are for the location of the bright dot in the corner of the window, and the Window cursor used depends on the last front-panel Cursor mode setting
# Notes:    The response can be specified to be in either string or scientific notation format
# Notes:    If the cursor is offscreen, the returned values will be preceded by question marks and are not valid

TEX_SET = "TEX {string}"
# Purpose:  Allows the controller to write a text string on the 371 display area of the CRT graticule
# Syntax:   TEX "<text>" where text is a message with a length of no more than 24 characters

TEX_QUERY = "TEX?"
# Purpose:  Queries the 371 for any text displayed in the text area of the CRT graticule
# Response: TEXT "<text>" where text is a message with a length of no more than 24 characters
# Notes:    Although text may be stored in bubble memory along with the settings, it can be sent over the bus only with this query
# Notes:    The SET? query does not send text over the bus. Use the TEX? query for this purpose.

"""
CRT Commands and Queries
The Cursor group selects Cursor mode and positions the selected cursor on the display, or queries the 371 on the position of the cursor
"""
CURS_SET = "CURS OFF"
# Purpose:  Sets the 371 cursor mode to off
# Syntax:   CURS OFF

DOT_SET = "DOT {data}"
# Purpose:  Sets the 371 dot cursor position to a specific point on the currently displayed curve
# Syntax:   DOT <data> where data is 0 or 1 or 2 or 3 ... 1024 (0 is the beginning of the curve and 1024 is the end)

DOT_QUERY = "DOT?"
# Purpose:  Queries the 371 for the location of the dot cursor on the currently displayed curve
# Response: DOT <NR1> where NR1 is a number between 0 and 1024 specifying the location of the dot cursor on the currently displayed curve (0 is the beginning of the curve and 1024 is the end)

LIN_SET = "LIN {hor},{ver}"
# Purpose:  Sets the f line cursor intercept position on the display
# Syntax:   LIN <data1>,<data2> where data1 goes from 0 to 1000 horizontal position and data2 from 0 to 1000 vertical position. This position must be a point on a square grid whose bottom left corner coordinates are 0,0 and top right corner coordinates are 1000,1000

LIN_QUERY = "LIN?"
# Purpose:  Queries the 371 for the intercept position of the f line cursor on the display
# Response: LINE <data1>,<data2> where data1 goes from 0 to 1000 horizontal position and data2 from 0 to 1000 vertical position.

WIN_SET = "WIN {bott_l_hor},{bott_l_ver},{up_r_hor},{up_r_ver}"
# Purpose:  Positions and sizes the window cursor on the CRT graticule
# Syntax:   WIN <data1>,<data2>,<data3>,<data4> where data1 0..1000 bottom left horizontal position, data2 0..1000 bottom left vertical position, data3 0..1000 upper right horizontal position, data4 0..1000 upper right vertical position

WIN_QUERY = "WIN?"
# Purpose:  Queries the 371 for the positions and sized the window cursor on the CRT graticule
# Response: WINDOW <data1>,<data2>,<data3>,<data4> where data1 0..1000 bottom left horizontal position, data2 0..1000 bottom left vertical position, data3 0..1000 upper right horizontal position, data4 0..1000 upper right vertical position

"""
Display Commands and Queries
The Display group of commands and queries controls and reports the status of the display. The 371 settings controlled by this grou include: mode, polarity, source, sensitivity, and calibration mode. Also included by this group is a command to send store-mode displays to specified location in mass storage
"""
# DIS_SET = "DIS {mode}"
# Purpose:  Set and change the display mode, polarity, and the calibration mode
# Syntax:   DIS <mode> where mode is NST (non-store mode) or STO (store mode), to select the mode
# Syntax:   DIS VIE:<index> where index is integer 1...16 (bubble memory index), to select view mode and display a curve from the specified location
# Syntax:   DIS COM:<index> where index is integer 1..16 (bubble memory index), to select Compare mode and display curve from the specified location
# Syntax:   DIS INV:<status> where status is ON or OFF, to set or reset the display to invert mode
# Syntax:   DIS CAL:<status> where status is ZER or OFF or FUL, to set the CRT calibration mode
# --- This command has been divided into different commands so it is easier to use ---
DIS_MODE_SET = "DIS {mode}"          # NST | STO
DIS_VIEW_SET = "DIS VIE:{index}"     # view mode, index is 1...16 bubble memory location
DIS_COMP_SET = "DIS COM:{index}"     # compare mode, index is 1...16 bubble memory location
DIS_INV_SET  = "DIS INV:{status}"    # ON | OFF, set display to invert mode
DIS_CAL_SET  = "DIS CAL:{status}"    # ZER | OFF | FUL


DIS_QUERY = "DIS?"
# Purpose:  Queries the 371 for the current settings of the display, polarity, and calibration mode
# Response: DISPLAY <mode1>,<mode2>,<mode3> where mode1 is NSTORE or STORE or VIEW:<index> (index is 1...16 bubble memory index of curve) or COMPARE:<index> (index is 1..16 bubble memory index of curve). mode2 is INVERT:OFF or INVERT:ON. mode3 is CAL:ZERO or CAL:OFF or CAL:FULL

ENT_SET = "ENT {index}"
# Purpose:  Store the display in mass storage. Only valid in store or view mode
# Syntax:   ENT <index> where index is 1...16 (memory location)

HOR_SET = "HOR {source}:{volt}"
# Purpose:  Set the 371 horizontal display source and sensitivity
# Syntax:   HOR <source>:<volt>, where source is COL | STP
# COL:      HOR COL:<volt>, volt may be 5.0E-1 to 5.0E+0 if peak watts is set to 3KW/300W, 5.0E+1 to 5.0E+2 if peak watts is set to 30W/3W
# STP:      HOR STP:<volt>, volt may be 1.0E-1 to 5.0E+0

HOR_QUERY = "HOR?"
# Purpose:  Queries the 371 for the current horizontal source and sensitivity settings
# Response: HORIZ <source>:<volt>, where source is COLLECT or STPGEN and volt is sensitivity (volt/div) in scientific notation (NR3)

VER_SET = "VER COL:{amp}"
# Purpose:  Set the vertical sensitivity of 371
# Syntax:   VER COL:<amp>, where amp 1.0E+0 to 5.0E+1 when peak watts is 3 kW, 500E-03 to 5.0E+0 when peak watts is 300W, 1.0E-4 to 5.0E-3 when peak watts is 30W, 1.0E-4 to 5.0E-3 when peak watts is 30W, 1.0E-5 to 5.0E-4 when peak watts is 3W

VER_QUERY = "VER?"
# Purpose:  Queries the 371 for the vertical sensitivity settings
# Response: VER COL:<amp>, where amp is sensitivity in A/div

"""
Instrument Parameter Commands and Queries
The Instrument Parameter commands and queries group is helpful for determining the status of the 371 when problems are encountered.
"""
DEB_SET = "DEB {status}"
# Purpose:  Sets the debug mode. When ON, the 371 momentarily displays the last 15 characters of the received string in the error message area of the display
# Syntax:   DEB <status>, where status is ON or OFF

DEB_QUERY = "DEB?"
# Purpose:  Queries the 371 for the status of the debug mode
# Response: DEBUG <status>, where status is ON or OFF

HEL_QUERY = "HEL?"
# Purpose:  Ask the 371 for a list of all valid command and query headers
# Response: HELP READOUT, TEXT, LINE, DOT, WINDOW, CURSOR, DISPLAY, HORIZ, VERT, STEPGEN, MEASURE, ENTER, RECALL, SAVE, PLOT, PSTATUS, PKPOWER, CSPOL, CSOUT, VCSPPLY, OUTPUTS, WFMPRE, CURVE, WAVFRM, RQS, OPC, EVENT, TEST, INIT, ID, DEBUG, SET

ID_QUERY = "ID?"
# Purpose:  Queries the 371 for its firmware version
# Response: ID SONY_TEK/371,V81.1F <version>, where version is the current firmware version

INI_SET = "INI"
# Purpose:  Initializes the 371 to its power-up settings
# Syntax:   INI
# Initializes the instrument as DIS STORE, CUR OFF, DIS CAL:OFF, DIS INV:OFF, STP CUR:1.0E-3, STP OFF:0.00, STP INV:0FF, PKP 300, CSP NPN, HOR COL:1.0E+0, OPC OFF, MEA REPEAT, STP NUM:2, STP MUL:OFF, VCS 0.0, VER COL:1.0E+0, RQS ON, DEB OFF

SET_QUERY = "SET?"
# Purpose:  Queries the 371 for its current front-panel settings
# Response: OPC <mode>;RQS <mode>;PKPOWER <watts>;CSPOL <polarity>;HORIZ <source:size>;VERT COLLECT:<size>;STEPGEN OUT:<mode>,NUMBER:<number>,OFFSET:<offset>,INVERT<mode>,MULT<mode>,<source:size>;VCSPPLY <percent>;MEASURE <mode>;DISPLAY INVERT:<mode>,CAL:<mode>,<display mode>;CURSOR<mode>
# Response when measurement mode is SWE (sweep) or SSW (slow sweep): OPC <mode>;RQS <mode>;PKPOWER <watts>;CSPOL <polarity>;HORIZ <source:size>;VERT COLLECT:<size>;STEPGEN OUT:<mode>,NUMBER:<number>;OFFSET:<offset>,INVERT:<mode>,MULT<mode>,<source:size>;VCSPPLY <percent>;DISPLAY INVERT:<mode>,CAL:<mode>,<display mode>;MEASURE SWEEP;CURSOR <mode>

TES_QUERY = "TES?"
# Purpose:  Perform tests on the ROM and RAM
# Response: TEST ROM:0000,RAM:0000 (no error found)
# See the Service Manual for codes other than 0000.

"""
Miscellaneous Commands and Queries
The miscellaneous commands and queries group contains queries for the status of the output connectors, and the measurement code, as well as commands to set the measurement mode and save and recall sets of front-panel settings.
"""

MEA_SET = "MEA {mode}"
# Purpose:  Selects the measurement mode
# Syntax:   MEA <mode>, where mode is REP | SIN | SWE | SSW
# REP:      Repeat
# SIN:      Single
# SWE:      Sweep
# SSW:      Slow sweep

MEA_QUERY = "MEA?"
# Purpose:  Queries the 371B for the current measurement mode setting
# Response: MEASURE <mode>, where mode is REPEAT or SINGLE or SWEEP or SSWEEP

OUT_QUERY = "OUT?"
# Purpose:  Queries the 371 for the status of the output connectors
# Response: OUTPUTS <status>, where status is ENABLED or DISABLED
# ENABLED:  All connector outputs enabled except interlock.
# DISABLED: All connector outputs disabled except interlock.

PLO_SET = "PLO {mode}"
# Purpose:  Defines which data will be printed and starts the printing process
# Syntax:   PLO <mode>, where mode is ALL | CUR
# ALL:      Curve without readout data and graticule
# CUR:      Curve only

PST_QUERY = "PST?"
# Purpose:  Queries the 371 for the status of the printer interface
# Response: PSTATUS <status>, where status is READY or BUSY

REC_SET = "REC {index}"
# Purpose:  Recalls a set of front-panel settings
# Syntax:   REC <index>, where index is 1...16 (bubble memory location)

SAV_SET = "SAV {index}"
# Purpose:  Saves the current set of front-panel settings
# Syntax:   SAV <index>, where index is 1...16 (bubble memory location)

"""
Status and Event Commands and Queries
The Status and Event Reporting group sets and reports the status of service requests and operation complete service requests. A query is also included for the event code of the latest event.
"""

EVE_QUERY = "EVE?"
# Purpose:  Queries the 371 for the event code of the most recent event
# Response: EVENT <code>, where code is a three-digit event code. Refer to "Event Codes" at table 4-10 of the manual

OPC_SET = "OPC {status}"
# Purpose:  Sets the status of operation complete service request
# Syntax:   OPC <status>, where status is ON or OFF
# Notes:    Enable or disable assertion of operation complete service request upon completion of an operation, a change in the circuit breaker status, or a change in the status of the interlock system

OPC_QUERY = "OPC?"
# Purpose:  Queries the 371 for the status of the operation complete service request (OPC)
# Response: OPC <status>, where status is ON or OFF

RQS_SET = "RQS {status}"
# Purpose:  Sets the status of Service Requests
# Syntax:   RQS <status>, where status is ON or OFF
# Notes:    Enabled or disabled assertion of service requests (SRQs)

RQS_QUERY = "RQS?"
# Purpose:  Queries the 371 for the status of service request
# Response: RQS <status>, where status is ON or OFF

"""
Step Generator Commands and Queries
The Step Generator command and query group sets and asks for the status of the step generator settings.
"""

# STP_SET = "STP {mode}"
# Purpose:  Sets the step generator source, step size, number of steps, polarity, step multiplication, and offset.
# Syntax:   STP OUT:<mode>, where mode is ON or OFF (enable or disable the step generator output)
# Syntax:   STP CUR:<val>, where val is 1.0E-6 through 2.0E-3 when peak power is 30W/3W and 1.0E-3 to 2.0E+0 when peak power is 3kW/300W for current step size (amp/div) (set the step generator to provide current steps and the step size in amperes)
# Syntax:   STP VOL:<val>, where val is 2.0E-1 through 5.0E+0 in a 1-2-5 sequence for voltage step size (volt/step) (set the step generator to provide voltage steps and the step size in volts)
# Syntax:   STP NUM:<val>, where val is 0,1,2,...,5 (number of steps to be generated)
# Syntax:   STP INV:<mode>, where mode is ON or OFF (set the step generator polarity)
# Syntax:   STP MUL:<mode>, where mode is ON or OFF (set the step generator step multi 0.1X mode)
# Syntax:   STP OFF:<mode>, where mode is ON or OFF (enable or disable the step generator offset)
# Syntax:   STP OFF:<val>, where val is 0 to 500 the step/offset setting with step multi on. 0 to 5 times the step/offset setting. (to set the offset of the step generator)
# --- This command has been divided into different commands so it is easier to use ---
STP_OUT_SET = "STP OUT:{mode}"       # ON | OFF, enable or disable step generator
STP_CUR_SET = "STP CUR:{val}"        # current step size
STP_VOL_SET = "STP VOL:{val}"        # voltage step size
STP_NUM_SET = "STP NUM:{val}"        # number of steps
STP_INV_SET = "STP INV:{mode}"       # ON | OFF, set polarity
STP_MUL_SET = "STP MUL:{mode}"       # ON | OFF, set step multi 0.1X mode
STP_OFF_MODE_SET = "STP OFF:{mode}"  # ON | OFF, enable or disable step generator offset
STP_OFF_SET = "STP OFF:{val}"        # offset value


STP_QUERY = "STP?"
# Purpose:  Queries the 371 for the current settings of the step generator source, amps/step or volt/step, number of steps, offset, polarity, multiplier mode and output mode
# Response: STPGEN NUMBER:<num>,OFFSET:<offset>,INVERT:<invert>,MULT:<mult>,<typ:size>,<mult>,<typ:size>,<output>, where num is number of steps, 0-5, offset is offset value multiplier, invert is inver mode status on or off, mult is step multi status on or off, typ:size is CURRENT:size (A/step) or VOLTAGE:size (V/step), output is on or off

"""
Waveform Transfer Commands and Queries
The Waveform Transfer group allows curve or preamble data (or both) to be stored in, or recalled from, mass storage. There is also a command to set the number of curve data points stored and related query to determine the length of a previously defined waveform.
"""

CUR_SAVE = "CUR {string}"
# Purpose:  Store a curve into the specified bubble memory location
# Syntax:   CUR <string>, where string is CURVID:<curveid>,%<binary data>
# curveid:  "INDEX <index>", where index is 1...16 for bubble memory storage location
# binary data:  <count><first point>...<last point><checksum>, where count is two bytes indicating the number of data points plus one, point is two bytes indicating the X coordinate and two bytes indicating the Y coordinate for a point (00 through FF) and checksum is one byte, the two's complement of the modulo-256 sum of the preceding binary data.

CUR_QUERY = "CUR?"
# Purpose:  Queries the 371 for curve data. It responds with the curve data for the view curve when in view mode, and with the curve data for the current display when in store mode.
# Response: CURVE CURVEID <curveid>,%<binary data>
# curveid:  "INDEX <index>", for a Store mode curve, index is 1...16 for bubble memory storage location
# binary data:  <count><first point>...<last point><checksum>, where count is two bytes indicating the number of data points plus one, point is two bytes indicating the X coordinate and two bytes indicating the Y coordinate for a point (00 through FF) and checksum is one byte, the two's complement of the modulo-256 sum of the preceding binary data.

WAV_QUERY = "WAV?"
# Purpose: Queries the 371 for the curve and preamble data. This query function as a combination of the WFM? and CUR? queries.
# Notes: Respond with both preamble data and curve data for the current waveform. See the discussions for WFM? and CUR? for details. The preamble and curve data are separated by a semicolon.

WFM_SAVE = "WFM {string}"
# Purpose:  Store the preamble data for the currently displayed waveform into a specified memory location
# Syntax:   WFM <string>, where string is WFID:<wfid>,ENCDG:BIN,NR.PT<point>,PT.FMT:XY,XMULT:<x multi>,XZERO:0,ZOFF:<xoff>,XUNIT:V,YMULT:<y multi>,YZERO:0,YOFF:<yoff>,YUNIT:A,BYT/NR:2,BN.FMT:RP,BIT/NR:10,CRVCHK:CHKSM0,LN.FMT<format>
# <wfid>:   "INDEX <num>/VERT <amp>/HORIZ <volt>/ STEP <step>/OFFSET <offset>/BGM <para>/VCS <percent>/TEXT <txt>/HSNS <mode>"
#           <num>: display address: 0 for CRT, 1...16 for memory location, integer
#           <amp>: sensitivity A/div, scientific notation number
#           <volt>: sensitivity V/div, scientific notation number
#           <step>: step amplitude, V or A/step, scientific notation number
#           <offset>: step offset, V or A
#           <para>: beta or gm
#           <percent>: Collector Supply Variable setting, %
#           <txt>: readout of text area
#           <mode>: horizontal source, VCE or VBE
# <point>:  number of points in the curve (1 through 1024)
# <x multi>: horizontal scale factor, scientific notation number
# <x off>:  horizontal offset, integer
# <y multi>: vertical scale factor, scientific notation number
# <y off>:  vertical offset, integer
# <format>: VECTOR, DOT or SWEEP <cnt>, where cnt is sweep count: 1...6

WFM_QUERY = "WFM?"
# Purpose:  Queries the 371 for the preamble data stored in a specified memory location
# Response: WFMPRE WFID:<wfid>,ENCDG:BIN,NR.PT<point>,PT.FMT:XY,XMULT:<x multi>,XZERO:0,XOFF:<xoff>,XUNIT:V,YMULT:<y multi>,YZERO:0,YOFF:<yoff>,YUNIT:A,BYT/NR:2,BN.FMT:RP,BIT/NR:10,CRVCHK:CHKSM0,LN.FMT:<format>
# <wfid>:   "INDEX <num>/VERT <amp>/HORIZ <volt>/ STEP <step>/OFFSET <offset>/BGM <para>/VCS <percent>/TEXT <txt>/HSNS <mode>"
#           <num>: display address: 0 for CRT, 1...16 for memory location, integer
#           <amp>: sensitivity A/div, scientific notation number
#           <volt>: sensitivity V/div, scientific notation number
#           <step>: step amplitude, V or A/step, scientific notation number
#           <offset>: step offset, V or A
#           <para>: beta or gm
#           <percent>: Collector Supply Variable setting, %
#           <txt>: readout of text area
#           <mode>: horizontal source, VCE or VBE
# <point>:  number of points in the curve (1 through 1024)
# <x multi>: horizontal scale factor, scientific notation number
# <x off>:  horizontal offset, integer
# <y multi>: vertical scale factor, scientific notation number
# <y off>:  vertical offset, integer
# <format>: VECTOR, DOT or SWEEP <cnt>, where cnt is sweep count: 1...6

WFM_LENGTH_SET = "WFM NR.PT:{points}"
# Purpose:  Sets the length of the waveform
# Syntax:   WFM NR.PT:<points>, where points is 1...1024
# Notes:    Set the number of points input for the CUR command

WFM_LENGTH_QUERY = "WFM? NR.PT"
# Purpose:  Queries the 371 for the length of a waveform previously defined with the WFM NR.PT command
# Response: WFM? NR.PT:<points>, where points is 1...1024