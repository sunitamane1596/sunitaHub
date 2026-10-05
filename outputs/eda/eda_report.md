# Exploratory Data Analysis

- Rows: 12,000
- Columns: 18
- Exact duplicate rows: 0
- Numeric columns: 17
- Categorical columns: 1

## Column types and missing values

| Column | Type | Missing | Unique |
|---|---|---:|---:|
| Equipment_ID | str | 0 | 120 |
| Cycle | int64 | 0 | 100 |
| Equipment_Age_Years | float64 | 0 | 114 |
| Operating_Hours | float64 | 0 | 10812 |
| Load_Percent | float64 | 0 | 4388 |
| RPM | float64 | 0 | 4063 |
| Ambient_Temperature_C | float64 | 0 | 2308 |
| Humidity_Percent | float64 | 0 | 4416 |
| Vibration_mm_s | float64 | 0 | 3747 |
| Temperature_C | float64 | 0 | 2472 |
| Pressure_bar | float64 | 0 | 1652 |
| Current_A | float64 | 0 | 825 |
| Voltage_V | float64 | 0 | 1532 |
| Oil_Temperature_C | float64 | 0 | 2168 |
| Flow_Rate_L_min | float64 | 0 | 1946 |
| Acoustic_dB | float64 | 0 | 1713 |
| Maintenance_Count | int64 | 0 | 10 |
| RUL_Cycles | float64 | 0 | 6039 |

## Numeric summary

                          count     mean      std      min      25%      50%      75%      max
Cycle                 12000.000   50.500   28.867    1.000   25.750   50.500   75.250  100.000
Equipment_Age_Years   12000.000    6.248    3.263    1.060    3.183    6.310    9.325   11.860
Operating_Hours       12000.000 3528.372 1647.578  538.800 2049.900 3501.250 5020.500 6778.000
Load_Percent          12000.000   69.927   11.922   35.000   61.950   69.915   77.992  100.000
RPM                   12000.000 2059.884  105.280 1666.300 1989.700 2059.150 2130.800 2400.000
Ambient_Temperature_C 12000.000   24.933    5.014   10.000   21.477   24.960   28.370   40.000
Humidity_Percent      12000.000   55.058   11.873   20.000   46.987   55.070   63.200   90.000
Vibration_mm_s        12000.000    4.983    0.955    2.038    4.287    4.976    5.687    7.879
Temperature_C         12000.000   85.273    5.530   67.690   81.338   85.250   89.200  104.950
Pressure_bar          12000.000    7.414    0.356    6.367    7.151    7.417    7.677    8.449
Current_A             12000.000   25.528    1.472   20.370   24.530   25.540   26.540   30.670
Voltage_V             12000.000  413.947    3.018  403.110  411.920  413.940  415.990  425.980
Oil_Temperature_C     12000.000   66.222    4.837   51.580   62.690   66.210   69.750   80.470
Flow_Rate_L_min       12000.000   86.090    4.084   71.720   83.210   86.100   88.920  101.160
Acoustic_dB           12000.000   70.261    3.676   59.190   67.620   70.260   72.960   81.650
Maintenance_Count     12000.000    3.419    1.888   -1.000    2.000    3.000    5.000    8.000
RUL_Cycles            12000.000   45.404   43.262    5.000    5.000   32.410   76.670  185.330

## Potential outliers (IQR rule)

Counts are diagnostic only; valid sensor extremes are not automatically removed.

| Column | Below Q1 - 1.5 IQR | Above Q3 + 1.5 IQR |
|---|---:|---:|
| Cycle | 0 | 0 |
| Equipment_Age_Years | 0 | 0 |
| Operating_Hours | 0 | 0 |
| Load_Percent | 60 | 0 |
| RPM | 44 | 30 |
| Ambient_Temperature_C | 38 | 34 |
| Humidity_Percent | 34 | 32 |
| Vibration_mm_s | 1 | 3 |
| Temperature_C | 6 | 9 |
| Pressure_bar | 0 | 0 |
| Current_A | 34 | 32 |
| Voltage_V | 36 | 38 |
| Oil_Temperature_C | 2 | 1 |
| Flow_Rate_L_min | 26 | 29 |
| Acoustic_dB | 4 | 1 |
| Maintenance_Count | 0 | 0 |
| RUL_Cycles | 0 | 1 |

## Target review

- Target: `RUL_Cycles`
- Valid target rows: 12,000
- Target at minimum value 5: 4,064 (33.9%)
- Confirm whether the repeated value of 5 represents a capped minimum RUL; this affects interpretation near end-of-life.

## Target correlations

Cycle                   -0.793
Vibration_mm_s          -0.761
Oil_Temperature_C       -0.751
Acoustic_dB             -0.730
Pressure_bar             0.717
Temperature_C           -0.707
Flow_Rate_L_min          0.613
Current_A               -0.561
Operating_Hours         -0.321
Maintenance_Count       -0.314
Equipment_Age_Years     -0.212
Load_Percent            -0.099
RPM                     -0.087
Voltage_V                0.085
Ambient_Temperature_C   -0.036
Humidity_Percent        -0.005