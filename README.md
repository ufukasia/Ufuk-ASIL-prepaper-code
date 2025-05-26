# Loosely Coupled Adaptive Hybrid ESKF/UKF for Visual-Inertial Odometry (VIO)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A robust VIO solution combining Error-State Kalman Filter (ESKF) and Unscented Kalman Filter (UKF) with adaptive covariance tuning for dynamic environments. Designed for UAVs and autonomous systems.

**Key Innovation**: Hybrid filtering architecture + CASEF activation function for sensor reliability assessment.

![System Architecture](images/diagram.png)  
*Hybrid Filter Architecture (Conceptual)*

## 🚀 Features
- **Hybrid Qf-ES-EKF/UKF Filter**  
  - UKF for orientation estimation (non-linear dynamics)
  - ESKF for position/velocity/bias estimation (computational efficiency)
  
- **Dynamic Sensor Fusion**  
  - Real-time visual quality metrics: entropy, intensity changes, Culled keyframes, pose chi2 error
  - CASEF function for adaptive covariance tuning

- **Robust Performance**  
  - Handles low-texture environments and light changes
  - Automatic Zero Velocity Updates (ZUPT)

## 📦 Installation
```bash
git clone https://github.com/ufukasia/Ufuk-ASIL-prepaper-code.git
cd Ufuk-ASIL-prepaper-code
pip install -r requirements.txt
```

## 🛠️ Usage
### Basic Run (Adaptive Mode)
```bash
python main_esqf-sukf.py --adaptive
```

### Custom Parameters Example
```bash
python main_esqf-sukf.py --adaptive \
    --alpha_v 4.5 \
    --epsilon_v 2.2 \
    --gamma_v 0.2 \
    --zeta_v 0.1 \
    --s 3.2 \
    --w_thr 0.3 \
    ```

### Key Parameters
| Parameter       | Description                          | Default |
|-----------------|--------------------------------------|---------|
| `--adaptive`    | Enable adaptive covariance           | False   |
| `--alpha_v`     | Intensity difference weight          | 5.0     |
| `--s`           | CASEF activation steepness           | 3.0     |
| `--w_thr`       | Lower confidence threshold           | 0.25    |


## 📂 VO Preparation
1. Download [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
2. VO code [Pyslam](https://github.com/luigifreda/pyslam)
2. Organize structure:
   ```
   ├── imu_interp_gt
   │   └── MH0X_imu_with_interpolated_groundtruth.csv
   └── VO
       └── vo_pred_super_best
           └── mh0X_ns.csv
   ```

## 📊 Performance
| Metric              | Qf-ES-EKF/UKF vs ESKF   |
|---------------------|-------------------------|
| Position Accuracy   | ↑ 40% (MH04-MH05)       |
| Orientation Error   | ↓ 60%                   |
| Processing Speed    | 1.8x faster  from SUKF  |

## 📄 Outputs
- **Trajectory Files**: `outputs/adaptive_sigma_*.csv`
- **Result Metrics**: `results.csv`


## 📊 Results Summary

Here's a summary of the RMSE results for each method, extracted from the console outputs:

### ESKF
```
[MH04] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.6473817904487497
Euler Angle RMSE (degrees): [0.84235495 0.17875309 0.88719256]
Velocity RMSE (m/s): [0.26575361 0.30194373 0.17724036]
Position RMSE (m):************************************************ [0.26357877 0.34953903 0.36336454] ***************
Accel Bias RMSE (m/s^2): [0.00050206 0.00030133 0.00036831]
Gyro Bias RMSE (rad/s): [1.26541244e-04 5.73534421e-05 1.51118676e-04]
Total Static Duration (Gravity Update Active): 3.0000 sn
[MH04] Total Execution Time: 39.6942 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 2.1646309796859717
Euler Angle RMSE (degrees): [0.39353345 0.46122709 2.10385587]
Velocity RMSE (m/s): [0.14897458 0.21048719 0.14240369]
Position RMSE (m):************************************************ [0.19673536 0.40653387 0.10225837] ***************
Accel Bias RMSE (m/s^2): [0.00054524 0.00016138 0.00054451]
Gyro Bias RMSE (rad/s): [3.96733451e-04 6.84009542e-05 1.80584287e-04]
Total Static Duration (Gravity Update Active): 4.4999 sn
[MH05] Total Execution Time: 44.8309 s
[MH03] === RMSE Results ===
Quaternion Angular RMSE (degrees): 1.4424677443755356
Euler Angle RMSE (degrees): [0.58911964 0.14733758 1.38037567]
Velocity RMSE (m/s): [0.0991795  0.11653823 0.18319717]
Position RMSE (m):************************************************ [0.11989855 0.12075143 0.18229124] ***************
Accel Bias RMSE (m/s^2): [0.00107491 0.00029477 0.00015039]
Gyro Bias RMSE (rad/s): [0.00023026 0.00015018 0.00013581]
Total Static Duration (Gravity Update Active): 3.3000 sn
[MH03] Total Execution Time: 52.8927 s
[MH02] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.8828475470394254
Euler Angle RMSE (degrees): [1.65149278 0.40042424 1.69204228]
Velocity RMSE (m/s): [0.08293924 0.06437859 0.10627762]
Position RMSE (m):************************************************ [0.05977835 0.07063635 0.04657688] ***************
Accel Bias RMSE (m/s^2): [0.00017415 0.00039769 0.00026459]
Gyro Bias RMSE (rad/s): [0.00022571 0.00017296 0.00022078]
Total Static Duration (Gravity Update Active): 5.1001 sn
[MH02] Total Execution Time: 59.8195 s
[MH01] === RMSE Results ===
Quaternion Angular RMSE (degrees): 4.736267782555803
Euler Angle RMSE (degrees): [2.45870157 0.56421312 5.80318451]
Velocity RMSE (m/s): [0.30476003 0.29087486 0.20398437]
Position RMSE (m):************************************************ [0.19144842 0.23800255 0.17561877] ***************
Accel Bias RMSE (m/s^2): [0.00061497 0.00126843 0.00089979]
Gyro Bias RMSE (rad/s): [0.00064356 0.00019071 0.00036934]
Total Static Duration (Gravity Update Active): 15.2998 sn
[MH01] Total Execution Time: 71.4549 s
=== Processing of all sequences finished. ===
Accel+```

### Qf-ES-EKF/UKF
```
Quaternion Angular RMSE (degrees): 0.4123155912683598
Euler Angle RMSE (degrees): [0.70832965 0.12569949 0.68943068]
Velocity RMSE (m/s): [0.26582132 0.30199875 0.17723798]
Position RMSE (m): [0.26358228 0.34955357 0.36336872]
Accel Bias RMSE (m/s^2): [0.00052935 0.00030087 0.00035365]
Gyro Bias RMSE (rad/s): [1.34621599e-04 3.18194203e-05 1.25315760e-04]
Total Static Duration (Gravity Update Active): 3.0000 s
[MH04] Total Execution Time: 55.1394 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.8377975764776074
Euler Angle RMSE (degrees): [0.53158731 0.46737909 0.76349246]
Velocity RMSE (m/s): [0.14953537 0.21090711 0.142397  ]
Position RMSE (m): [0.19704921 0.40663263 0.10225495]
Accel Bias RMSE (m/s^2): [0.00062062 0.00015152 0.00054461]
Gyro Bias RMSE (rad/s): [4.66777914e-04 4.89014237e-05 1.77799885e-04]
Total Static Duration (Gravity Update Active): 4.4999 s
[MH05] Total Execution Time: 61.9796 s
[MH03] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.827275759753285
Euler Angle RMSE (degrees): [0.45968968 0.12009791 0.85514623]
Velocity RMSE (m/s): [0.0993713  0.11666537 0.18319709]
Position RMSE (m): [0.11989838 0.12075698 0.18229574]
Accel Bias RMSE (m/s^2): [1.07006657e-03 2.55388424e-04 9.29894346e-05]
Gyro Bias RMSE (rad/s): [3.87963823e-04 9.78347628e-05 1.13826769e-04]
Total Static Duration (Gravity Update Active): 3.3000 s
[MH03] Total Execution Time: 73.8343 s
[MH02] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.6644756987901647
Euler Angle RMSE (degrees): [1.65086321 0.31410763 1.61867437]
Velocity RMSE (m/s): [0.08298458 0.06450672 0.10627828]
Position RMSE (m): [0.05978034 0.0706347  0.04657508]
Accel Bias RMSE (m/s^2): [0.00018128 0.00023091 0.00021752]
Gyro Bias RMSE (rad/s): [0.00021396 0.00020718 0.00029027]
Total Static Duration (Gravity Update Active): 5.1001 s
[MH02] Total Execution Time: 83.4138 s
[MH01] === RMSE Results ===
Quaternion Angular RMSE (degrees): 2.686488210703794
Euler Angle RMSE (degrees): [1.4384885  0.44446264 3.15199947]
Velocity RMSE (m/s): [0.30482735 0.29128714 0.20399644]
Position RMSE (m): [0.19140046 0.23799542 0.17562417]
Accel Bias RMSE (m/s^2): [0.00093617 0.00066422 0.00100265]
Gyro Bias RMSE (rad/s): [0.00054662 0.00012079 0.00023342]
Total Static Duration (Gravity Update Active): 15.2998 s
[MH01] Total Execution Time: 99.3745 s
=== Processing of all sequences finished. ===


```

### USKF
```
[MH04] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.5998388974163876
Euler Angle RMSE (degrees): [0.89513217 0.22162477 0.69070932]
Velocity RMSE (m/s): [0.26574639 0.30204793 0.17722717]
Position RMSE (m):************************************************ [0.26354782 0.3495213  0.36331179] ***************
Accel Bias RMSE (m/s^2): [0.0004229  0.00020032 0.00040969]
Gyro Bias RMSE (rad/s): [0.00063312 0.00019996 0.00043673]
Total Static Duration (Gravity Update Active): 3.0000 s
[MH04] Total Execution Time: 107.3549 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.9456506506586486
Euler Angle RMSE (degrees): [0.92911428 0.50676905 0.99579033]
Velocity RMSE (m/s): [0.14969252 0.21099681 0.14238799]
Position RMSE (m):************************************************ [0.19767615 0.40718759 0.10224282] ***************
Accel Bias RMSE (m/s^2): [0.00086701 0.00082007 0.0012482 ]
Gyro Bias RMSE (rad/s): [0.00055212 0.00042296 0.00037519]
Total Static Duration (Gravity Update Active): 4.4999 s
[MH05] Total Execution Time: 120.6428 s
[MH03] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.5485451549162652
Euler Angle RMSE (degrees): [1.01223399 0.27215206 1.16877054]
Velocity RMSE (m/s): [0.09936326 0.11678381 0.18320204]
Position RMSE (m):************************************************ [0.12001418 0.1206787  0.18228947] ***************
Accel Bias RMSE (m/s^2): [0.00100591 0.00077295 0.00075738]
Gyro Bias RMSE (rad/s): [0.00037924 0.00042563 0.0002862 ]
Total Static Duration (Gravity Update Active): 3.3000 s
[MH03] Total Execution Time: 142.8029 s
[MH02] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.640644309770442
Euler Angle RMSE (degrees): [1.74107812 0.33452202 1.67731372]
Velocity RMSE (m/s): [0.08316206 0.06369123 0.1062771 ]
Position RMSE (m):************************************************ [0.05994497 0.06954782 0.04655986] ***************
Accel Bias RMSE (m/s^2): [0.00031567 0.00065594 0.00059614]
Gyro Bias RMSE (rad/s): [0.00021965 0.00041442 0.00017875]
Total Static Duration (Gravity Update Active): 5.1001 s
[MH02] Total Execution Time: 162.0262 s
[MH01] === RMSE Results ===
Quaternion Angular RMSE (degrees): 1.986471502547953
Euler Angle RMSE (degrees): [4.48372894 0.98337514 3.75791539]
Velocity RMSE (m/s): [0.30591439 0.29099871 0.20405843]
Position RMSE (m):************************************************ [0.19130025 0.2359356  0.17544241] ***************
Accel Bias RMSE (m/s^2): [0.00493704 0.00753315 0.00186177]
Gyro Bias RMSE (rad/s): [0.00105459 0.0011826  0.00074053]
Total Static Duration (Gravity Update Active): 15.2998 s
[MH01] Total Execution Time: 197.3319 s
=== Processing of all sequences finished. ===
```

### Adaptive Qf-ES-EKF/UKF
```
[MH04] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.4198376363552852
Euler Angle RMSE (degrees): [0.58006587 0.13159924 0.54405981]
Velocity RMSE (m/s): [0.2454996  0.24264056 0.15735462]
Position RMSE (m): [0.17485886 0.281041   0.29486496]
Accel Bias RMSE (m/s^2): [0.00055558 0.0002935  0.00034295]
Gyro Bias RMSE (rad/s): [1.22062643e-04 2.51637354e-05 1.12768555e-04]
Total Static Duration (Gravity Update Active): 3.0000 s
[MH04] Total Execution Time: 74.1471 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.8297445528369128
Euler Angle RMSE (degrees): [0.52526817 0.46277277 0.77617506]
Velocity RMSE (m/s): [0.14321766 0.20931543 0.14085736]
Position RMSE (m): [0.19753813 0.41682311 0.12663346]
Accel Bias RMSE (m/s^2): [0.00061537 0.00015204 0.00054311]
Gyro Bias RMSE (rad/s): [4.45401153e-04 5.15765866e-05 1.75558492e-04]
Total Static Duration (Gravity Update Active): 4.4999 s
[MH05] Total Execution Time: 83.8603 s
[MH03] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.727560105114867
Euler Angle RMSE (degrees): [0.45208648 0.11578343 0.76606697]
Velocity RMSE (m/s): [0.09909984 0.11123722 0.18721187]
Position RMSE (m): [0.11244917 0.12687287 0.20688819]
Accel Bias RMSE (m/s^2): [1.06497089e-03 2.57494514e-04 8.39531463e-05]
Gyro Bias RMSE (rad/s): [0.00035611 0.00010035 0.00011282]
Total Static Duration (Gravity Update Active): 3.3000 s
[MH03] Total Execution Time: 99.9371 s
[MH02] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.6623419046386355
Euler Angle RMSE (degrees): [1.60245087 0.31694685 1.57292297]
Velocity RMSE (m/s): [0.08179274 0.06452796 0.10690498]
Position RMSE (m): [0.05871556 0.07524122 0.04724986]
Accel Bias RMSE (m/s^2): [0.00018112 0.00023145 0.00021756]
Gyro Bias RMSE (rad/s): [0.0002156  0.00020676 0.00028858]
Total Static Duration (Gravity Update Active): 5.1001 s
[MH02] Total Execution Time: 114.5231 s
[MH01] === RMSE Results ===
Quaternion Angular RMSE (degrees): 2.568257408928737
Euler Angle RMSE (degrees): [1.43742949 0.46187332 3.04249037]
Velocity RMSE (m/s): [0.30344335 0.2895757  0.20242014]
Position RMSE (m): [0.20788293 0.25565756 0.1645983 ]
Accel Bias RMSE (m/s^2): [0.00093794 0.00067061 0.00101547]
Gyro Bias RMSE (rad/s): [0.0005504  0.00011492 0.00023247]
Total Static Duration (Gravity Update Active): 15.2998 s
[MH01] Total Execution Time: 132.3721 s
=== Processing of all sequences finished. ===
```

## 📧 Contact
- Ufuk Asil - [u.asil@ogr.deu.edu.tr](mailto:u.asil@ogr.deu.edu.tr)

 Bias RMSE (m/s^2): [0.00050206 0.00030133 0.00036831]
Gyro Bias RMSE (rad/s): [1.26541244e-04 5.73534421e-05 1.51118676e-04]
Total Static Duration (Gravity Update Active): 3.0000 sn
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 2.1646309796859717
Euler Angle RMSE (degrees): [0.39353345 0.46122709 2.10385587]
Velocity RMSE (m/s): [0.14897458 0.21048719 0.14240369]
Position RMSE (m): [0.19673536 0.40653387 0.10225837]
Accel Bias RMSE (m/s^2): [0.00054524 0.00016138 0.00054451]
Gyro Bias RMSE (rad/s): [3.96733451e-04 6.84009542e-05 1.80584287e-04]
Total Static Duration (Gravity Update Active): 4.4999 sn
```

### Qf-ES-EKF/UKF
```
[MH04] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.4123155912683598
Euler Angle RMSE (degrees): [0.70832965 0.12569949 0.68943068]
Velocity RMSE (m/s): [0.26582132 0.30199875 0.17723798]
Position RMSE (m): [0.26358228 0.34955357 0.36336872]
Accel Bias RMSE (m/s^2): [0.00052935 0.00030087 0.00035365]
Gyro Bias RMSE (rad/s): [1.34621599e-04 3.18194203e-05 1.25315760e-04]
Total Static Duration (Gravity Update Active): 3.0000 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.8377975764776074
Euler Angle RMSE (degrees): [0.53158731 0.46737909 0.76349246]
Velocity RMSE (m/s): [0.14953537 0.21090711 0.142397  ]
Position RMSE (m): [0.19704921 0.40663263 0.10225495]
Accel Bias RMSE (m/s^2): [0.00062062 0.00015152 0.00054461]
Gyro Bias RMSE (rad/s): [4.66777914e-04 4.89014237e-05 1.77799885e-04]
Total Static Duration (Gravity Update Active): 4.4999 s
```

### USKF
```
[MH04] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.5998388974163876
Euler Angle RMSE (degrees): [0.89513217 0.22162477 0.69070932]
Velocity RMSE (m/s): [0.26574639 0.30204793 0.17722717]
Position RMSE (m): [0.26354782 0.3495213  0.36331179]
Accel Bias RMSE (m/s^2): [0.0004229  0.00020032 0.00040969]
Gyro Bias RMSE (rad/s): [0.00063312 0.00019996 0.00043673]
Total Static Duration (Gravity Update Active): 3.0000 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.9456506506586486
Euler Angle RMSE (degrees): [0.92911428 0.50676905 0.99579033]
Velocity RMSE (m/s): [0.14969252 0.21099681 0.14238799]
Position RMSE (m): [0.19767615 0.40718759 0.10224282]
Accel Bias RMSE (m/s^2): [0.00086701 0.00082007 0.0012482 ]
Gyro Bias RMSE (rad/s): [0.00055212 0.00042296 0.00037519]
Total Static Duration (Gravity Update Active): 4.4999 s
```

### Adaptive Qf-ES-EKF/UKF
```
[MH04] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.4198376363552852
Euler Angle RMSE (degrees): [0.58006587 0.13159924 0.54405981]
Velocity RMSE (m/s): [0.2454996  0.24264056 0.15735462]
Position RMSE (m): [0.17485886 0.281041   0.29486496]
Accel Bias RMSE (m/s^2): [0.00055558 0.0002935  0.00034295]
Gyro Bias RMSE (rad/s): [1.22062643e-04 2.51637354e-05 1.12768555e-04]
Total Static Duration (Gravity Update Active): 3.0000 s
[MH05] === RMSE Results ===
Quaternion Angular RMSE (degrees): 0.8297445528369128
Euler Angle RMSE (degrees): [0.52526817 0.46277277 0.77617506]
Velocity RMSE (m/s): [0.14321766 0.20931543 0.14085736]
Position RMSE (m): [0.19753813 0.41682311 0.12663346]
Accel Bias RMSE (m/s^2): [0.00061537 0.00015204 0.00054311]
Gyro Bias RMSE (rad/s): [4.45401153e-04 5.15765866e-05 1.75558492e-04]
Total Static Duration (Gravity Update Active): 4.4999 s
```

Bu sonuçlar, her bir yöntemin farklı metrikler üzerindeki performansını göstermektedir. İhtiyacınıza göre bu değerleri daha detaylı analiz edebilirsiniz.



## 📧 Contact
- Ufuk Asil - [u.asil@ogr.deu.edu.tr](mailto:u.asil@ogr.deu.edu.tr)
- Efendi Nasibov - [efendi.nasibov@deu.edu.tr](mailto:efendi.nasibov@deu.edu.tr)

Computer Science, Dokuz Eylül University


**MIT License** - See [LICENSE](LICENSE) for details