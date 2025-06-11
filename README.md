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
    --alpha_v 18 \
    --epsilon_v 3.3 \
    --gamma_v 0.2 \
    --zeta_H 0.1 \
    --s 1 \
    --w_thr 0.2\
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




## 📧 Contact
- Ufuk Asil - [u.asil@ogr.deu.edu.tr](mailto:u.asil@ogr.deu.edu.tr)
- Efendi Nasibov - [efendi.nasibov@deu.edu.tr](mailto:efendi.nasibov@deu.edu.tr)

Computer Science, Dokuz Eylül University


**MIT License** - See [LICENSE](LICENSE) for details