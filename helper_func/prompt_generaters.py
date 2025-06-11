# This script generates batch commands for the main_qf-es-ekf-ukf.py project
# based on user-defined parameter ranges for single parameter sweeps.

import numpy as np
import argparse

def make_range(start, end, step):
    # If step is 0, return an array with a single value.
    if step == 0:
        return np.array([start])
    else:
        # Add a small tolerance to ensure the end point is included.
        return np.arange(start, end + step / 2, step)

def main():
    parser = argparse.ArgumentParser(
        description="Batch command generator for single parameter sweeps of main_qf-es-ekf-ukf.py."
    )



    # Beta_p parametreleri
    parser.add_argument("--beta_p_start", type=float, default=-1, help="beta_p başlangıç değeri")
    parser.add_argument("--beta_p_end", type=float, default=1, help="beta_p end value")
    parser.add_argument("--beta_p_step", type=float, default=0, help="beta_p step value")
    parser.add_argument("--beta_p_active", action="store_true", help="Generate commands for beta_p parameter")
    
    # Epsilon_p parametreleri
    parser.add_argument("--epsilon_p_start", type=float, default=1, help="epsilon_p start value")
    parser.add_argument("--epsilon_p_end", type=float, default=1, help="epsilon_p end value")
    parser.add_argument("--epsilon_p_step", type=float, default=0, help="epsilon_p step value")
    parser.add_argument("--epsilon_p_active", action="store_true", help="Generate commands for epsilon_p parameter")

    # Zeta_p parametreleri
    parser.add_argument("--zeta_p_start", type=float, default=1, help="zeta_p start value")
    parser.add_argument("--zeta_p_end", type=float, default=1, help="zeta_p end value")
    parser.add_argument("--zeta_p_step", type=float, default=0, help="zeta_p step value")
    parser.add_argument("--zeta_p_active", action="store_true", help="Generate commands for zeta_p parameter")


#--------------------------------------------------------------------------------------------------------------------------------------

    
    # Alpha_v parametreleri
    parser.add_argument("--alpha_v_start", type=float, default=0.1, help="alpha_v start value")
    parser.add_argument("--alpha_v_end", type=float, default=20, help="alpha_v end value")
    parser.add_argument("--alpha_v_step", type=float, default=0.2, help="alpha_v step value")
    parser.add_argument("--alpha_v_active", action="store_false", help="Generate commands for alpha_v parameter")
    
    # Epsilon_v parametreleri
    parser.add_argument("--epsilon_v_start", type=float, default=0.1, help="epsilon_v start value")
    parser.add_argument("--epsilon_v_end", type=float, default=20, help="epsilon_v end value")
    parser.add_argument("--epsilon_v_step", type=float, default=0.2, help="epsilon_v step value")
    parser.add_argument("--epsilon_v_active", action="store_false", help="Generate commands for epsilon_v parameter")
    
    
    # Zeta_H parametreleri
    parser.add_argument("--zeta_H_start", type=float, default=0.1, help="zeta_H start value")
    parser.add_argument("--zeta_H_end", type=float, default=20, help="zeta_H end value")
    parser.add_argument("--zeta_H_step", type=float, default=0.2, help="zeta_H step value")
    parser.add_argument("--zeta_H_active", action="store_false", help="Generate commands for zeta_H parameter")
    
    # Zeta_L parametreleri
    parser.add_argument("--zeta_L_start", type=float, default=0.1, help="zeta_L start value")
    parser.add_argument("--zeta_L_end", type=float, default=20, help="zeta_L end value")
    parser.add_argument("--zeta_L_step", type=float, default=0.2, help="zeta_L step value")
    parser.add_argument("--zeta_L_active", action="store_false", help="Generate commands for zeta_L parameter")
    

#--------------------------------------------------------------------------------------------------------------------------------------


    # w_thr parametreleri
    parser.add_argument("--w_thr_start", type=float, default=0.2, help="w_thr start value")
    parser.add_argument("--w_thr_end", type=float, default=0.3, help="w_thr end value")
    parser.add_argument("--w_thr_step", type=float, default=0.01, help="w_thr step value")
    parser.add_argument("--w_thr_active", action="store_true", help="Generate commands for w_thr parameter")
    
    # d_thr parametreleri
    parser.add_argument("--d_thr_start", type=float, default=0.8, help="d_thr start value")
    parser.add_argument("--d_thr_end", type=float, default=0.9, help="d_thr end value")
    parser.add_argument("--d_thr_step", type=float, default=0.01, help="d_thr step value")
    parser.add_argument("--d_thr_active", action="store_true", help="Generate commands for d_thr parameter")
    
    # s parametreleri
    parser.add_argument("--s_start", type=float, default=3.0, help="s start value")
    parser.add_argument("--s_end", type=float, default=3.0, help="s end value")
    parser.add_argument("--s_step", type=float, default=0.1, help="s step value (0 for single value)")
    parser.add_argument("--s_active", action="store_true", help="Generate commands for s parameter")
    


#--------------------------------------------------------------------------------------------------------------------------------------



    # entropy_norm_min parametreleri
    parser.add_argument("--entropy_norm_min_start", type=float, default=-1.0, help="entropy_norm_min start value")
    parser.add_argument("--entropy_norm_min_end", type=float, default=2.0, help="entropy_norm_min end value")
    parser.add_argument("--entropy_norm_min_step", type=float, default=0.1, help="entropy_norm_min step value (0 for single value)")
    parser.add_argument("--entropy_norm_min_active", action="store_false", help="Generate commands for entropy_norm_min parameter")
    
    # pose_chi2_norm_min parametreleri-1.0, help="pose_chi2_norm_min başlangıç değeri")
    parser.add_argument("--pose_chi2_norm_min_start", type=float, default=-1.0, help="pose_chi2_norm_min start value")
    parser.add_argument("--pose_chi2_norm_min_end", type=float, default=2.0, help="pose_chi2_norm_min end value")
    parser.add_argument("--pose_chi2_norm_min_step", type=float, default=0.1, help="pose_chi2_norm_min step value (0 for single value)")
    parser.add_argument("--pose_chi2_norm_min_active", action="store_false", help="Generate commands for pose_chi2_norm_min parameter")
    
    # culled_norm_min parametreleri
    parser.add_argument("--culled_norm_min_start", type=float, default=-1.0, help="culled_norm_min start value")
    parser.add_argument("--culled_norm_min_end", type=float, default=2.0, help="culled_norm_min end value")
    parser.add_argument("--culled_norm_min_step", type=float, default=0.1, help="culled_norm_min step value (0 for single value)")
    parser.add_argument("--culled_norm_min_active", action="store_false", help="Generate commands for culled_norm_min parameter")
    
    parser.add_argument("--output_file", type=str, default="parameter_commands.txt", help="Name of the output file")
    
    args = parser.parse_args()
    
    commands = []
    
    # For active parameters only, a separate command line is generated for each value in the respective range.
    if args.alpha_v_active:
        for val in make_range(args.alpha_v_start, args.alpha_v_end, args.alpha_v_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --alpha_v {val:.3f}')
    if args.beta_p_active:
        for val in make_range(args.beta_p_start, args.beta_p_end, args.beta_p_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --beta_p {val:.3f}')
    if args.epsilon_v_active:
        for val in make_range(args.epsilon_v_start, args.epsilon_v_end, args.epsilon_v_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --epsilon_v {val:.3f}')
    if args.epsilon_p_active:
        for val in make_range(args.epsilon_p_start, args.epsilon_p_end, args.epsilon_p_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --epsilon_p {val:.3f}')
    if args.zeta_H_active:
        for val in make_range(args.zeta_H_start, args.zeta_H_end, args.zeta_H_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --zeta_H {val:.3f}')
    if args.zeta_L_active:
        for val in make_range(args.zeta_L_start, args.zeta_L_end, args.zeta_L_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --zeta_L {val:.3f}')
    if args.zeta_p_active:
        for val in make_range(args.zeta_p_start, args.zeta_p_end, args.zeta_p_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --zeta_p {val:.3f}')
    if args.w_thr_active:
        for val in make_range(args.w_thr_start, args.w_thr_end, args.w_thr_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --w_thr {val:.4f}')
    if args.d_thr_active:
        for val in make_range(args.d_thr_start, args.d_thr_end, args.d_thr_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --d_thr {val:.4f}')
    if args.s_active:
        for val in make_range(args.s_start, args.s_end, args.s_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --s {val:.3f}')
    if args.entropy_norm_min_active:
        for val in make_range(args.entropy_norm_min_start, args.entropy_norm_min_end, args.entropy_norm_min_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --entropy_norm_min {val:.3f}')
    if args.pose_chi2_norm_min_active:
        for val in make_range(args.pose_chi2_norm_min_start, args.pose_chi2_norm_min_end, args.pose_chi2_norm_min_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --pose_chi2_norm_min {val:.3f}')
    if args.culled_norm_min_active:
        for val in make_range(args.culled_norm_min_start, args.culled_norm_min_end, args.culled_norm_min_step):
            commands.append(f'python "main_qf-es-ekf-ukf.py" --culled_norm_min {val:.3f}')
    
    # Write each generated command line to the output file.
    with open(args.output_file, "w") as f:
        for command in commands:
            f.write(command + "\n")
    
    print(f"{len(commands)} commands were written to '{args.output_file}'.")

if __name__ == "__main__":
    main()