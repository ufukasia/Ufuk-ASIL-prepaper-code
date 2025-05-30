import numpy as np
import argparse

def make_range(start, end, step):
    # Eğer step 0 ise tek değerlik array döndür
    if step == 0:
        return np.array([start])
    else:
        # Küçük tolerans ekleyerek dahil olmasını sağlıyoruz
        return np.arange(start, end + step / 2, step)

def main():
    parser = argparse.ArgumentParser(
        description="eskf-vio için tek parametreli batch komutları üreticisi."
    )



    # Beta_p parametreleri
    parser.add_argument("--beta_p_start", type=float, default=-1, help="beta_p başlangıç değeri")
    parser.add_argument("--beta_p_end", type=float, default=1, help="beta_p bitiş değeri")
    parser.add_argument("--beta_p_step", type=float, default=0, help="beta_p adım değeri")
    parser.add_argument("--beta_p_active", action="store_true", help="beta_p parametresi için komut üret")
    
    # Epsilon_p parametreleri
    parser.add_argument("--epsilon_p_start", type=float, default=1, help="epsilon_p başlangıç değeri")
    parser.add_argument("--epsilon_p_end", type=float, default=1, help="epsilon_p bitiş değeri")
    parser.add_argument("--epsilon_p_step", type=float, default=0, help="epsilon_p adım değeri")
    parser.add_argument("--epsilon_p_active", action="store_true", help="epsilon_p parametresi için komut üret")

    # Zeta_p parametreleri
    parser.add_argument("--zeta_p_start", type=float, default=1, help="zeta_p başlangıç değeri")
    parser.add_argument("--zeta_p_end", type=float, default=1, help="zeta_p bitiş değeri")
    parser.add_argument("--zeta_p_step", type=float, default=0, help="zeta_p adım değeri")
    parser.add_argument("--zeta_p_active", action="store_true", help="zeta_p parametresi için komut üret")


#--------------------------------------------------------------------------------------------------------------------------------------

    
    # Alpha_v parametreleri
    parser.add_argument("--alpha_v_start", type=float, default=0.1, help="alpha_v başlangıç değeri")
    parser.add_argument("--alpha_v_end", type=float, default=20, help="alpha_v bitiş değeri")
    parser.add_argument("--alpha_v_step", type=float, default=0.2, help="alpha_v adım değeri")
    parser.add_argument("--alpha_v_active", action="store_false", help="alpha_v parametresi için komut üret")
    
    # Epsilon_v parametreleri
    parser.add_argument("--epsilon_v_start", type=float, default=0.1, help="epsilon_v başlangıç değeri")
    parser.add_argument("--epsilon_v_end", type=float, default=20, help="epsilon_v bitiş değeri")
    parser.add_argument("--epsilon_v_step", type=float, default=0.2, help="epsilon_v adım değeri")
    parser.add_argument("--epsilon_v_active", action="store_false", help="epsilon_v parametresi için komut üret")
    
    
    # Zeta_H parametreleri
    parser.add_argument("--zeta_H_start", type=float, default=0.1, help="zeta_H başlangıç değeri")
    parser.add_argument("--zeta_H_end", type=float, default=20, help="zeta_H bitiş değeri")
    parser.add_argument("--zeta_H_step", type=float, default=0.2, help="zeta_H adım değeri")
    parser.add_argument("--zeta_H_active", action="store_false", help="zeta_H parametresi için komut üret")
    
    # Zeta_L parametreleri
    parser.add_argument("--zeta_L_start", type=float, default=0.1, help="zeta_L başlangıç değeri")
    parser.add_argument("--zeta_L_end", type=float, default=20, help="zeta_L bitiş değeri")
    parser.add_argument("--zeta_L_step", type=float, default=0.2, help="zeta_L adım değeri")
    parser.add_argument("--zeta_L_active", action="store_false", help="zeta_L parametresi için komut üret")
    

#--------------------------------------------------------------------------------------------------------------------------------------


    # w_thr parametreleri
    parser.add_argument("--w_thr_start", type=float, default=0.2, help="w_thr başlangıç değeri")
    parser.add_argument("--w_thr_end", type=float, default=0.3, help="w_thr bitiş değeri")
    parser.add_argument("--w_thr_step", type=float, default=0.01, help="w_thr adım değeri")
    parser.add_argument("--w_thr_active", action="store_true", help="w_thr parametresi için komut üret")
    
    # d_thr parametreleri
    parser.add_argument("--d_thr_start", type=float, default=0.8, help="d_thr başlangıç değeri")
    parser.add_argument("--d_thr_end", type=float, default=0.9, help="d_thr bitiş değeri")
    parser.add_argument("--d_thr_step", type=float, default=0.01, help="d_thr adım değeri")
    parser.add_argument("--d_thr_active", action="store_true", help="d_thr parametresi için komut üret")
    
    # s parametreleri
    parser.add_argument("--s_start", type=float, default=3.0, help="s başlangıç değeri")
    parser.add_argument("--s_end", type=float, default=3.0, help="s bitiş değeri")
    parser.add_argument("--s_step", type=float, default=0.1, help="s adım değeri (0 ise tek değer)")
    parser.add_argument("--s_active", action="store_true", help="s parametresi için komut üret")
    


#--------------------------------------------------------------------------------------------------------------------------------------



    # entropy_norm_min parametreleri
    parser.add_argument("--entropy_norm_min_start", type=float, default=-1.0, help="entropy_norm_min başlangıç değeri")
    parser.add_argument("--entropy_norm_min_end", type=float, default=2.0, help="entropy_norm_min bitiş değeri")
    parser.add_argument("--entropy_norm_min_step", type=float, default=0.1, help="entropy_norm_min adım değeri (0 ise tek değer)")
    parser.add_argument("--entropy_norm_min_active", action="store_false", help="entropy_norm_min parametresi için komut üret")
    
    # pose_chi2_norm_min parametreleri-1.0, help="pose_chi2_norm_min başlangıç değeri")
    parser.add_argument("--pose_chi2_norm_min_start", type=float, default=-1.0, help="entropy_norm_start başlangıç değeri")
    parser.add_argument("--pose_chi2_norm_min_end", type=float, default=2.0, help="pose_chi2_norm_min bitiş değeri")
    parser.add_argument("--pose_chi2_norm_min_step", type=float, default=0.1, help="pose_chi2_norm_min adım değeri (0 ise tek değer)")
    parser.add_argument("--pose_chi2_norm_min_active", action="store_false", help="pose_chi2_norm_min parametresi için komut üret")
    
    # culled_norm_min parametreleri
    parser.add_argument("--culled_norm_min_start", type=float, default=-1.0, help="culled_norm_min başlangıç değeri")
    parser.add_argument("--culled_norm_min_end", type=float, default=2.0, help="culled_norm_min bitiş değeri")
    parser.add_argument("--culled_norm_min_step", type=float, default=0.1, help="culled_norm_min adım değeri (0 ise tek değer)")
    parser.add_argument("--culled_norm_min_active", action="store_false", help="culled_norm_min parametresi için komut üret")
    
    parser.add_argument("--output_file", type=str, default="parameter_commands.txt", help="Çıktı dosyasının adı")
    
    args = parser.parse_args()
    
    commands = []
    
    # Sadece aktif olan parametreler için, ilgili değer aralığındaki her bir değer için ayrı komut satırı oluşturuluyor.
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
    
    # Oluşturulan her komut satırını çıktı dosyasına yazıyoruz.
    with open(args.output_file, "w") as f:
        for command in commands:
            f.write(command + "\n")
    
    print(f"{len(commands)} komut '{args.output_file}' dosyasına yazıldı.")

if __name__ == "__main__":
    main()