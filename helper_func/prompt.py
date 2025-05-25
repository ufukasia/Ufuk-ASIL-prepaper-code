import numpy as np
import itertools
import argparse

def make_range(start, end, step):
    # Eğer step 0 ise tek değerlik array döndür
    if step == 0:
        return np.array([start])
    else:
        # Küçük tolerans ekleyerek dahil olmasını sağlıyoruz
        return np.arange(start, end + step / 2, step)

def main():
    parser = argparse.ArgumentParser(description="eskf-vio için batch parametre kombinasyonları üreticisi.")
    
    # Alpha parametreleri
    parser.add_argument("--alpha_start", type=float, default=0.1, help="Alpha başlangıç değeri")
    parser.add_argument("--alpha_end", type=float, default=20, help="Alpha bitiş değeri")
    parser.add_argument("--alpha_step", type=float, default=0.1, help="Alpha adım değeri")
    parser.add_argument("--alpha_single", action="store_true", help="Alpha için tek değer üretimi (range kapalı)")
    
    # Beta parametreleri
    parser.add_argument("--beta_start", type=float, default=0.7, help="Beta başlangıç değeri")
    parser.add_argument("--beta_end", type=float, default=1.5, help="Beta bitiş değeri")
    parser.add_argument("--beta_step", type=float, default=0.1, help="Beta adım değeri")
    parser.add_argument("--beta_single", action="store_false", help="Beta için tek değer üretimi (range kapalı)")
    
    # Gamma parametreleri
    parser.add_argument("--gamma_start", type=float, default=1.0, help="Gamma başlangıç değeri")
    parser.add_argument("--gamma_end", type=float, default=1.0, help="Gamma bitiş değeri")
    parser.add_argument("--gamma_step", type=float, default=0.0, help="Gamma adım değeri (0 ise tek değer)")
    parser.add_argument("--gamma_single", action="store_false", help="Gamma için tek değer üretimi (range kapalı)")
    
    # w_thr parametreleri
    parser.add_argument("--w_thr_start", type=float, default=0.2, help="w_thr başlangıç değeri")
    parser.add_argument("--w_thr_end", type=float, default=0.3, help="w_thr bitiş değeri")
    parser.add_argument("--w_thr_step", type=float, default=0.01, help="w_thr adım değeri")
    parser.add_argument("--w_thr_single", action="store_false", help="w_thr için tek değer üretimi (range kapalı)")
    
    # d_thr parametreleri
    parser.add_argument("--d_thr_start", type=float, default=0.8, help="d_thr başlangıç değeri")
    parser.add_argument("--d_thr_end", type=float, default=0.9, help="d_thr bitiş değeri")
    parser.add_argument("--d_thr_step", type=float, default=0.01, help="d_thr adım değeri")
    parser.add_argument("--d_thr_single", action="store_false", help="d_thr için tek değer üretimi (range kapalı)")
    
    # Epsilon parametreleri
    parser.add_argument("--epsilon_start", type=float, default=0.5, help="Epsilon başlangıç değeri")
    parser.add_argument("--epsilon_end", type=float, default=1.5, help="Epsilon bitiş değeri")
    parser.add_argument("--epsilon_step", type=float, default=0.1, help="Epsilon adım değeri")
    parser.add_argument("--epsilon_single", action="store_false", help="Epsilon için tek değer üretimi (range kapalı)")
    
    # Zeta parametreleri
    parser.add_argument("--zeta_start", type=float, default=1.0, help="Zeta başlangıç değeri")
    parser.add_argument("--zeta_end", type=float, default=1.0, help="Zeta bitiş değeri")
    parser.add_argument("--zeta_step", type=float, default=0.0, help="Zeta adım değeri (0 ise tek değer)")
    parser.add_argument("--zeta_single", action="store_false", help="Zeta için tek değer üretimi (range kapalı)")
    
    parser.add_argument("--output_file", type=str, default="parameter_combinations.txt", help="Çıktı dosyasının adı")
    
    # Aktivasyon fonksiyonları listesi (sabit)
    activation_functions = [
        "relu",
        #"quadratic_unit_step",
        #"cubic_unit_step",
        #"quartic_unit_step",
        #"double_exponential_sigmoid",
        #"triple_exponential_sigmoid",
        #"quadruple_exponential_sigmoid",
        #"step"
    ]
    
    args = parser.parse_args()
    
    # Belirtilen aralıklarda değerler oluşturuluyor. Eğer ilgili single flag aktifse yalnızca başlangıç değeri kullanılır.
    alpha_values   = np.array([args.alpha_start]) if args.alpha_single else make_range(args.alpha_start, args.alpha_end, args.alpha_step)
    beta_values    = np.array([args.beta_start]) if args.beta_single else make_range(args.beta_start, args.beta_end, args.beta_step)
    gamma_values   = np.array([args.gamma_start]) if args.gamma_single else make_range(args.gamma_start, args.gamma_end, args.gamma_step)
    w_thr_values   = np.array([args.w_thr_start]) if args.w_thr_single else make_range(args.w_thr_start, args.w_thr_end, args.w_thr_step)
    d_thr_values   = np.array([args.d_thr_start]) if args.d_thr_single else make_range(args.d_thr_start, args.d_thr_end, args.d_thr_step)
    epsilon_values = np.array([args.epsilon_start]) if args.epsilon_single else make_range(args.epsilon_start, args.epsilon_end, args.epsilon_step)
    zeta_values    = np.array([args.zeta_start]) if args.zeta_single else make_range(args.zeta_start, args.zeta_end, args.zeta_step)
    
    # Tüm kombinasyonları oluşturmak için Cartesian product kullanılıyor
    combinations = list(itertools.product(
        alpha_values, beta_values, gamma_values,
        w_thr_values, d_thr_values,
        epsilon_values, zeta_values,
        activation_functions
    ))
    
    with open(args.output_file, "w") as f:
        for comb in combinations:
            alpha, beta, gamma, w_thr, d_thr, epsilon, zeta, activation = comb
            # Her satırın başında "eskf-vio batch.py" ifadesi olacak şekilde formatlıyoruz
            line = (
                f'python "main_module.py" --alpha {alpha:.1f} --beta {beta:.1f} --gamma {gamma:.1f} '
                f'--w_thr {w_thr:.2f} --d_thr {d_thr:.2f} '
                f'--epsilon {epsilon:.1f} --zeta {zeta:.1f} --activation {activation}\n'
            )
            f.write(line)
    
    print(f"{len(combinations)} kombinasyon '{args.output_file}' dosyasına yazıldı.")

if __name__ == "__main__":
    main()