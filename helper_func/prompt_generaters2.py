import numpy as np
import itertools
import argparse

def make_range(start, end, step):
    # If step is 0, return an array with a single value.
    if step == 0:
        return np.array([start])
    else:
        # Add a small tolerance to ensure the end point is included.
        return np.arange(start, end + step / 2, step)

def main():
    parser = argparse.ArgumentParser(description="Batch parameter combination generator for eskf-vio.")
    
    # Alpha parameters
    parser.add_argument("--alpha_start", type=float, default=0.1, help="Alpha start value")
    parser.add_argument("--alpha_end", type=float, default=20, help="Alpha end value")
    parser.add_argument("--alpha_step", type=float, default=0.1, help="Alpha step value")
    parser.add_argument("--alpha_single", action="store_true", help="Generate single value for Alpha (range disabled)")
    
    # Beta parameters
    parser.add_argument("--beta_start", type=float, default=0.7, help="Beta start value")
    parser.add_argument("--beta_end", type=float, default=1.5, help="Beta end value")
    parser.add_argument("--beta_step", type=float, default=0.1, help="Beta step value")
    parser.add_argument("--beta_single", action="store_false", help="Generate single value for Beta (range disabled)")
    
    # Gamma parameters
    parser.add_argument("--gamma_start", type=float, default=1.0, help="Gamma start value")
    parser.add_argument("--gamma_end", type=float, default=1.0, help="Gamma end value")
    parser.add_argument("--gamma_step", type=float, default=0.0, help="Gamma step value (0 for single value)")
    parser.add_argument("--gamma_single", action="store_false", help="Generate single value for Gamma (range disabled)")
    
    # w_thr parameters
    parser.add_argument("--w_thr_start", type=float, default=0.2, help="w_thr start value")
    parser.add_argument("--w_thr_end", type=float, default=0.3, help="w_thr end value")
    parser.add_argument("--w_thr_step", type=float, default=0.01, help="w_thr step value")
    parser.add_argument("--w_thr_single", action="store_false", help="Generate single value for w_thr (range disabled)")
    
    # d_thr parameters
    parser.add_argument("--d_thr_start", type=float, default=0.8, help="d_thr start value")
    parser.add_argument("--d_thr_end", type=float, default=0.9, help="d_thr end value")
    parser.add_argument("--d_thr_step", type=float, default=0.01, help="d_thr step value")
    parser.add_argument("--d_thr_single", action="store_false", help="Generate single value for d_thr (range disabled)")
    
    # Epsilon parameters
    parser.add_argument("--epsilon_start", type=float, default=0.5, help="Epsilon start value")
    parser.add_argument("--epsilon_end", type=float, default=1.5, help="Epsilon end value")
    parser.add_argument("--epsilon_step", type=float, default=0.1, help="Epsilon step value")
    parser.add_argument("--epsilon_single", action="store_false", help="Generate single value for Epsilon (range disabled)")
    
    # Zeta parameters
    parser.add_argument("--zeta_start", type=float, default=1.0, help="Zeta start value")
    parser.add_argument("--zeta_end", type=float, default=1.0, help="Zeta end value")
    parser.add_argument("--zeta_step", type=float, default=0.0, help="Zeta step value (0 for single value)")
    parser.add_argument("--zeta_single", action="store_false", help="Generate single value for Zeta (range disabled)")
    
    parser.add_argument("--output_file", type=str, default="parameter_combinations.txt", help="Name of the output file")
    
    # List of activation functions (fixed)
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
    
    # Values are generated in the specified ranges. If the respective single flag is active, only the start value is used.
    alpha_values   = np.array([args.alpha_start]) if args.alpha_single else make_range(args.alpha_start, args.alpha_end, args.alpha_step)
    beta_values    = np.array([args.beta_start]) if args.beta_single else make_range(args.beta_start, args.beta_end, args.beta_step)
    gamma_values   = np.array([args.gamma_start]) if args.gamma_single else make_range(args.gamma_start, args.gamma_end, args.gamma_step)
    w_thr_values   = np.array([args.w_thr_start]) if args.w_thr_single else make_range(args.w_thr_start, args.w_thr_end, args.w_thr_step)
    d_thr_values   = np.array([args.d_thr_start]) if args.d_thr_single else make_range(args.d_thr_start, args.d_thr_end, args.d_thr_step)
    epsilon_values = np.array([args.epsilon_start]) if args.epsilon_single else make_range(args.epsilon_start, args.epsilon_end, args.epsilon_step)
    zeta_values    = np.array([args.zeta_start]) if args.zeta_single else make_range(args.zeta_start, args.zeta_end, args.zeta_step)
    
    # Cartesian product is used to generate all combinations
    combinations = list(itertools.product(
        alpha_values, beta_values, gamma_values,
        w_thr_values, d_thr_values,
        epsilon_values, zeta_values,
        activation_functions
    ))
    
    with open(args.output_file, "w") as f:
        for comb in combinations:
            alpha, beta, gamma, w_thr, d_thr, epsilon, zeta, activation = comb
            # Formatting each line to start with "python main_module.py"
            line = (
                f'python "main_module.py" --alpha {alpha:.1f} --beta {beta:.1f} --gamma {gamma:.1f} '
                f'--w_thr {w_thr:.2f} --d_thr {d_thr:.2f} '
                f'--epsilon {epsilon:.1f} --zeta {zeta:.1f} --activation {activation}\n'
            )
            f.write(line)
    
    print(f"{len(combinations)} combinations were written to '{args.output_file}'.")

if __name__ == "__main__":
    main()