import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import sys

# --- Configuration and Setup ---
pio.renderers.default = "browser"

# Valid parameters for analysis
valid_params = [
    "Entropy_Norm_Min", "Pose_Chi2_Norm_Min", "Culled_Norm_Min",
    "Alpha_v", "Epsilon_v", "Zeta_H", "Zeta_L"
]

# Parameter Selection for Plot 1:
initial_selected_param_plot1 = valid_params[0]
if len(sys.argv) > 1:
    if sys.argv[1] in valid_params:
        initial_selected_param_plot1 = sys.argv[1]
    else:
        print(f"Warning: Command-line argument '{sys.argv[1]}' is not a valid parameter for Plot 1. Using '{initial_selected_param_plot1}'.")
        print(f"Valid parameters: {', '.join(valid_params)}")
else:
    print(f"No command-line argument provided for Plot 1. Using default initial parameter: '{initial_selected_param_plot1}'")

print(f"Initial selected parameter for Plot 1 analysis: {initial_selected_param_plot1}")

# Load data
file_path = "results.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Results CSV file not found at '{file_path}'.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

missing_params_cols = [p for p in valid_params if p not in df.columns]
if missing_params_cols:
    print(f"Error: Required parameter columns missing in {file_path}: {', '.join(missing_params_cols)}")
    sys.exit(1)

try:
    df["RMSE_Position"] = df["RMSE_Position"].apply(
        lambda x: np.mean(np.fromstring(str(x).strip("[]"), sep=" ")) if isinstance(x, str) and x.strip("[]") else np.nan
    )
    df.dropna(subset=["RMSE_Position"], inplace=True)
    if df.empty:
        print("Error: No valid RMSE_Position data after processing.")
        sys.exit(1)
except Exception as e:
    print(f"Error processing RMSE_Position: {e}")
    sys.exit(1)

all_sequences = sorted(df["Sequence"].unique())
if not list(all_sequences):
    print("Error: No sequences found.")
    sys.exit(1)

colors = [
    "blue", "red", "green", "orange", "purple", "brown",
    "pink", "gray", "cyan", "magenta", "yellow", "black", "lime"
]
# --- End Configuration and Setup ---

# --- Plot 1: Mean RMSE vs. Selected Parameter (with Radio Buttons for Parameter) ---
print(f"\nGenerating Plot 1: All sequences vs. selected parameter '{initial_selected_param_plot1}'.")
fig1 = go.Figure()

for i, sequence in enumerate(all_sequences):
    df_seq = df[df["Sequence"] == sequence].copy()
    chosen_color = colors[i % len(colors)]
    df_seq[initial_selected_param_plot1] = pd.to_numeric(df_seq[initial_selected_param_plot1], errors='coerce')
    df_seq_cleaned_for_initial_param = df_seq.dropna(subset=[initial_selected_param_plot1, "RMSE_Position"])

    rmse_val_false = np.nan
    if not df_seq_cleaned_for_initial_param.empty:
        df_false = df_seq_cleaned_for_initial_param[df_seq_cleaned_for_initial_param["Adaptive"] == False]
        if not df_false.empty: rmse_val_false = df_false["RMSE_Position"].mean()

    x_line_initial = []
    if not np.isnan(rmse_val_false) and not df_seq_cleaned_for_initial_param.empty:
        param_vals_true_initial = df_seq_cleaned_for_initial_param[df_seq_cleaned_for_initial_param["Adaptive"] == True][initial_selected_param_plot1].unique()
        param_min_overall = df_seq_cleaned_for_initial_param[initial_selected_param_plot1].min()
        param_max_overall = df_seq_cleaned_for_initial_param[initial_selected_param_plot1].max()
        min_val_init, max_val_init = np.nan, np.nan
        if len(param_vals_true_initial) > 0 and not pd.isna(param_vals_true_initial).all():
            min_val_init = np.nanmin(param_vals_true_initial)
            max_val_init = np.nanmax(param_vals_true_initial)
        if np.isnan(min_val_init) or np.isnan(max_val_init):
            if not np.isnan(param_min_overall) and not np.isnan(param_max_overall):
                min_val_init = param_min_overall
                max_val_init = param_max_overall
        if not np.isnan(min_val_init) and not np.isnan(max_val_init):
            if min_val_init == max_val_init: x_line_initial = [min_val_init - 0.00005, min_val_init + 0.00005]
            else: x_line_initial = np.linspace(min_val_init, max_val_init, 50)

    fig1.add_trace(go.Scatter(
        x=x_line_initial, y=[rmse_val_false] * len(x_line_initial) if not np.isnan(rmse_val_false) else [],
        mode='lines', name=f"{sequence} (False)", legendgroup=f"{sequence}",
        showlegend= True if i == 0 else False, line=dict(color=chosen_color, dash='dash'),
        hovertemplate=(f'{initial_selected_param_plot1}: %{{x}}<br>RMSE: {rmse_val_false:.4f}<br>Adaptive: False<br>Sequence: {sequence}<extra></extra>'
                       if not np.isnan(rmse_val_false) else f'No Data for {sequence} (False)'),
        visible=True
    ))

    x_markers_initial, y_markers_initial = [], []
    if not df_seq_cleaned_for_initial_param.empty:
        df_true_initial = df_seq_cleaned_for_initial_param[df_seq_cleaned_for_initial_param["Adaptive"] == True]
        if not df_true_initial.empty:
            param_rmse_mean_true_initial = df_true_initial.groupby(initial_selected_param_plot1)["RMSE_Position"].mean().reset_index()
            x_markers_initial = param_rmse_mean_true_initial[initial_selected_param_plot1]
            y_markers_initial = param_rmse_mean_true_initial["RMSE_Position"]

    fig1.add_trace(go.Scatter(
        x=x_markers_initial, y=y_markers_initial, mode='lines+markers', name=f"{sequence} (True)",
        legendgroup=f"{sequence}", showlegend=True, line=dict(color=chosen_color), marker=dict(color=chosen_color),
        hovertemplate=(f'{initial_selected_param_plot1}: %{{x}}<br>RMSE: %{{y:.4f}}<br>Adaptive: True<br>Sequence: {sequence}<extra></extra>'),
        visible=True
    ))

updatemenus_buttons_plot1 = []
for param_to_show in valid_params:
    trace_data_updates = {'x': [], 'y': [], 'hovertemplate': []}
    annotations_for_button = []
    for i_seq_btn, sequence_btn in enumerate(all_sequences):
        df_seq_btn = df[df["Sequence"] == sequence_btn].copy()
        df_seq_btn[param_to_show] = pd.to_numeric(df_seq_btn[param_to_show], errors='coerce')
        df_seq_cleaned_btn = df_seq_btn.dropna(subset=[param_to_show, "RMSE_Position"])
        rmse_val_false_btn, x_line_btn = np.nan, []
        if not df_seq_cleaned_btn.empty:
            df_false_btn = df_seq_cleaned_btn[df_seq_cleaned_btn["Adaptive"] == False]
            if not df_false_btn.empty: rmse_val_false_btn = df_false_btn["RMSE_Position"].mean()
            if not np.isnan(rmse_val_false_btn):
                param_vals_true_btn_cp = df_seq_cleaned_btn[df_seq_cleaned_btn["Adaptive"] == True][param_to_show].unique()
                param_min_btn_overall_cp = df_seq_cleaned_btn[param_to_show].min()
                param_max_btn_overall_cp = df_seq_cleaned_btn[param_to_show].max()
                min_val_btn_cp, max_val_btn_cp = np.nan, np.nan
                if len(param_vals_true_btn_cp) > 0 and not pd.isna(param_vals_true_btn_cp).all():
                    min_val_btn_cp, max_val_btn_cp = np.nanmin(param_vals_true_btn_cp), np.nanmax(param_vals_true_btn_cp)
                if np.isnan(min_val_btn_cp) or np.isnan(max_val_btn_cp):
                    if not np.isnan(param_min_btn_overall_cp) and not np.isnan(param_max_btn_overall_cp):
                        min_val_btn_cp, max_val_btn_cp = param_min_btn_overall_cp, param_max_btn_overall_cp
                if not np.isnan(min_val_btn_cp) and not np.isnan(max_val_btn_cp):
                    if min_val_btn_cp == max_val_btn_cp: x_line_btn = [min_val_btn_cp - 0.00005, min_val_btn_cp + 0.00005]
                    else: x_line_btn = np.linspace(min_val_btn_cp, max_val_btn_cp, 50)
        trace_data_updates['x'].append(x_line_btn)
        trace_data_updates['y'].append([rmse_val_false_btn] * len(x_line_btn) if not np.isnan(rmse_val_false_btn) else [])
        trace_data_updates['hovertemplate'].append(
            f'{param_to_show}: %{{x}}<br>RMSE: {rmse_val_false_btn:.4f}<br>Adaptive: False<br>Sequence: {sequence_btn}<extra></extra>'
            if not np.isnan(rmse_val_false_btn) else f'No Data for {sequence_btn} (False)')
        x_markers_btn, y_markers_btn = [], []
        if not df_seq_cleaned_btn.empty:
            df_true_btn = df_seq_cleaned_btn[df_seq_cleaned_btn["Adaptive"] == True]
            if not df_true_btn.empty:
                param_rmse_mean_true_btn = df_true_btn.groupby(param_to_show)["RMSE_Position"].mean().reset_index()
                x_markers_btn, y_markers_btn = param_rmse_mean_true_btn[param_to_show], param_rmse_mean_true_btn["RMSE_Position"]
        trace_data_updates['x'].append(x_markers_btn)
        trace_data_updates['y'].append(y_markers_btn)
        trace_data_updates['hovertemplate'].append(
            f'{param_to_show}: %{{x}}<br>RMSE: %{{y:.4f}}<br>Adaptive: True<br>Sequence: {sequence_btn}<extra></extra>')
    current_annotation_args = {}
    df_all_true_plot1 = df[df["Adaptive"] == True].copy()
    if not df_all_true_plot1.empty:
        df_all_true_plot1[param_to_show] = pd.to_numeric(df_all_true_plot1[param_to_show], errors='coerce')
        df_all_true_cleaned_plot1 = df_all_true_plot1.dropna(subset=[param_to_show, "RMSE_Position"])
        if not df_all_true_cleaned_plot1.empty:
            overall_stats_plot1 = df_all_true_cleaned_plot1.groupby(param_to_show).agg(mean_rmse=("RMSE_Position", "mean"), seq_count=("Sequence", "nunique")).reset_index()
            complete_params_df_plot1 = overall_stats_plot1[overall_stats_plot1["seq_count"] == len(all_sequences)]
            if not complete_params_df_plot1.empty:
                min_rmse_row_plot1 = complete_params_df_plot1.loc[complete_params_df_plot1["mean_rmse"].idxmin()]
                min_param_val_plot1, min_rmse_val_plot1 = min_rmse_row_plot1[param_to_show], min_rmse_row_plot1["mean_rmse"]
                current_annotation_args = dict(x=min_param_val_plot1, y=min_rmse_val_plot1,
                    text=f"Min Avg RMSE ({min_rmse_val_plot1:.4f})<br>@ {param_to_show}={min_param_val_plot1:.3f}",
                    showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(255,255,224,0.8)", bordercolor="black", borderwidth=1)
                annotations_for_button.append(current_annotation_args)
            else: print(f"Plot 1 Uyarı ({param_to_show}): Tüm sekanslar için ortak bir '{param_to_show}' değeri bulunamadı. Anotasyon eklenemedi.")
        else: print(f"Plot 1 Uyarı ({param_to_show}): Adaptive=True için '{param_to_show}' ile ilgili geçerli veri bulunamadı.")
    updatemenus_buttons_plot1.append(dict(label=param_to_show, method="update",
        args=[trace_data_updates, {'title.text': f"Mean RMSE_Position vs. {param_to_show} (All Sequences)",
               'xaxis.title.text': f"{param_to_show} Value", 'annotations': annotations_for_button}]))

initial_annotation_args_plot1 = {}
df_all_true_init_plot1 = df[df["Adaptive"] == True].copy()
if not df_all_true_init_plot1.empty:
    df_all_true_init_plot1[initial_selected_param_plot1] = pd.to_numeric(df_all_true_init_plot1[initial_selected_param_plot1], errors='coerce')
    df_all_true_init_cleaned_plot1 = df_all_true_init_plot1.dropna(subset=[initial_selected_param_plot1, "RMSE_Position"])
    if not df_all_true_init_cleaned_plot1.empty:
        overall_stats_init_plot1 = df_all_true_init_cleaned_plot1.groupby(initial_selected_param_plot1).agg(mean_rmse=("RMSE_Position", "mean"), seq_count=("Sequence", "nunique")).reset_index()
        complete_params_df_init_plot1 = overall_stats_init_plot1[overall_stats_init_plot1["seq_count"] == len(all_sequences)]
        if not complete_params_df_init_plot1.empty:
            min_rmse_row_init_plot1 = complete_params_df_init_plot1.loc[complete_params_df_init_plot1["mean_rmse"].idxmin()]
            initial_annotation_args_plot1 = dict(x=min_rmse_row_init_plot1[initial_selected_param_plot1], y=min_rmse_row_init_plot1["mean_rmse"],
                text=f"Min Avg RMSE ({min_rmse_row_init_plot1['mean_rmse']:.4f})<br>@ {initial_selected_param_plot1}={min_rmse_row_init_plot1[initial_selected_param_plot1]:.3f}",
                showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(255,255,224,0.8)", bordercolor="black", borderwidth=1)
        else: print(f"Plot 1 Başlangıç Uyarı ({initial_selected_param_plot1}): Tüm sekanslar için ortak '{initial_selected_param_plot1}' değeri bulunamadı. Anotasyon yok.")
    else: print(f"Plot 1 Başlangıç Uyarı ({initial_selected_param_plot1}): Adaptive=True için '{initial_selected_param_plot1}' ile ilgili veri yok.")

fig1.update_layout(
    updatemenus=[dict(type="buttons", direction="right", active=valid_params.index(initial_selected_param_plot1),
        x=0.5, xanchor="center", y=1.15, yanchor="top", showactive=True, buttons=updatemenus_buttons_plot1)],
    title=f"Mean RMSE_Position vs. {initial_selected_param_plot1} (All Sequences)",
    xaxis_title=f"{initial_selected_param_plot1} Value", yaxis_title="Mean RMSE_Position",
    legend_title_text="Sequence (Adaptive)", hovermode="x unified",
    annotations=[initial_annotation_args_plot1] if initial_annotation_args_plot1 else []
)
fig1.show()


# --- Plot 2: Subplots for SELECTED Sequence - RMSE vs. All Valid Parameters (with Radio Buttons for Sequence) ---
print(f"\nGenerating Plot 2: Subplots for a selected sequence vs. all {len(valid_params)} parameters (Grid Layout).")

if not list(all_sequences):
    print("Cannot generate Plot 2: No sequences available.")
else:
    # Plot 2 için subplot düzeni: 4 satır, 2 sütun (7 parametre için)
    n_params = len(valid_params)
    n_cols_plot2 = 2
    n_rows_plot2 = (n_params + n_cols_plot2 - 1) // n_cols_plot2 # Tavana yuvarlama (örn: 7 parametre, 2 sütun -> 4 satır)

    # Subplot başlıklarını oluştururken dikkatli olmalıyız, çünkü grid tam dolmayabilir.
    # subplot_titles listesi, make_subplots'a verilirken linear bir şekilde (soldan sağa, yukarıdan aşağıya) beklenir.
    subplot_titles_plot2 = [f"RMSE vs {param_name}" for param_name in valid_params]
    # Eğer grid tam dolmuyorsa, boş hücreler için başlık eklememek gerekir.
    # Ancak make_subplots, subplot_titles listesinin uzunluğunun row*col sayısına eşit olmasını bekler.
    # Bu yüzden, eğer 7 parametre varsa ve 4x2 grid kullanıyorsak, 8 başlık gerekir. Sonuncusu boş olabilir.
    # Ya da, subplot_titles'ı sadece var olan parametreler için oluşturup,
    # add_trace sırasında doğru row/col hesaplaması yaparız. make_subplots tüm hücrelere başlık atamaya çalışır.
    # En temizi, tüm hücrelere bir başlık atamak ve boş olanın başlığını sonradan silmek veya boş bırakmak.
    # Veya, subplot_titles'ı parametre sayısı kadar verip, boş hücrelerin başlığı olmayacak.
    # Plotly genellikle bunu tolere eder.

    fig2 = make_subplots(
        rows=n_rows_plot2, cols=n_cols_plot2,
        subplot_titles=subplot_titles_plot2, # Sadece mevcut parametreler için başlık
        vertical_spacing=0.1, # Dikey aralık
        horizontal_spacing=0.08 # Yatay aralık
    )

    initial_sequence_plot2_idx = 0
    preferred_initial_sequence = "MH01"
    if preferred_initial_sequence in all_sequences:
        try: initial_sequence_plot2_idx = list(all_sequences).index(preferred_initial_sequence)
        except ValueError: pass
    initial_sequence_plot2 = all_sequences[initial_sequence_plot2_idx]
    print(f"Plot 2: Initial sequence set to '{initial_sequence_plot2}'.")

    # Her bir parametre için bir subplot hücresi belirle
    for i_param, param_name_subplot in enumerate(valid_params):
        # Mevcut parametrenin subplot grid'indeki satır ve sütununu hesapla
        current_row = (i_param // n_cols_plot2) + 1
        current_col = (i_param % n_cols_plot2) + 1

        # Her bir sekans için trace'leri ekle (başlangıçta sadece initial_sequence_plot2 için görünür)
        for i_seq_trace, seq_name_trace in enumerate(all_sequences):
            df_seq_subplot_trace = df[df["Sequence"] == seq_name_trace].copy()
            color_for_seq_trace = colors[i_seq_trace % len(colors)]

            df_seq_subplot_trace[param_name_subplot] = pd.to_numeric(df_seq_subplot_trace[param_name_subplot], errors='coerce')
            df_param_cleaned_subplot_trace = df_seq_subplot_trace.dropna(subset=[param_name_subplot, "RMSE_Position"])

            # Adaptive=True trace
            x_true_vals_subplot, y_true_vals_subplot = [], []
            if not df_param_cleaned_subplot_trace.empty:
                df_true_for_subplot = df_param_cleaned_subplot_trace[df_param_cleaned_subplot_trace["Adaptive"] == True]
                if not df_true_for_subplot.empty:
                    param_rmse_mean_true_subplot = df_true_for_subplot.groupby(param_name_subplot)["RMSE_Position"].mean().reset_index()
                    x_true_vals_subplot = param_rmse_mean_true_subplot[param_name_subplot]
                    y_true_vals_subplot = param_rmse_mean_true_subplot["RMSE_Position"]

            fig2.add_trace(
                go.Scatter(
                    x=x_true_vals_subplot, y=y_true_vals_subplot, mode='lines+markers',
                    name=f"{seq_name_trace} (True)", legendgroup=f"{seq_name_trace}",
                    showlegend= (i_param == 0), # Sadece ilk subplot parametresi için legend göster
                    line=dict(color=color_for_seq_trace), marker=dict(color=color_for_seq_trace),
                    hovertemplate=(f'{param_name_subplot}: %{{x}}<br>RMSE: %{{y:.4f}}<br>Adaptive: True<br>Sequence: {seq_name_trace}<extra></extra>'),
                    visible=(seq_name_trace == initial_sequence_plot2)
                ),
                row=current_row, col=current_col
            )

            # Adaptive=False trace
            x_false_line_subplot, y_false_val_scalar_subplot = [], np.nan
            if not df_param_cleaned_subplot_trace.empty:
                df_false_for_subplot = df_param_cleaned_subplot_trace[df_param_cleaned_subplot_trace["Adaptive"] == False]
                if not df_false_for_subplot.empty:
                    y_false_val_scalar_subplot = df_false_for_subplot["RMSE_Position"].mean()

                if not np.isnan(y_false_val_scalar_subplot):
                    param_vals_true_for_x_subplot = []
                    if 'df_true_for_subplot' in locals() and not df_true_for_subplot.empty:
                         param_vals_true_for_x_subplot = df_true_for_subplot[param_name_subplot].unique()
                    param_min_overall_subplot_trace = df_param_cleaned_subplot_trace[param_name_subplot].min()
                    param_max_overall_subplot_trace = df_param_cleaned_subplot_trace[param_name_subplot].max()
                    min_x_line_sp, max_x_line_sp = np.nan, np.nan
                    if len(param_vals_true_for_x_subplot) > 0 and not pd.isna(param_vals_true_for_x_subplot).all():
                        min_x_line_sp, max_x_line_sp = np.nanmin(param_vals_true_for_x_subplot), np.nanmax(param_vals_true_for_x_subplot)
                    if np.isnan(min_x_line_sp) or np.isnan(max_x_line_sp):
                        if not np.isnan(param_min_overall_subplot_trace) and not np.isnan(param_max_overall_subplot_trace):
                            min_x_line_sp, max_x_line_sp = param_min_overall_subplot_trace, param_max_overall_subplot_trace
                    if not np.isnan(min_x_line_sp) and not np.isnan(max_x_line_sp):
                        if min_x_line_sp == max_x_line_sp: x_false_line_subplot = [min_x_line_sp - 0.00005, min_x_line_sp + 0.00005]
                        else: x_false_line_subplot = np.linspace(min_x_line_sp, max_x_line_sp, 50)
            fig2.add_trace(
                go.Scatter(
                    x=x_false_line_subplot,
                    y=[y_false_val_scalar_subplot] * len(x_false_line_subplot) if not np.isnan(y_false_val_scalar_subplot) else [],
                    mode='lines', name=f"{seq_name_trace} (False)", legendgroup=f"{seq_name_trace}",
                    showlegend= (i_param == 0 and i_seq_trace == 0),
                    line=dict(color=color_for_seq_trace, dash='dash'),
                    hovertemplate=(f'{param_name_subplot}: %{{x}}<br>RMSE: {y_false_val_scalar_subplot:.4f}<br>Adaptive: False<br>Sequence: {seq_name_trace}<extra></extra>'
                                   if not np.isnan(y_false_val_scalar_subplot) else f'No Data for {seq_name_trace} (False)'),
                    visible=(seq_name_trace == initial_sequence_plot2)
                ),
                row=current_row, col=current_col
            )

    updatemenus_buttons_plot2 = []
    for i_seq_btn_plot2, button_seq_name_plot2 in enumerate(all_sequences):
        visibility_list_plot2 = []
        # Her parametre için (her subplot hücresi için)
        for i_p_vis in range(n_params): # Sadece parametre sayısı kadar dön
            # Her sekans için (her olası trace için)
            for current_trace_seq_name_plot2 in all_sequences:
                is_visible = (current_trace_seq_name_plot2 == button_seq_name_plot2)
                visibility_list_plot2.append(is_visible) # True trace için
                visibility_list_plot2.append(is_visible) # False trace için
        # Eğer grid tam dolmuyorsa, boş hücrelere karşılık gelen görünürlük değerleri eklememek gerekir.
        # Ancak plotly.py'da trace'ler fig.data içinde lineer bir sırada tutulur.
        # Yukarıdaki döngü, her parametre için (yani her subplot için) tüm sekans trace'lerini (True ve False) oluşturur.
        # Bu nedenle visibility_list_plot2'nin uzunluğu toplam trace sayısına eşit olmalıdır.
        # Mevcut yapı zaten bunu sağlıyor, çünkü i_param üzerinden dönerek trace'leri ekliyoruz.

        updatemenus_buttons_plot2.append(dict(
            label=button_seq_name_plot2, method="update",
            args=[{'visible': visibility_list_plot2},
                  {'title.text': f"Mean RMSE vs. Parameters for Sequence: {button_seq_name_plot2}"}]
        ))

    fig2.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", active=initial_sequence_plot2_idx,
            x=0.5, xanchor="center", y=1.03, yanchor="bottom", # Butonları başlığın hemen altına
            showactive=True, buttons=updatemenus_buttons_plot2
        )],
        height=max(700, 280 * n_rows_plot2), # Yüksekliği satır sayısına göre ayarla
        width=1200, # Genişlik sabit kalabilir veya ayarlanabilir
        title_text=f"Mean RMSE vs. Parameters for Sequence: {initial_sequence_plot2}",
        title_y=0.98, # Ana başlık konumu
        showlegend=True,
        legend_title_text="Sequence (Adaptive)",
        hovermode="x unified"
    )

    # Her subplot için X ve Y eksen başlıklarını (gerekirse) güncelle
    # Subplot başlıkları make_subplots'ta zaten ayarlandı.
    # Eksen başlıklarını (xlabel, ylabel) her subplot için ayrıca ayarlamak isteyebiliriz.
    for i_param_axis, param_name_axis in enumerate(valid_params):
        current_row_axis = (i_param_axis // n_cols_plot2) + 1
        current_col_axis = (i_param_axis % n_cols_plot2) + 1
        fig2.update_xaxes(title_text=f"{param_name_axis} Value", row=current_row_axis, col=current_col_axis)
        fig2.update_yaxes(title_text="Mean RMSE", row=current_row_axis, col=current_col_axis, matches=None, autorange=True)

    fig2.show()