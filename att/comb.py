import pandas as pd

for i in range(1, 6):
    # Dosya yollarını dinamik olarak oluşturma
    seq_num = f"{i:02d}"  # 01, 02 formatı için
    file_path_1 = f"att/MH{seq_num}.csv"
    file_path_2 = f"1024_aliked/mh{i}_ns.csv"
    output_path = f"mh{seq_num}_ns.csv"
    
    try:
        # CSV dosyalarını yükle
        df1 = pd.read_csv(file_path_1)
        df2 = pd.read_csv(file_path_2)
        
        # Birleştirme ve sıralama işlemleri
        merged_df = pd.merge(df1, df2, on="#timestamp [ns]", how="outer")
        merged_df = merged_df.sort_values(by="#timestamp [ns]")
        
        # Sonucu kaydet
        merged_df.to_csv(output_path, index=False)
        print(f"{file_path_1} ve {file_path_2} başarıyla birleştirildi -> {output_path}")
        
    except Exception as e:
        print(f"Hata oluştu (MH{seq_num}): {str(e)}")