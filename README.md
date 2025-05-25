Harika bir çalışma! Hem makaleniz hem de kodunuz oldukça detaylı. GitHub için kapsamlı bir README dosyası oluşturalım. Markdown formatında olacak ve belirttiğiniz gibi görseller için yer tutucular içerecektir. Bu görselleri projenizin `assets` veya `docs/images` gibi bir klasörüne ekleyip README dosyasındaki yolları güncellemeniz gerekecek.

```markdown
# Görsel-Ataletsel Odometri İçin Uyarlanabilir Kovaryans ve Kuaterniyon Odaklı Hibrit Hata Durumlu EKF/UKF Yaklaşımı

Bu proje, İnsansız Hava Araçları (İHA) gibi otonom platformların zorlu ve dinamik ortamlarda konum kestirim performansını artırmayı hedefleyen, uyarlanabilir kovaryans güncelleme mekanizmasına sahip yenilikçi bir hibrit Görsel-Ataletsel Odometri (VIO) yaklaşımını sunmaktadır. Sistem, gevşek bağlı (loosely-coupled) bir sensör füzyon mimarisi üzerine kuruludur.

**Makale:** [Görsel-Ataletsel Odometri İçin Uyarlanabilir Kovaryans ve Kuaterniyon Odaklı Hibrit Hata Durumlu EKF/UKF Yaklaşımı](httpsTBD_Link_To_Your_Paper_Here_If_Published_Else_Remove)
**Kod:** [Ufuk-ASIL-prepaper-code](https://github.com/ufukasia/Ufuk-ASIL-prepaper-code.git) (Bu link makalede belirtilen linktir, gerekirse güncelleyiniz)

## 📜 Genel Bakış

Otonom sistemlerin, özellikle GNSS sinyallerinin zayıf veya erişilemez olduğu ortamlarda güvenilir bir şekilde seyrüsefer yapabilmesi, VIO teknolojilerinin temel hedeflerindendir. Bu çalışma, bu hedefe ulaşmak için aşağıdaki temel yenilikleri sunmaktadır:

1.  **Hibrit Qf-ES-EKF/UKF Filtre Mimarisi:** Ataletsel Ölçüm Birimi (IMU) verilerini işlemek için Kuaterniyon Odaklı Hata Durumlu Genişletilmiş Kalman Filtresi/Yönelim İçin Ölçeklenmiş Kokusuz Kalman Filtresi (Qf-ES-EKF/UKF) adı verilen özgün bir hibrit filtre mimarisi geliştirilmiştir. Bu mimari, yönelim kestiriminde SUKF'nin doğrusal olmayan sistemlerdeki üstün modelleme yeteneğini, diğer durum değişkenlerinin (konum, hız, sapmalar) kestiriminde ise ESKF'nin hesaplama verimliliğini birleştirir.
2.  **CASEF ile Dinamik ve Adaptif Sensör Füzyonu:** Görsel odometri ölçümlerinin güvenilirliği; görüntü entropisi, yoğunluk değişimi, hareket bulanıklığı ve çıkarım kalitesi (örneğin, poz optimizasyonundaki ki-kare hatası) gibi metriklere dayalı olarak dinamik bir sensör güven skoru üzerinden değerlendirilir. Bu skor, özgün **Kırpılmış Uyarlanabilir Doygunluk Üstel Fonksiyonu (CASEF)** kullanılarak ölçüm gürültüsü kovaryansının adaptif bir şekilde ayarlanmasında kullanılır.
3.  **Yüksek Dinamikli Ortamlarda Kanıtlanmış Performans:** Önerilen sistemin sağlamlığı ve doğruluğu, EuRoC MAV veri seti üzerinde, özellikle zorlu senaryolarda (ani yön değişiklikleri, yüksek ivmeli manevralar, zorlu aydınlatma koşulları) kapsamlı olarak doğrulanmıştır. Konum kestiriminde %40’a varan, rotasyon kestiriminde ise ESKF tabanlı yöntemlere kıyasla %60’a kadar iyileşme gözlemlenmiştir.

## ⚙️ Sistem Mimarisi

Önerilen VIO sisteminin genel mimarisi ve temel bileşenleri arasındaki etkileşim aşağıdaki blok diyagramında gösterilmiştir:

![Sistem Blok Diyagramı](assets/diagram.png)
*(Not: `assets/diagram.png` yolunu kendi sistem diyagramı resminizin yolu ile güncelleyiniz. Makalenizdeki Şekil 1'i kullanabilirsiniz.)*

Sistem temel olarak şu modüllerden oluşur:
1.  **Ataletsel Navigasyon (Qf-ES-EKF/UKF):** IMU ölçümlerini (ivme ve açısal hız) işleyerek durum yayılımını gerçekleştirir.
2.  **Görsel Odometri Modülü:** Stereo kamera görüntülerinden pozisyon ve hız ölçümlerini çıkarır. (Bu çalışmada PySLAM tabanlı, ALIKED özellik çıkarıcı ve LightGlue eşleştirici kullanan bir VO ön-ucu varsayılmıştır.)
3.  **Görsel Veri Kalite Analizi:** Gelen görsel verilerin kalitesini (entropi, yoğunluk değişimi, hareket bulanıklığı, VO metrikleri vb.) analiz eder.
4.  **Uyarlanabilir Ölçüm Güncelleme (Sensör Füzyonu):** Kalite analizinden elde edilen güven skoruna göre görsel ölçümlerin kovaryansını CASEF fonksiyonu ile ayarlar ve filtrenin durumunu günceller.

## 🔑 Temel Özellikler ve Katkılar

*   **Gelişmiş Durum Kestirimi:** Yönelim için SUKF ve diğer durumlar için ESKF kullanan hibrit filtreleme ile yüksek doğruluk ve hesaplama verimliliği dengesi.
*   **Dinamik Sensör Güvenilirliği Analizi:** Görsel ölçümlerin kalitesini çeşitli metriklerle anlık olarak değerlendirme.
*   **Uyarlanabilir Kovaryans Ayarı:** CASEF aktivasyon fonksiyonu ile ölçüm gürültüsü kovaryansının dinamik modülasyonu, düşük kaliteli görsel verilerin etkisini azaltma ve sensörler arası durumsal geçiş sağlama.
*   **Sağlamlık:** Zorlu çevresel koşullarda (hızlı hareket, ışık değişimi, düşük doku) kararlı ve güvenilir poz tahmini.

## 🔬 Teorik Arka Plan

### 1. Durum ve Hata Temsili
Sistem, bir nominal durum ($\hat{\mathbf{x}}$) ve bu nominal durumdan küçük sapmaları ifade eden bir hata durumu ($\delta\mathbf{x}$) kullanarak durumu modeller:
```
$\hat{\mathbf{x}} = [\hat{\mathbf{q}}^T, \hat{\mathbf{v}}^T, \hat{\mathbf{p}}^T, \hat{\mathbf{b}}_{a}^T, \hat{\mathbf{b}}_{g}^T]^T \in \mathbb{R}^{16}$
$\delta\mathbf{x} = [\delta\boldsymbol{\theta}^T, \delta\mathbf{v}^T, \delta\mathbf{p}^T, \delta\mathbf{b}_{a}^T, \delta\mathbf{b}_{g}^T]^T \in \mathbb{R}^{15}$
```
burada $\hat{\mathbf{q}}$ yönelim kuaterniyonu, $\hat{\mathbf{p}}$ konum, $\hat{\mathbf{v}}$ hız, $\hat{\mathbf{b}}_{a}$ ivmeölçer sapması, $\hat{\mathbf{b}}_{g}$ jiroskop sapması ve $\delta\boldsymbol{\theta}$ yönelim hatasıdır.

### 2. Sistem Dinamikleri ve Ayrıklaştırma
Nominal durum dinamikleri standart IMU kinematiklerini takip eder. Hata durumu dinamikleri ise şu şekilde ifade edilir:
$\dot{\delta\mathbf{x}} = \mathbf{A}\delta\mathbf{x} + \mathbf{G}\mathbf{n}$
Bu sürekli zaman modeli, filtreleme adımlarında kullanılmak üzere Van Loan yöntemi ile ayrıklaştırılır.

### 3. Hibrit Qf-ES-EKF/UKF ile Durum Yayılımı
Önerilen hibrit filtre, durum yayılımında iki aşamalı bir strateji izler:
1.  **ESKF Tabanlı Ön Yayılım:** Tüm durum vektörü ve hata kovaryansı standart ESKF adımlarıyla yayılır.
2.  **Yönelim Kovaryansının SUKF ile İyileştirilmesi:** ESKF ile yayılmış olan yönelim hatası kovaryans bloğu ($\mathbf{P}_{\theta\theta}$) üzerinde SUKF tabanlı bir iyileştirme uygulanır. Bu, sadece 3 boyutlu yönelim hatası ($\delta\boldsymbol{\theta}$) için sigma noktaları üretilerek ve bu noktalar IMU dinamikleriyle yayılarak gerçekleştirilir.

Bu yaklaşım, `ErrorStateKalmanFilterVIO_Hybrid` sınıfında uygulanmıştır. `predict` metodu önce `super().predict()` (ESKF yayılımı) çağrısını yapar, ardından yönelim kovaryansını SUKF adımlarıyla ( `_sigma_points_theta_from_S` ve `_propagate_nominal` kullanarak) rafine eder.

### 4. Uyarlanabilir Ölçüm Güncelleme Mekanizması
Görsel ölçümlerin (pozisyon $\mathbf{p}_{\mathrm{vis}}$ ve hız $\mathbf{v}_{\mathrm{vis}}$) filtreye entegrasyonunda, ölçüm gürültüsü kovaryansı $\mathbf{R}_{\scriptscriptstyle\mathrm{VIS}}$ dinamik olarak ayarlanır.

*   **Görsel Veri Kalite Analizi:**
    *   **Statik Metrikler ($\theta_p$ için):** Terslenmiş Shannon entropisi ($1 - \text{entropy}_{\mathcal{N}}$), hareket bulanıklığı ($\text{blur}_{\mathcal{N}}$), poz optimizasyon ki-kare hatası ($\chi^2_{\text{pose}_\mathcal{N}}$), elenen anahtar kare sayısı ($\text{keyf}^{\text{c}}_{\mathcal{N}}$).
    *   **Dinamik Metrikler ($\theta_v$ için):** Ardışık kareler arası normalize edilmiş yoğunluk değişimi ($\Delta\text{intensity}_{\mathcal{N}}$), bulanıklık değişimi ($\Delta\text{blur}_{\mathcal{N}}$), ki-kare hatası değişimi ($\Delta\chi^2_{\text{pose}_\mathcal{N}}$), elenen anahtar kare sayısındaki değişim ($\Delta\text{keyf}^{\text{c}}_{\mathcal{N}}$).
    Bu metrikler `cov_sigma_p.py` ve `cov_sigma_v.py` dosyalarındaki `compute_adaptive_sigma_p` ve `compute_adaptive_sigma_v` fonksiyonları içinde hesaplanır.

*   **Güven Skoru ve CASEF Aktivasyon Fonksiyonu:**
    Hesaplanan normalize edilmiş metriklerin maksimumu alınarak birleştirilir ve ardından CASEF fonksiyonuna beslenir:
    $\text{CASEF}(x; s) = \frac{\exp(s \cdot \text{clip}(x, 0.0, 1.0)) - 1}{\exp(s) - 1}$
    Bu fonksiyon, $s$ parametresi ile ayarlanabilen bir doygunluk karakteristiği sunar.

    ![CASEF Aktivasyon Fonksiyonu](assets/activation_functions.png)
    *(Not: `assets/activation_functions.png` yolunu kendi aktivasyon fonksiyonu grafiğinizin yolu ile güncelleyiniz. Makalenizdeki Şekil 5'i kullanabilirsiniz.)*

    Elde edilen $\theta_p$ ve $\theta_v$ skorları, `config['w_thr']` (ağırlıklandırma eşiği) ve `config['d_thr']` (kesme eşiği) kullanılarak nihai güven değerine dönüştürülür. Bu değerler, $\sigma_p$ ve $\sigma_v$ kovaryanslarını `MIN_COV` ve `MAX_COV` aralığında ölçekler.

### 5. Görsel Odometri Ön-Ucu (Varsayılan)
Kod, harici bir görsel odometri (VO) sisteminden gelen poz ve hız ölçümlerini kullanır. Makalede PySLAM tabanlı, ALIKED özellik çıkarıcı, LightGlue eşleştirici ve SGBM derinlik hesaplama yöntemlerini kullanan bir VO mimarisi tanımlanmıştır. Bu README'deki kod, bu VO çıktılarının (`mhX_ns.csv` dosyaları) hazır olduğunu varsayar.

## 💾 Kurulum

1.  Projeyi klonlayın:
    ```bash
    git clone https://github.com/ufukasia/Ufuk-ASIL-prepaper-code.git # Veya kendi repo linkiniz
    cd Ufuk-ASIL-prepaper-code
    ```
2.  Gerekli Python kütüphanelerini kurun. Bir `requirements.txt` dosyası oluşturmanız önerilir:
    ```
    numpy
    pandas
    scipy
    opencv-python 
    # Muhtemelen matplotlib (görselleştirme için)
    ```
    Kurulum:
    ```bash
    pip install -r requirements.txt 
    # Veya manuel olarak:
    # pip install numpy pandas scipy opencv-python
    ```

## 📊 Veri Seti

Bu proje, performans değerlendirmesi için **EuRoC MAV** veri setini kullanır.
*   Veri setini [buradan](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) indirebilirsiniz.
*   İşlenmiş IMU ve yer gerçeği verileri (`imu_interp_gt/` klasörü altında) ve görsel odometri çıktıları (`VO/vo_pred_super_best/` klasörü altında) aşağıdaki gibi bir dosya yapısında bulunmalıdır:

    ```
    .
    ├── main.py                 # Ana betik
    ├── cov_sigma_p.py          # Adaptif sigma_p hesaplama modülü
    ├── cov_sigma_v.py          # Adaptif sigma_v hesaplama modülü
    ├── imu_interp_gt/          # İşlenmiş IMU ve yer gerçeği verileri
    │   ├── MH01_imu_with_interpolated_groundtruth.csv
    │   ├── ...
    │   └── MH05_imu_with_interpolated_groundtruth.csv
    ├── VO/                     # Görsel Odometri tahminleri
    │   └── vo_pred_super_best/
    │       ├── mh1_ns.csv
    │       ├── ...
    │       └── mh5_ns.csv
    ├── outputs/                # Oluşturulan CSV sonuçları için dizin
    └── README.md
    ```

## 🚀 Kullanım

Ana betik `main.py` (veya eşdeğeri) üzerinden çalıştırılır. Çeşitli parametreler komut satırından ayarlanabilir.

### Parametreler

`main.py` betiği aşağıdaki gibi `argparse` ile tanımlanmış parametreleri kabul eder:

*   `--adaptive`: Adaptif kovaryans mekanizmasının kullanılıp kullanılmayacağını belirler (varsayılan: `False`).
*   **Sigma_p Ağırlıkları (Statik Metrikler):**
    *   `--beta_p`: Terslenmiş entropi için ağırlık (varsayılan: 1).
    *   `--epsilon_p`: Poz ki-kare hatası için ağırlık (varsayılan: 1).
    *   `--zeta_p`: Elenen anahtar kare sayısı için ağırlık (varsayılan: 1).
*   **Sigma_p Normalizasyon Minimum Değerleri:**
    *   `--entropy_norm_min`: Entropi normalizasyonu için min değer (varsayılan: 0).
    *   `--pose_chi2_norm_min`: Poz ki-kare normalizasyonu için min değer (varsayılan: 1).
    *   `--culled_norm_min`: Elenen anahtar kare normalizasyonu için min değer (varsayılan: 0).
*   **Sigma_v Ağırlıkları (Dinamik Metrikler):**
    *   `--alpha_v`: Yoğunluk değişimi için ağırlık (varsayılan: 5).
    *   `--epsilon_v`: Poz ki-kare değişimi için ağırlık (varsayılan: 2).
    *   `--zeta_H`: Artan elenmiş anahtar kare sayısı için ağırlık (varsayılan: 1).
    *   `--zeta_L`: Azalan elenmiş anahtar kare sayısı için ağırlık (varsayılan: 0).
*   **Eşik ve CASEF Parametreleri:**
    *   `--w_thr`: Görüntü güveni için ağırlıklandırma eşiği $W_{thr}$ (varsayılan: 0.25).
    *   `--d_thr`: Görüntü güveni için kesme eşiği $D_{thr}$ (varsayılan: 0.99).
    *   `--s`: CASEF aktivasyon fonksiyonu için $s$ parametresi (varsayılan: 3.0).
*   **ZUPT Parametreleri:**
    *   `--zupt_acc_thr`: ZUPT için ivme std eşiği [m/s²] (varsayılan: 0.1).
    *   `--zupt_gyro_thr`: ZUPT için jiroskop std eşiği [rad/s] (varsayılan: 0.1).
    *   `--zupt_win`: ZUPT için pencere boyutu (örnek sayısı) (varsayılan: 60).

### Örnek Çalıştırma Komutu

Adaptif mekanizmayı varsayılan parametrelerle çalıştırmak için:
```bash
python main.py --adaptive
```

Belirli parametreleri ayarlayarak çalıştırmak için:
```bash
python main.py --adaptive --alpha_v 4.5 --epsilon_v 2.2 --s 3.2 --w_thr 0.3
```

Betik, `MH01`'den `MH05`'e kadar olan sekansları `concurrent.futures.ProcessPoolExecutor` kullanarak paralel olarak işleyecektir. Sonuçlar `outputs/` klasörüne ve genel bir özet `results.csv` (veya `SAVE_RESULTS_CSV_NAME` ile belirtilen) dosyasına kaydedilecektir.

## 📈 Sonuçlar (Özet)

Önerilen adaptif hibrit Qf-ES-EKF/UKF yaklaşımı, EuRoC MAV veri setinde yapılan kapsamlı deneylerde standart ESKF ve diğer yöntemlere kıyasla belirgin performans iyileştirmeleri göstermiştir.
*   **Konum Kestirimi (ATE):** Özellikle zorlu MH04 ve MH05 sekanslarında %40'a varan iyileşme.
*   **Yönelim Kestirimi (Quaternion RMSE):** ESKF tabanlı yöntemlere göre %60'a varan daha iyi sonuçlar.
*   **Hesaplama Verimliliği:** Hibrit filtre, tam SUKF uygulamasına göre yaklaşık %47 daha hızlı çalışırken, ESKF'ye kıyasla makul bir ek yük getirmektedir.

Detaylı sayısal sonuçlar ve karşılaştırmalar için lütfen makaleye (Tablo 3, 4, 5) ve kod ile üretilen `outputs/` klasöründeki CSV dosyalarına bakınız.

Örnek bir yörünge karşılaştırması (Makale Şekil 7'den uyarlanabilir):
![Trajectory Comparison](assets/trajectory_comparison.png)
*(Not: `assets/trajectory_comparison.png` yolunu kendi yörünge karşılaştırma grafiğinizin yolu ile güncelleyiniz.)*

## 💡 Gelecek Çalışmalar

*   **Parçacık Filtresi (Particle Filter) için Uyarlanabilir Parçacık Sayısı Optimizasyonu:** Görüntü kalitesine bağlı olarak parçacık sayısını dinamik olarak ayarlama.
*   **Çoklu Sensör Füzyonu için Genişletilmiş Güven Metriği:** LiDAR, GPS, 5G gibi ek sensörleri dahil etme.
*   **Derin Öğrenme Tabanlı Güven Tahmini:** Görüntü kalitesi ve bilgi içeriği değerlendirmesi için derin öğrenme modelleri kullanma.
*   **Gerçek Zamanlı Uygulama Optimizasyonu:** Algoritma optimizasyonları ve paralel işleme teknikleri ile hesaplama verimliliğini artırma.

## 📄 Atıf

Bu çalışmayı veya kodu kullanırsanız, lütfen aşağıdaki şekilde (veya yayınlandığında makaleyi) referans gösterin:

```bibtex
@article{AsilNasibovVIO2024,
  author    = {Ufuk Asil and Efendi Nasibov},
  title     = {Görsel-Ataletsel Odometri İçin Uyarlanabilir Kovaryans ve Kuaterniyon Odaklı Hibrit Hata Durumlu EKF/UKF Yaklaşımı},
  journal   = {Henüz Yayınlanmadı / Ön Baskı},
  year      = {2024},
  note      = {Kod adresi: https://github.com/ufukasia/Ufuk-ASIL-prepaper-code.git}
}
% Yayınlandığında bu BibTeX girdisini güncelleyiniz.
```

## 📝 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır. (Projenize bir LISANS dosyası ekleyiniz.)

## 📧 İletişim

*   Ufuk Asil: `u.asil@ogr.deu.edu.tr`
*   Efendi Nasibov: `efendi.nasibov@deu.edu.tr`

Dokuz Eylül Üniversitesi, Bilgisayar Bilimleri Bölümü

---

Bu README dosyasını projenizin kök dizinine `README.md` olarak kaydedin. `assets/` klasörünü oluşturup ilgili görselleri (diagram.png, activation_functions.png, trajectory_comparison.png) bu klasöre eklemeyi unutmayın.
```

**Önemli Notlar:**
1.  **Görsel Dosyaları:** `assets/diagram.png`, `assets/activation_functions.png` ve `assets/trajectory_comparison.png` gibi yer tutucular kullandım. Bu dosyaları makalenizden alıp projenizde uygun bir `assets` (veya benzeri) klasörüne koyup yollarını README içinde güncellemelisiniz.
2.  **Makale Linki:** `[Görsel-Ataletsel Odometri İçin Uyarlanabilir Kovaryans ve Kuaterniyon Odaklı Hibrit Hata Durumlu EKF/UKF Yaklaşımı](httpsTBD_Link_To_Your_Paper_Here_If_Published_Else_Remove)` kısmını makaleniz yayınlandığında (örneğin arXiv, IEEE Xplore linki) güncelleyin.
3.  **GitHub Repo Linki:** Makalede belirtilen `https://github.com/ufukasia/Ufuk-ASIL-prepaper-code.git` linkini kullandım. Eğer bu README başka bir repo içinse, linki uygun şekilde değiştirin.
4.  **`requirements.txt`:** Kodunuzun bağımlılıklarını içeren bir `requirements.txt` dosyası oluşturmanız, kullanıcıların kurulum yapmasını kolaylaştıracaktır.
5.  **LISANS:** Projenize bir `LICENSE` dosyası (örneğin, MIT lisansı metnini içeren `LICENSE.txt` veya `LICENSE.md`) eklemeniz iyi bir pratiktir.
6.  **Dil:** README'yi tamamen Türkçe tuttuk, isteğiniz doğrultusunda.

Bu README, projenizi GitHub'da sunmak için iyi bir başlangıç noktası olacaktır. Başarılar dilerim!