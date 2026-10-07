# Pose + Depth — Egzersiz Form Analizi

Videodan **egzersiz formu değerlendiren** bir boru hattı. Doğru formla çekilmiş bir
referans videodan kabul aralıkları öğrenilir; sonra herhangi bir test videosu bu
aralıklara göre kare kare ölçülür ve **hangi karede neyin bozulduğu** videonun
üzerine yazılır.

İki model birlikte çalışır:

- **MediaPipe Pose** — iskelet noktaları (omuz, dirsek, el) → eklem açıları
- **MiDaS** (`MiDaS_small`, torch.hub) — tek kameradan **göreli derinlik**, yani
  gövdenin kameraya yaklaşıp uzaklaşması

Derinlik katmanı, tek kameralı analizin klasik körlüğünü kapatıyor: 2B iskelet
projeksiyonda doğru görünen ama gerçekte öne eğilmiş bir omuz, açı ölçümünde
yakalanmaz — derinlikte yakalanır.

---

## Yaklaşım: eşik yazmak değil, referanstan öğrenmek

Koda "dirsek açısı 70–135 derece olmalı" diye bir sayı yazılmadı. Sayılar **doğru
formla çekilmiş videodan** çıkarılıyor:

1. `--mode ref` doğru form videosunu baştan sona ölçer ve her metriğin
   **15–85 persentil** aralığını referans olarak yazar. Persentil kullanılır çünkü
   uç kareler (harekete giriş, çıkış, anlık takip hatası) aralığı gereksiz genişletir.
2. `--mode eval` test videosunu bu aralığa göre ölçer, aralık dışına çıkan her kare
   için insan diline çevrilmiş bir uyarı üretir ("Sol dirsek fazla açık (142° > 136°)")
   ve bunu videonun üstüne basar.

Böylece araç tek bir harekete gömülü kalmıyor: referans videosu değiştirilerek başka
bir hareket için yeniden kullanılabilir.

### Ölçülen metrikler

| Metrik | Ne yakalar |
|--------|-----------|
| `elbow_left` · `elbow_right` | Dirsek açısı — asimetri burada görünür |
| `shoulder_left` · `shoulder_right` | Omuz açısı |
| `bar_drop` | Barın referansa göre ne kadar indiği / yükseldiği |
| `shoulder_protr` | Omzun öne eğilmesi (protraksiyon) |
| `z` | MiDaS göreli derinliği, omuz ortalaması |

`bar_drop` ve `shoulder_protr` kontrollerinde bir **tolerans payı** (0.02) var:
ölçüm gürültüsü yüzünden sınırda titreyen karelerin her birinde uyarı basmak,
çıktıyı okunamaz hâle getiriyor.

---

## Ölçülen sonuç

Referans doğru form videosundan kuruldu, iki hatalı form videosu buna karşı
değerlendirildi:

| | kare | pose kaybı | en çok ihlal edilen |
|---|---|---|---|
| `hatalı form 1` | 65 | 1 kare | sağ dirsek **55/65**, sol omuz 54, sağ omuz 51 |
| `hatalı form 2` | 140 | 2 kare | **omuz protraksiyonu 56**, sağ dirsek 48 |

İki videonun **farklı** metriklerde takılması, ölçümün tek bir genel "kötü form"
sinyali üretmediğini gösteriyor: birincide açı hatası baskın, ikincide omzun öne
kaçması. Pose kaybı 65 ve 140 karede toplam 3 — takip pratikte sürekli.

Ham sayılar `ref_metrics.json`, `eval_metrics.json`, `eval2_metrics.json`
dosyalarında; özetler `*.report.json`'da.

---

## Kurulum

```bash
pip install -r requirements.txt
```

MiDaS ağırlıkları ilk çalıştırmada `torch.hub` ile indirilir (internet gerekir,
sonraki çalıştırmalarda önbellekten gelir). GPU varsa kendiliğinden kullanılır,
yoksa CPU'ya düşer.

## Kullanım

```bash
# 1) Dogru form videosundan referans arahklarini uret
python pose_depth_pipeline.py --mode ref \
    --src "doğru form ön açı.mp4" \
    --refjson reference_correct.json \
    --out ref_overlay.mp4 \
    --save_metrics ref_metrics.json

# 2) Test videosunu referansa gore degerlendir
python pose_depth_pipeline.py --mode eval \
    --src "hatalı form 1 ön açı.mp4" \
    --refjson reference_correct.json \
    --out eval_overlay.mp4 \
    --save_metrics eval_metrics.json
```

| Bayrak | Görevi |
|--------|--------|
| `--mode` | `ref` (referans üret) \| `eval` (referansa göre değerlendir) — zorunlu |
| `--src` | Giriş videosu — zorunlu |
| `--refjson` | Referans JSON: `ref`te çıktı, `eval`de girdi |
| `--out` | Overlay'li çıktı videosu |
| `--save_metrics` | Ham kare kare metrikleri JSON'a yazar |
| `--calib` | Opsiyonel omuz/dirsek piksel ofsetleri (kamera açısı düzeltmesi) |
| `--show` | Çalışırken pencere göster |

### Yardımcı scriptler

```bash
python play_video.py "hatalı form 1 ön açı.mp4"   # videoyu oynat
python play_pose.py                               # yalnizca iskelet + aci, derinlik yok
```

`play_pose.py` MiDaS yüklemeden hızlı bakmak için — iskelet takibinin videoda tutup
tutmadığını saniyeler içinde görmeyi sağlar.

---

## Dosyalar

| Dosya | Görevi |
|-------|--------|
| `pose_depth_pipeline.py` | Ana boru hattı: Pose + MiDaS, referans üretme ve değerlendirme |
| `play_pose.py` | Hafif iskelet/açı görüntüleyici (derinliksiz) |
| `play_video.py` | Basit video oynatıcı |
| `reference_correct.json` | Doğru form videosundan öğrenilen kabul aralıkları |
| `ref_metrics.json` · `eval*_metrics.json` | Kare kare ham ölçümler |
| `*.report.json` | Kare sayısı, pose kaybı, metrik başına ihlal sayısı |
| `*.mp4` | Giriş videoları ve referans overlay çıktısı |

---

## Bilinen sınırlar

- **Tek kamera, tek açı.** Videolar ön açıdan çekildi; referans aralıkları o açıya
  özgüdür. Yan açıdan çekilmiş bir video için referans yeniden üretilmelidir.
- **MiDaS göreli derinlik verir**, metrik değil. `z` değerleri karşılaştırma için
  anlamlı, mutlak mesafe olarak değil — bu yüzden `z` ihlalinde yön belirtmeyen nötr
  bir mesaj basılır.
- **Yazı tipi yolu Windows'a sabit** (`C:\Windows\Fonts\arial.ttf`). Linux/macOS'ta
  bu satır içe aktarma sırasında hata verir; o platformda yolu sistemdeki bir TTF ile
  değiştirmek gerekir.
- **Referans tek videodan** öğreniliyor; birden çok doğru form videosunu birleştirmek
  aralıkları daha sağlam yapar.
- Değerlendirme **kare bağımsızdır**: tekrar (rep) sayma veya hareketin fazına
  (iniş/çıkış) göre ayrı aralık yok.

## Lisans

Eğitim ve portföy amaçlı yayımlanmıştır; okuyabilir ve inceleyebilirsiniz.
Tüm hakları saklıdır — bkz. [`LICENSE`](LICENSE).
