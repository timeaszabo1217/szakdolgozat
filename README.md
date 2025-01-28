# Szakdolgozat Projekt

Ez a projekt a szakdolgozatomhoz készült, amely a **CASIA2.0** adatkészlet használatával különböző képfeldolgozási és gépi tanulási technikákat vizsgál a manipulált képek felismerésére, a **Detection of Digital Image Forgery using Fast FourierTransform and Local Features** nevű kutatásra alapozva. A kutatás célja a digitális képhamisítás kimutatása gyors Fourier-transzformáció (FFT) és helyi textúra-leírók (LBP, LTP és ELTP) alkalmazásával.

## Tartalomjegyzék
- [Bevezetés](#bevezetés)
- [Követelmények](#követelmények)
- [Telepítés](#telepítés)
- [Használat](#használat)
- [Eredmények](#eredmények)
- [Következtetések](#következtetések)

## Bevezetés
A projekt célja, hogy egy gépi tanulási modellt fejlesszen ki a manipulált képek felismerésére. A projekt az alábbi fő lépéseket tartalmazza:
1. Az adatok előfeldolgozása és jellemzők kinyerése
2. A modell betanítása és értékelése
3. Az osztályozó tesztelése új adatkészleteken

## Követelmények
A projekt futtatásához a következő könyvtárak és eszközök szükségesek:
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Telepítés
1. **Könyvtárak Telepítése**: Szükséges Python könyvtárak telepítése az alábbi parancs futtatásával:
   ```bash
   pip install opencv-python numpy matplotlib scikit-learn
   ```
   
2. **Adatok Letöltése**:
   [CASIA2.0 adatkészlet](https://paperswithcode.com/dataset/casia-v2) letöltése, és elhelyezése a `data/` könyvtárban.

## Használat
1. **Adatok Előfeldolgozása**: Futtasuk az `preprocess.py` scriptet az adatok előfeldolgozásához:
   ```bash
   python scripts/preprocess.py
   ```

2. **Jellemzők kinyerése**: Futtasuk a `feature_extraction.py` scriptet a jellemzők kinyeréséhez:
   ```bash
   python scripts/feature_extraction.py
   ```

3. **Modell Betanítása**: Futtasuk a `train_classifier.py` scriptet a modell betanításához:
   ```bash
   python scripts/train_classifier.py
   ```
   
4. **Osztályozó Tesztelése**: Futtasuk a `test_classifier.py` scriptet az új adatkészleten történő teszteléshez:
   ```bash
   python scripts/test_classifier.py
   ```

## Eredmények
A betanított modell teljesítményének értékeléséhez a következő metrikákat használom:

**Pontosság (Accuracy)**: Az összes helyes előrejelzés aránya az összes előrejelzéshez képest.

**Recall**: A helyesen előrejelzett pozitív esetek aránya az összes tényleges pozitív esethez képest.

Az eredményeket a `results` mappában tároljuk:

- `evaluation_metrics.txt`: A modell teljesítményének metrikái

- `metrics_plot.png`: A teljesítménymutatók grafikonja

## Következtetések
További munkák és kutatások szükségesek a modell pontosságának és recall értékének növelése érdekében.

## Fájlstruktúra
A projekt könyvtárszerkezete a következő:
```
├── data/
│   ├── CASIA2.0_revised/
│   └── CASIA2.0_Groundtruth/
├── scripts/
│   ├── results/
│   │   ├── classifier_model.pkl
│   │   ├── evaluation_metrics.txt
│   │   ├── features_labels.npz
│   │   └── metrics_plot.png
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   └── test_classifier.py
└── README.md
```

## Hibakeresés
Ha problémák merülnek fel a script futtatása közben, itt van néhány gyakori hiba és megoldás:

**Adatok Betöltési Hibája**:

Győződjünk meg róla, hogy az `data/` könyvtárban találhatóak a szükséges képek, és a fájlnevek helyesen vannak megadva.

**Kép Konvertálási Hiba**:

Ellenőrizzük, hogy az OpenCV megfelelően telepítve van, és a képek elérhetők-e.

**Modell Betöltési Hiba**:

Győződjünk meg róla, hogy a `classifier_model.pkl` fájl elérhető a `results/` könyvtárban, és helyesen van elmentve.

## Kapcsolat
Email: [timeaszabo1217@gmail.com](mailto:timeaszabo1217@gmail.com)

LinkedIn: [timeaszabo1217](https://www.linkedin.com/in/timeaszabo1217/)
