
<img alt="szakdolgozat" src="https://github.com/user-attachments/assets/8363fefc-825f-4a6c-9181-8f56f87406ec" />

# Szakdolgozat

Ez a projekt a szakdolgozatomhoz készült, amely a **[CASIA1.0](https://ieeexplore.ieee.org/document/6625374)** adatkészlet használatával különböző képfeldolgozási és gépi tanulási technikákat vizsgál a manipulált képek felismerésére, a **[Detection of Digital Image Forgery using Fast FourierTransform and Local Features](https://ieeexplore.ieee.org/document/8776709)** nevű kutatásra alapozva. A kutatás célja a digitális képhamisítás kimutatása Fast Fourier-transzformáció (FFT) és helyi textúra-leírók (LBP, LTP és ELTP) alkalmazásával.

## Tartalomjegyzék
- [Bevezetés](#bevezetés)
- [Feladat szöveges leírása](#feladat-szöveges-leírása)
- [Követelmények](#követelmények)
- [Telepítés](#telepítés)
- [Használat](#használat)
- [Eredmények](#eredmények)
- [Következtetések](#következtetések)
- [Fájlstruktúra](#fájlstruktúra)
- [Hibakeresés](#hibakeresés)
- [Kapcsolat](#kapcsolat)

## Bevezetés
A projekt célja egy gépi tanulási modell kifejlesztése, amely képes azonosítani a módosított képeket.

A projekt az alábbi fő lépéseket tartalmazza:
1. Az adatok előfeldolgozása
2. A képjellemzők kinyerése
3. A modell betanítása és kiértékelése

**Az 1. módszer folyamatai:**
<img alt="lepesek1" src="https://github.com/user-attachments/assets/ebdefab5-197f-40d4-b6bc-a40771778034" />

**A 2. módszer folyamatai:**
<img alt="lepesek2" src="https://github.com/user-attachments/assets/9bfac8e3-4191-4180-b3f4-72dd633abbb8" />

## Követelmények
A projekt futtatásához a következő könyvtárak és eszközök szükségesek:
- Python 3.x
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## Telepítés
1. **Könyvtárak Telepítése**: Szükséges Python könyvtárak telepítése az alábbi parancs futtatásával:
   ```bash
   pip install opencv-python numpy scikit-learn matplotlib
   ```
   
2. **Adatok Letöltése**:
   [CASIA1.0 adatkészlet](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) letöltése, és elhelyezése a `data/` könyvtárban.
   
   A projekt már tartalmazza az adatkészletet.

## Futtatás

### Futtatás egyben: 
   Futtasuk az `run_scripts.py` scriptet:
   ```bash
   python src/run_scripts.py
   ```

### Futtatás külön-külön:
1. **Adatok Előfeldolgozása**: Futtasuk az `preprocess.py` scriptet az adatok előfeldolgozásához:
   ```bash
   python src/preprocess.py
   ```

2. **Jellemzők kinyerése**: Futtasuk a `feature_extraction.py` scriptet a jellemzők kinyeréséhez:
   ```bash
   python src/feature_extraction.py
   ```

3. **Modell Betanítása**: Futtasuk a `train_classifier.py` scriptet a modell betanításához:
   ```bash
   python src/train_classifier.py
   ```

## Eredmények
A betanított modell teljesítményének értékeléséhez a következő metrikákat használom:

**Pontosság (Accuracy)**: Az összes helyes előrejelzés aránya az összes előrejelzéshez képest.

**Visszahívás (Recall)**: A helyesen előrejelzett pozitív esetek aránya az összes tényleges pozitív esethez képest.

<img alt="eredmenyek" src="https://github.com/user-attachments/assets/c42e41bf-b44f-492e-b155-fd5a1022d368" />

Az eredményeket a `results` mappában tároljuk.

## Következtetések
Az eredmények alapján az FFT-ELTP módszerrel a modell közepesen hatékonyan ismerte fel és  különböztette meg az eltérő textúrákat. További munka lehet szükséges a modell fejlesztéséhez – például többféle jellemző kivonása, más tanító-tesztadat arány, másik adatkészlet (CASIA2.0) használata segíthet.

## Fájlstruktúra
A projekt könyvtárszerkezete a következő:
```
├── data/
│   └── CASIA1.0/
│         ├── Au/
│         └── Tp/
├── src/
│   ├── results/
│   │   ├── metrics/
│   │   │   └── evaluation_metrics.txt
│   │   ├── plots/
│   │   │   ├── data_distribution.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── metrics_plot.png
│   │   │   └── roc_curve.png
│   │   ├── classifier.joblib
│   │   ├── features_labels.joblib
│   │   ├── preprocessed_data.joblib
│   │   └── results.txt
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── run_scripts.py
│   └── train_classifier.py
├── .gitattributes
├── .gitignore
└── README.md
```

## Hibakeresés
Ha problémák merülnek fel a scriptek futtatása közben, itt van néhány gyakori hiba és megoldás:

**Adat Betöltési Hiba**:

Győződjünk meg róla, hogy a `data/` mappában találhatóak a szükséges `Au` és `Tp` mappák, és a fájlnevek helyesen vannak megadva.

**Kép Konvertálási Hiba**:

Ellenőrizzük, hogy az `OpenCV` megfelelően telepítve van és a képek elérhetők.

**Import Hiba**:

Ha valamely csomag nem megfelelően töltödött le, próbáljuk frissíteni, vagy újra letölteni a `pip` segítségével.

**Virtuális Környezet Hiba**:

Ha a virtuális környezetben nem fut le megfeleleően, töröljük és hozzuk újra létre a `venv`-et.

**Modell Betöltési Hiba**:

Győződjünk meg róla, hogy a `classifier.joblib` fájl elérhető a `results/` mappában, és helyesen van elmentve.

**Memória Túlcsordulás, Lassú Futás Hiba**:

Ha memória problémák merülnek fel, próbáljunk kisebb batch méretet választani (pl. `batch_size=100`).

## Kapcsolat
Email: [timeaszabo1217@gmail.com](mailto:timeaszabo1217@gmail.com)

LinkedIn: [timeaszabo1217](https://www.linkedin.com/in/timeaszabo1217/)
