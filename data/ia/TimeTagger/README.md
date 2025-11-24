# TimeTagger

## How to Use

1. **Install dependencies**
```bash
cd frontend
npm install
```

2. **Run the development server**

```bash
npm run dev
```

Open the URL shown in the terminal (typically `http://localhost:5173`).

3. **Load videos**

* Drag a folder containing `.mp4` files onto the **Folder of videos** panel, or click to select.

4. **Annotate**

* Enter an FM label (e.g., `3-8L`, `12R`).
* Use keyboard shortcuts to mark **start** (`s`), **end** (`e`), and **timestamp** (`t`).

5. **Save CSV**

* Click **Save CSV** to download annotations in the custom semicolon‑delimited format.

## Features

* **Client‑side only**: no backend required — everything runs in the browser.
* **Folder ingest**: drag‑and‑drop a folder of `.mp4` files.
* **Per‑video annotations**:

  * `s` = start of range
  * `e` = end of range
  * `t` = timestamp marker
* **Strict start/end pairing** per FM label with overlap prevention.
* **FM list loading** from `/fm_list.txt` with label validation.
* **Cross‑video persistence**: switching videos doesn’t clear existing annotations.
* **Completion tracking**: FM base (e.g., `3-8`) gets struck through when both `L` and `R` are completed.
* **Export format**:

    * One line per `(video, fm_item)`
    * Fields separated by semicolons
    * `times` field contains comma‑separated `s:`/`e:` pairs

Example CSV output:

```text
video_path;fm_item;times
S0001_FM1_1.mp4;3-8L;s:10.34,e:13.73,s:17.07,e:19.07
S0001_FM1_1.mp4;9-11R;s:87.87,e:91.19,s:98.63,e:102.68
```