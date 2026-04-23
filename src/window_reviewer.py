import io
import json
import os

import ipywidgets as widgets
import librosa
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from tqdm.notebook import tqdm





class WindowReviewer:

    def __init__(self, X, Y, Z, save_path="window_decisions.json",
                 sr=128000, already_spectro=False, type_spec='mel-spectro',
                 n_fft=512, hop_length=128, fmin=0, fmax=None,
                 grid_cols=5, grid_rows=4,
                 class_filter=None, rms_min=None, rms_max=None,
                 y_reference_hz=None):

        self.X = X
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.save_path = save_path
        self.sr = sr
        self.type_spec = type_spec
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.y_reference_hz = y_reference_hz
        self.n_per_page = grid_cols * grid_rows
        self.page = 0

        # Filter
        mask = np.ones(len(X), dtype=bool)
        if class_filter is not None:
            mask &= np.isin(self.Y, class_filter)
        if rms_min is not None:
            mask &= self.Z >= rms_min
        if rms_max is not None:
            mask &= self.Z <= rms_max
        self.review_idx = np.where(mask)[0]
        self.total_pages = int(np.ceil(len(self.review_idx) / self.n_per_page))

        print(f"Reviewing {len(self.review_idx)} / {len(X)} windows")

        if already_spectro:
            # X is already a list/array of spectrograms — just index it
            print("Using pre-computed spectrograms.")
            self.specs = np.array([X[i] for i in self.review_idx])
        else:
            # X is audio — convert now
            print("Pre-computing spectrograms...")
            specs = []
            for i in tqdm(self.review_idx):
                specs.append(self._convert_single_to_image(self.X[i]))
            self.specs = np.array(specs)
            print(f"Done. {len(self.specs)} spectrograms ready.")

        self.decisions = self._load_decisions()
        self._build_ui()



    # ------------------------------------------------------------------
    # Spectrogram conversion
    # ------------------------------------------------------------------

    def _convert_single_to_image(self, x):
        if self.type_spec == 'spectro':
            D = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
            image = librosa.amplitude_to_db(abs(D))
        elif self.type_spec == 'pcen':
            x32 = (x * (2 ** 31)).astype("float32")
            stft = librosa.stft(x32, n_fft=self.n_fft, hop_length=self.hop_length)
            abs2 = stft.real ** 2 + stft.imag ** 2
            melspec = librosa.feature.melspectrogram(
                y=None, S=abs2, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax
            )
            return librosa.pcen(
                S=melspec, sr=self.sr, hop_length=self.hop_length,
                gain=0.8, bias=10, power=0.25, time_constant=0.4, eps=1e-6
            ).astype("float32")
        else:  # mel-spectro (default)
            S = librosa.feature.melspectrogram(
                y=x, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax
            )
            image = librosa.core.power_to_db(S)

        mean = image.flatten().mean()
        std = image.flatten().std()
        eps = 1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        return (spec_norm - spec_min) / (spec_max - spec_min)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_decisions(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                saved = json.load(f)
            print(f"Loaded {len(saved)} prior decisions from {self.save_path}")
            decisions = {}
            for k, v in saved.items():
                decisions[int(k)] = v["keep"] if isinstance(v, dict) else v
        else:
            decisions = {}
        for idx in self.review_idx:
            if idx not in decisions:
                decisions[idx] = True
        return decisions

    def save_decisions(self, _=None):
        output = {}
        for idx, keep in self.decisions.items():
            output[str(idx)] = {
                "keep":  keep,
                "label": int(self.Y[idx]),
                "rms":   float(self.Z[idx])
            }
        with open(self.save_path, 'w') as f:
            json.dump(output, f, indent=2)

        mask = np.array([self.decisions.get(i, True) for i in range(len(self.X))])
        np.save(self.save_path.replace('.json', '_mask.npy'), mask)
        kept = int(mask.sum())
        print(f"Saved: {kept} kept / {int((~mask).sum())} removed  →  {self.save_path}")
        return mask

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Navigation buttons
        self.btn_prev  = widgets.Button(description='← Prev',    layout=widgets.Layout(width='100px'))
        self.btn_next  = widgets.Button(description='Next →',    layout=widgets.Layout(width='100px'))
        self.btn_save  = widgets.Button(description='Save',      layout=widgets.Layout(width='100px'), button_style='success')
        self.btn_all   = widgets.Button(description='Keep all',  layout=widgets.Layout(width='110px'), button_style='info')
        self.btn_none  = widgets.Button(description='Remove all',layout=widgets.Layout(width='110px'), button_style='warning')
        self.status    = widgets.Label(value=self._status_text())

        self.btn_prev.on_click(self._prev_page)
        self.btn_next.on_click(self._next_page)
        self.btn_save.on_click(self.save_decisions)
        self.btn_all.on_click(lambda _: self._set_page(True))
        self.btn_none.on_click(lambda _: self._set_page(False))

        nav = widgets.HBox([self.btn_prev, self.btn_next,
                            self.btn_save, self.btn_all, self.btn_none,
                            self.status])

        self.output = widgets.Output()
        self.ui = widgets.VBox([nav, self.output])
        display(self.ui)
        self._draw_page()

    def _status_text(self):
        kept = sum(self.decisions.get(i, True) for i in self.review_idx)
        return f"Page {self.page+1}/{self.total_pages}  |  {kept}/{len(self.review_idx)} kept"

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _get_freq_axis_hz(self, spectro):
        n_bins = spectro.shape[0]
        fmax = self.fmax if self.fmax is not None else self.sr / 2
        if self.type_spec == 'spectro':
            return np.linspace(0, fmax, n_bins)
        return librosa.mel_frequencies(n_mels=n_bins, fmin=self.fmin, fmax=fmax)

    def _hz_to_y(self, hz, freq_axis_hz):
        hz = np.clip(hz, freq_axis_hz[0], freq_axis_hz[-1])
        return np.interp(hz, freq_axis_hz, np.arange(len(freq_axis_hz)))

    def _decorate_spectrogram(self, ax, spectro):
        if self.y_reference_hz is None:
            return
        freq_axis_hz = self._get_freq_axis_hz(spectro)
        y_ref = self._hz_to_y(self.y_reference_hz, freq_axis_hz)
        y_frac = 1.0 - (y_ref / max(1, spectro.shape[0] - 1))
        ax.axhline(y_ref, color='cyan', linestyle='--', linewidth=1)
        ax.text(-0.40, 0.5, 'Hz', transform=ax.transAxes, rotation=90,
                va='center', ha='center', fontsize=8, color='black')
        ax.text(-0.08, y_frac, f'{int(self.y_reference_hz):,}'.replace(',', ' '),
                transform=ax.transAxes, va='center', ha='right', fontsize=8, color='black')

    def _draw_page(self):
        start = self.page * self.n_per_page
        end   = min(start + self.n_per_page, len(self.review_idx))
        page_indices = self.review_idx[start:end]

        page_specs = self.specs[start:end]

        tile_buttons = []
        for tile, idx in enumerate(page_indices):
            keep = self.decisions.get(idx, True)

            fig, ax = plt.subplots(figsize=(2.2, 1.8))
            ax.imshow(page_specs[tile], aspect='auto', origin='lower', cmap='magma')
            ax.set_xticks([])
            ax.set_yticks([])
            self._decorate_spectrogram(ax, page_specs[tile])
            for spine in ax.spines.values():
                spine.set_edgecolor('#2ecc71' if keep else '#e74c3c')
                spine.set_linewidth(4)
            ax.set_title(f"#{idx} class {self.Y[idx]}", fontsize=8, pad=2)

            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', pad_inches=0.02)
            buffer.seek(0)
            img_out = widgets.Image(
                value=buffer.getvalue(),
                format='png',
                layout=widgets.Layout(width='250px')
            )
            buffer.close()
            plt.close(fig)

            btn = widgets.ToggleButton(
                value=keep,
                description='✓ Kept' if keep else '✗ Removed',
                button_style='success' if keep else 'danger',
                layout=widgets.Layout(width='120px', height='28px'),
            )
            btn._window_idx = idx

            def make_callback(b, real_idx):
                def callback(change):
                    self.decisions[real_idx] = change['new']
                    b.description = '✓ Kept' if change['new'] else '✗ Removed'
                    b.button_style = 'success' if change['new'] else 'danger'
                    self.status.value = self._status_text()
                return callback

            btn.observe(make_callback(btn, idx), names='value')
            tile_buttons.append(widgets.VBox([img_out, btn],
                                layout=widgets.Layout(align_items='center', margin='4px')))

        rows = []
        for r in range(self.grid_rows):
            row_tiles = tile_buttons[r * self.grid_cols:(r + 1) * self.grid_cols]
            if row_tiles:
                rows.append(widgets.HBox(row_tiles))

        self.status.value = self._status_text()
        with self.output:
            clear_output(wait=True)
            display(widgets.VBox(rows))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev_page(self, _=None):
        if self.page > 0:
            self.page -= 1
            self._draw_page()

    def _next_page(self, _=None):
        if self.page < self.total_pages - 1:
            self.page += 1
            self._draw_page()

    def _set_page(self, keep: bool):
        start = self.page * self.n_per_page
        end   = min(start + self.n_per_page, len(self.review_idx))
        for idx in self.review_idx[start:end]:
            self.decisions[idx] = keep
        self._draw_page()