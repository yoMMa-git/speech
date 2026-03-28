import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, tf2zpk, freqz, welch
from typing import Literal, Optional, List, Tuple
import os
from datetime import datetime

import pywt


class AudioFilterProcessor:
    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.fs, self.original_data = self._load_audio(input_file)
        self.current_data = self.original_data.copy()
        self.history: List[dict] = []
        print(f"Загружено: {input_file} ({self.fs} Гц, {len(self.original_data)/self.fs:.2f} с)")
    
    def _load_audio(self, filepath: str) -> Tuple[int, np.ndarray]:
        fs, data = wavfile.read(filepath)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32767.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483647.0
        return fs, data.astype(np.float32)
    
    # КИХ-ФИЛЬТРЫ
    
    def _create_fir(self, ftype: Literal['lowpass', 'highpass', 'bandpass'], 
                    cutoff, numtaps: int = 101) -> Tuple[np.ndarray, np.ndarray]:
        nyq = self.fs / 2
        Wn = [cutoff[0]/nyq, cutoff[1]/nyq] if ftype == 'bandpass' else cutoff / nyq
        b = firwin(numtaps, Wn, window='hamming', pass_zero=ftype)
        return b, np.array([1.0])
    
    # ВЕЙВЛЕТ
    
    def _wavelet(self, data: np.ndarray, wavelet: str = 'db4', 
                         level: int = 5, threshold: Optional[float] = None) -> np.ndarray:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Автоматический порог (универсальный порог Донохо)
        if threshold is None:
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Пороговая обработка деталей
        coeffs_den = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs_den, wavelet)[:len(data)]
    
    def _plot_wavelet_coeffs(self, coeffs: List[np.ndarray], save_path: str):
        fig, axes = plt.subplots(len(coeffs), 1, figsize=(10, 2*len(coeffs)))
        if len(coeffs) == 1:
            axes = [axes]
        
        names = ['Approx'] + [f'Detail {i+1}' for i in range(len(coeffs)-1)]
        for ax, c, name in zip(axes, coeffs, names):
            ax.plot(c[:1000], linewidth=0.5)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    
    # Построение графиков
    
    def _plot_comparison(self, orig: np.ndarray, filt: np.ndarray, save_path: str):
        max_s = min(len(orig), self.fs * 3)
        orig, filt = orig[:max_s], filt[:max_s]
        time = np.arange(len(orig)) / self.fs
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Спектрограммы
        for ax, data, title in [(axes[0,0], orig, 'До'), (axes[0,1], filt, 'После')]:
            d = data[:, 0] if len(data.shape) == 2 else data
            f, t, Sxx = signal.spectrogram(d, self.fs, nperseg=512)
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax.set_title(title); ax.set_ylim(0, min(10000, self.fs/2))
            fig.colorbar(im, ax=ax, label='дБ')
        
        # Временные сигналы
        for ax, data, title in [(axes[1,0], orig, 'До'), (axes[1,1], filt, 'После')]:
            d = data[:, 0] if len(data.shape) == 2 else data
            ax.plot(time, d); ax.set_title(title); ax.grid(True)
        
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    
    def _plot_freq_response(self, orig: np.ndarray, filt: np.ndarray, save_path: str):
        orig_1d = orig[:, 0] if orig.ndim == 2 else orig
        filt_1d = filt[:, 0] if filt.ndim == 2 else filt

        nperseg_orig = min(4096, len(orig_1d))
        nperseg_filt = min(4096, len(filt_1d))

        f1, p1 = welch(orig_1d, self.fs, nperseg=nperseg_orig)
        f2, p2 = welch(filt_1d, self.fs, nperseg=nperseg_filt)

        plt.figure(figsize=(10, 4))
        plt.semilogx(f1, 10 * np.log10(p1 + 1e-10), label='До', alpha=0.7)
        plt.semilogx(f2, 10 * np.log10(p2 + 1e-10), label='После', alpha=0.7)
        plt.xlabel('Гц')
        plt.ylabel('дБ/Гц')
        plt.title('АЧХ сигнала')
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_filter_response(self, b: np.ndarray, a: np.ndarray, save_path: str):
        w, h = freqz(b, a, worN=2000, fs=self.fs)
        plt.figure(figsize=(10, 4))
        plt.semilogx(w, 20 * np.log10(np.abs(h) + 1e-10))
        plt.xlabel('Гц'); plt.ylabel('дБ'); plt.title('АЧХ Фильтра')
        plt.grid(True, which='both'); plt.ylim(-100, 5)
        plt.savefig(save_path, dpi=150); plt.close()
    
    # Применение фильтров
    
    def apply_fir(self, ftype: Literal['lowpass', 'highpass', 'bandpass'], 
                  cutoff, numtaps: int = 101, output_dir: str = "./output") -> np.ndarray:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"FIR_{ftype}_{cutoff if isinstance(cutoff, int) else '-'.join(map(str, cutoff))}"
        
        print(f"\n{'='*40}\n{name}\n{'='*40}")
        
        b, a = self._create_fir(ftype, cutoff, numtaps)
        
        if len(self.current_data.shape) == 2:
            filtered = np.zeros_like(self.current_data)
            for ch in range(self.current_data.shape[1]):
                filtered[:, ch] = lfilter(b, a, self.current_data[:, ch])
        else:
            filtered = lfilter(b, a, self.current_data)
        
        filtered = np.clip(filtered, -0.99, 0.99)
        
        self._plot_comparison(self.current_data, filtered, os.path.join(output_dir, f"{name}_spectro.png"))
        self._plot_freq_response(self.current_data, filtered, os.path.join(output_dir, f"{name}_freq.png"))
        
        self.history.append({'type': 'FIR', 'name': name, 'time': ts})
        self.current_data = filtered
        print(f"Применено, графики в: {output_dir}")
        return filtered
    
    def apply_wavelet(self, wavelet: str = 'db4', level: int = 5, 
                               threshold: Optional[float] = None, 
                               output_dir: str = "./output") -> np.ndarray:
        
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"Wavelet_{wavelet}_L{level}"
        
        print(f"\n{'='*40}\n{name}\n{'='*40}")
        
        if len(self.current_data.shape) == 2:
            filtered = np.zeros_like(self.current_data)
            for ch in range(self.current_data.shape[1]):
                filtered[:, ch] = self._wavelet(self.current_data[:, ch], wavelet, level, threshold)
        else:
            filtered = self._wavelet(self.current_data, wavelet, level, threshold)
        
        filtered = np.clip(filtered, -0.99, 0.99)

        coeffs = pywt.wavedec(filtered if len(filtered.shape)==1 else filtered[:,0], wavelet, level=level)
        self._plot_wavelet_coeffs(coeffs, os.path.join(output_dir, f"{name}_wavelet_coeffs.png"))
        
        self._plot_comparison(self.current_data, filtered, os.path.join(output_dir, f"{name}_spectro.png"))
        self._plot_freq_response(self.current_data, filtered, os.path.join(output_dir, f"{name}_freq.png"))
        
        self.history.append({'type': 'Wavelet', 'name': name, 'time': ts})
        self.current_data = filtered
        print(f"Применено, графики в: {output_dir}")
        return filtered
    
    # Сохранение
    
    def save_result(self, output_file: str, target_dbfs: float = -12):
        peak = np.max(np.abs(self.current_data)) or 1e-10
        scale = (10 ** (target_dbfs / 20)) / peak
        data = np.clip(self.current_data * scale, -0.99, 0.99)
        wavfile.write(output_file, self.fs, (data * 32767).astype(np.int16))
        print(f"\nСохранено: {output_file}")
    
    def reset(self):
        self.current_data = self.original_data.copy()
        self.history = []
        print("Сброшено к оригиналу")
    
    def summary(self):
        print("\n" + "="*40)
        print("ИСТОРИЯ ОБРАБОТКИ")
        print("="*40)
        for i, h in enumerate(self.history, 1):
            print(f"{i}. [{h['type']}] {h['name']} ({h['time']})")
        print(f"Всего: {len(self.history)} операций")
        print("="*40)
    
    # Удобные алиасы
    def lowpass(self, cutoff, numtaps=101, **kwargs):
        return self.apply_fir('lowpass', cutoff, numtaps, **kwargs)
    
    def highpass(self, cutoff, numtaps=101, **kwargs):
        return self.apply_fir('highpass', cutoff, numtaps, **kwargs)
    
    def bandpass(self, low, high, numtaps=101, **kwargs):
        return self.apply_fir('bandpass', [low, high], numtaps, **kwargs)
    
    def wavelet(self, **kwargs):
        return self.apply_wavelet(**kwargs)


# Пример использования

if __name__ == "__main__":
    proc = AudioFilterProcessor("filter_results/speech/Tim Urban_ Inside the mind of a master procrastinator TED.wav")
    
    # proc.bandpass(low=2000, high=8000, numtaps=101)

    proc.lowpass(cutoff=7000)

    proc.wavelet(wavelet='db2', level=2)
    
    
    # История и сохранение
    proc.summary()
    proc.save_result("final_output.wav")
    
    # Сброс
    # proc.reset()