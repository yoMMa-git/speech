import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def pink_kelleher(n_samples):
    b0, b1, b2, b3, b4, b5, b6 = 0, 0, 0, 0, 0, 0, 0
    out = np.zeros(n_samples)
    white = np.random.normal(size=n_samples)
    
    for i in range(n_samples):
        w = white[i]
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        b6 = w * 0.115926
        out[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
    return out

def pink_voss(n_samples, n_rows=16):
    array = np.zeros((n_rows, n_samples))
    white = np.random.normal(size=(n_rows, n_samples))
    

    for r in range(n_rows):
        array[r, :] = white[r, 0]
        
    for i in range(n_samples):

        idx = 0
        temp = i
        while temp % 2 == 0 and idx < n_rows - 1:
            temp //= 2
            idx += 1
        

        array[idx, i:] = white[idx, i]
        
    return np.sum(array, axis=0)

def pink_fft(n_samples, fs):
    white = np.random.normal(size=n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    

    scale = np.zeros_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0
    
    pink_fft = np.fft.irfft(fft * scale, n=n_samples)
    return pink_fft


def save_audio(data, fs, filename, target_dbfs=-12):
    current_peak = np.max(np.abs(data))
    if current_peak == 0:
        current_peak = 1e-10

    target_peak = 10 ** (target_dbfs / 20)
    scale = target_peak / current_peak

    normalized = data * scale

    normalized = np.clip(normalized, -0.99, 0.99)

    audio_int16 = (normalized * 32767).astype(np.int16)

    wavfile.write(filename, fs, audio_int16)
    print(f"✓ Сохранено: {filename} (пик: {target_dbfs} dBFS)")


def plot_analysis(data, fs, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    freqs, psd = signal.welch(data, fs, nperseg=4096)
    ax1.semilogx(freqs, 10 * np.log10(psd + 1e-10))
    ax1.set_title(f'{title} - Спектральная плотность мощности')
    ax1.set_xlabel('Частота (Гц)')
    ax1.set_ylabel('дБ/Гц')
    ax1.grid(True, which='both')
    ref_freq = freqs[1:]
    ax1.plot(ref_freq, -10 * np.log10(ref_freq) + 50, 'r--', alpha=0.5, label='-10 дБ/дек')
    ax1.legend()

    f, t, Sxx = signal.spectrogram(data, fs, nperseg=512)
    max_freq_idx = np.searchsorted(f, fs/2) 
    im = ax2.pcolormesh(t, f[:max_freq_idx], 10 * np.log10(Sxx[:max_freq_idx] + 1e-10), shading='gouraud')
    ax2.set_ylabel('Частота (Гц)')
    ax2.set_xlabel('Время (с)')
    ax2.set_title(f'{title} - Спектрограмма')
    ax2.set_ylim(0, fs/2)
    fig.colorbar(im, ax=ax2, label='дБ')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fs = 44100
    duration = 10
    n_samples = fs * duration

    methods = [
        ("kelleher", "Фильтр Келлета", pink_kelleher(n_samples)),
        ("voss", "Восс-Маккартни", pink_voss(n_samples)),
        ("fft", "FFT метод", pink_fft(n_samples, fs))
    ]

    print("Генерация розового шума и сохранение файлов...\n")
    
    for short_name, full_name, data in methods:
        print(f"\n{full_name}:")
        
        data_normalized = data / np.max(np.abs(data))

        plot_analysis(data_normalized, fs, full_name)

        filename = f"pink_noise_{short_name}.wav"
        save_audio(data, fs, filename, target_dbfs=-12)
