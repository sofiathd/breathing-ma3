import os
import matplotlib.pyplot as plt

def save_overlay_plot(t, vid_z, ref_z, out_path, title):
    plt.figure(figsize=(12, 4))
    plt.plot(t, ref_z, label="Ref (z)", linewidth=2, alpha=0.8)
    plt.plot(t, vid_z, label="Video (z)", linewidth=1.5, alpha=0.9)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("z-score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def save_coherence_plot(f, Cxy, out_path, title, lo_hz=0.07, hi_hz=1.0):
    plt.figure(figsize=(8, 4))
    plt.plot(f, Cxy, linewidth=2)
    plt.axvspan(lo_hz, hi_hz, alpha=0.15, label="Breathing band")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def save_waveform_plot(time_s, sig_raw, sig_filtered, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_s, sig_raw, label="Raw Displacement", color='gray', alpha=0.7, linewidth=1)
    plt.title(f"{stem} | Raw Displacement (Median/PCA)")
    plt.ylabel("Displacement")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    
    plt.subplot(2, 1, 2)
    plt.plot(time_s, sig_filtered, label="Filtered Signal", color='blue', linewidth=1.5)
    plt.title(f"Filtered Breathing Signal")
    plt.xlabel("Time (s)"); plt.ylabel("Amp"); plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}_waveform.png"), dpi=100)
    plt.close()