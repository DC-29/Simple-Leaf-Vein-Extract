import os
import subprocess


# ─── Config ───────────────────────────────────────────────────────────────────

# FluidR3_GM: free, warm, well-balanced General MIDI soundfont
SOUNDFONT_PATH = os.path.abspath("FluidR3_GM.sf2")
SAMPLE_RATE    = 44100



def check_tool(name: str) -> bool:
    """Return True if a CLI tool is available on PATH."""
    try:
        subprocess.run([name, "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def render_midi(midi_path: str, wav_path: str, soundfont_path: str,
                sample_rate: int = SAMPLE_RATE, gain: float = 0.8) -> bool:
    """
    Render a MIDI file to WAV using fluidsynth.
    gain: master volume multiplier (0.0–1.0, default 0.8 avoids clipping)
    Returns True on success.
    """
    if not check_tool("fluidsynth"):
        print("[export] ERROR: fluidsynth not found. Install with:")
        print("  macOS:  brew install fluid-synth")
        print("  Ubuntu: sudo apt install fluidsynth")
        return False

    cmd = [
        "fluidsynth",
        "-ni",                      # non-interactive, no MIDI driver
        "-g", str(gain),            # master gain
        "-F", wav_path,             # output WAV
        "-r", str(sample_rate),     # sample rate
        soundfont_path,             # soundfont
        midi_path,                  # MIDI input
    ]

    print(f"[export] Rendering MIDI → WAV...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[export] fluidsynth error:\n{result.stderr}")
        return False

    print(f"[export] Rendered to: {wav_path}")
    return True


def apply_eq(input_wav: str, output_wav: str) -> bool:
    """
    Apply EQ and reverb via sox to warm up the sound:
      - High shelf cut  at 8kHz  (-4dB)  : tame harshness
      - High shelf cut  at 12kHz (-6dB)  : remove brittle top end
      - Low shelf boost at 200Hz (+2dB)  : add warmth
      - Mid cut        at 3kHz  (-3dB)  : reduce piano attack edge
      - Reverb                           : small room, subtle
    Returns True on success, False if sox not available (wav unchanged).
    """
    if not check_tool("sox"):
        print("[export] sox not found — skipping EQ.")
        print("  macOS:  brew install sox")
        print("  Ubuntu: sudo apt install sox")
        return False

    cmd = [
        "sox", input_wav, output_wav,

        "equalizer", "300",   "0.7q", "+3",    # warmth boost
"equalizer", "2000",  "1.0q", "-5",    # cut harshness (presence range)
"equalizer", "4000",  "1.0q", "-6",    # cut attack edge (main harsh zone)
"equalizer", "8000",  "0.7q", "-6",    # soften high mids
"equalizer", "12000", "0.5q", "-9",    # heavily tame brittle top end

        # ── Reverb ──────────────────────────────────────────────────────────
        # reverb [reverberance] [hf-damping] [room-scale] [stereo-depth]
        #        [pre-delay ms] [wet-only]
        "reverb", "40", "50", "30", "30", "10",

        # ── Normalise to -1dB after processing ──────────────────────────────
        "norm", "-1",
    ]

    print(f"[export] Applying EQ + reverb...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[export] sox error:\n{result.stderr}")
        return False

    print(f"[export] EQ applied → {output_wav}")
    return True


# ─── Main export function ─────────────────────────────────────────────────────

def export_midi_to_wav(
    midi_path: str,
    output_path: str = None,
    soundfont_path: str = None,
    apply_eq_processing: bool = True,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """
    Full pipeline: MIDI → fluidsynth WAV → sox EQ → final WAV.

    Parameters
    ----------
    midi_path           : path to input .mid file
    output_path         : path for final .wav (default: same name as midi)
    soundfont_path      : path to .sf2 soundfont (auto-downloads FluidR3 if None)
    apply_eq_processing : whether to run sox EQ/reverb pass
    sample_rate         : audio sample rate in Hz

    Returns
    -------
    Path to the final output WAV file.
    """
    midi_path = os.path.abspath(midi_path)
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Default output path: same dir as MIDI, .wav extension
    if output_path is None:
        base = os.path.splitext(midi_path)[0]
        output_path = base + ".wav"
    output_path = os.path.abspath(output_path)

    # Soundfont
    soundfont_path = os.path.abspath(soundfont_path)
    
    # If EQ is enabled, render to a temp file first
    if apply_eq_processing:
        raw_wav = output_path.replace(".wav", "_raw.wav")
    else:
        raw_wav = output_path

    # Step 1: fluidsynth render
    ok = render_midi(midi_path, raw_wav, soundfont_path, sample_rate)
    if not ok:
        raise RuntimeError("fluidsynth render failed.")

    # Step 2: sox EQ (optional)
    if apply_eq_processing:
        eq_ok = apply_eq(raw_wav, output_path)
        if eq_ok:
            os.remove(raw_wav)   # clean up raw render
        else:
            # sox failed or not installed — just use the raw render
            os.rename(raw_wav, output_path)
            print("[export] Using raw render (no EQ).")

    print(f"[export] Done: {output_path}")
    return output_path


# ─── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    export_midi_to_wav(
        midi_path    = "leaf_drums.mid",
        output_path  = "leaf_music.wav",
        soundfont_path=SOUNDFONT_PATH
    )