# Video Editor Pipeline

An intelligent video editing system that automatically creates music-synchronized video edits with AI-powered clip selection and customizable visual effects.

<img width="1919" height="1047" alt="Screenshot_1" src="https://github.com/user-attachments/assets/af08b174-56e6-45c4-86f9-be31f1b87e0f" />

## Features

### Core Capabilities
- **Automatic Beat Detection** - Analyzes music tracks to identify beats and tempo
- **Intelligent Clip Selection** - Uses motion analysis and audio energy to find the most "epic" moments
- **Music Synchronization** - Automatically syncs video clips to music beats
- **Visual Effects Pipeline** - Extensive collection of color grading, motion, and glitch effects
- **Machine Learning Enhancement** - Train custom models to improve clip selection based on your preferences

### Effect System
- Color grading (cinematic, vintage, vibrant)
- Motion effects (zoom, pan, shake)
- Visual enhancements (vignette, sharpen, glow, blur)
- Glitch effects (RGB split, tone swap, chromatic aberration, line glitches)
- Real-time effect preview with adjustable parameters

### Training System
- Label clips as "good" or "bad" to train custom selection models
- Fine-tune existing models with additional training data
- Models automatically improve clip selection based on your preferences

## Requirements

- Python 3.8+
- FFmpeg and FFprobe (must be in system PATH)

## Installation

### 1. Install FFmpeg

**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Add to system PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
ffprobe -version
```

### 2. Install Python Dependencies
```bash
pip install PySide6 opencv-python numpy librosa scipy scikit-learn joblib
```

### 3. Create Required Directories
```bash
mkdir input output tmp model_output
```

## Project Structure
```
.
├── gui.py                  # Main GUI application
├── main.py                 # Command-line interface
├── editor/
│   ├── vpipe.py           # Main video pipeline
│   ├── ecore.py           # Video rendering engine
│   ├── presets.py         # Effect presets
│   └── effectsp.py        # Effect processor and templates
├── cliper/
│   ├── fin.py             # Clip processor
│   ├── epicdetector.py    # Intelligent clip detection
│   ├── audioanalyzer.py   # Music analysis
│   └── vinloader.py       # Video loader
├── model_trainer/
│   ├── start_training.py  # Initial model training
│   ├── cont_training.py   # Model fine-tuning
│   ├── train_model.py     # ML model trainer
│   ├── dataset.py         # Dataset builder
│   └── ui.py              # Training UI
├── tools/
│   ├── videocombiner.py   # Video merger
│   ├── addaudio.py        # Audio merger
│   └── filtertester.py    # Live effect tester
├── videoeffects.py        # Glitch effects processor
├── input/                 # Source videos and audio
├── output/                # Final rendered videos
├── tmp/                   # Temporary processing files
└── model_output/          # Trained ML models
```

## Quick Start

### GUI Mode (Recommended)
```bash
python gui.py
```

### Command-Line Mode
```bash
python main.py
```

## Usage Guide

### 1. Basic Video Edit

**Using GUI:**

1. Launch GUI: `python gui.py`
2. Click **"Start (VideoPipeline)"**
3. Select your audio file (MP3, WAV, OGG, FLAC)
4. Set trim seconds (removes from start/end of each video)
5. Select one or more video files
6. Wait for processing
7. Find your edit at `output/final.mp4`

**What it does:**
- Merges selected videos
- Analyzes music beats
- Selects best moments based on motion and audio
- Syncs clips to music beats
- Applies automatic effects
- Renders final video with audio

### 2. Effect Testing & Preview

1. Click **"EffectTester"**
2. Load a test image: `input/test1.jpg`
3. Adjust sliders in real-time:
   - Brightness (0-200)
   - Contrast (0-300)
   - Saturation (0-300)
   - Tone Swap (0-100)
   - Mono Hue (0-179)
   - RGB Split (0-30)
   - Line Glitch (0-50)
4. Close window when satisfied (parameters are saved)

### 3. Apply Glitch Effects

1. First, run the main pipeline to create `output/final.mp4`
2. Use EffectTester to set desired glitch parameters
3. Click **"Glitch FX"**
4. Glitch version saved to `output/glitch.mp4`

### 4. Train Custom Model

**Initial Training:**

1. Click **"Train Model"** → **"start"**
2. Select audio and video files for training
3. System shows you clips one by one
4. Rate each clip:
   - Click **"Good clip"** - This is the type of clip you want
   - Click **"Bad clip"** - This is NOT what you want
5. Click **"Finish training"** when done
6. Model saved as `model_output/epic_model_TIMESTAMP.pkl`

**Fine-Tuning Existing Model:**

1. Click **"Train Model"** → **"continue"**
2. Follow same rating process
3. Model improves based on new feedback

**Tips for Training:**
- Label at least 20-50 clips for best results
- Be consistent in your ratings
- Model automatically loads in future edits
- Each training session creates a new model file

## Configuration

### Pipeline Parameters

Edit in your code or create custom scripts:
```python
from editor import VideoPipeline

pipeline = VideoPipeline(
    input_video="input/g1.mp4",
    output_video="output/final.mp4",
    music_file="input/m1.mp3",
    bpm=120,                    # Beats per minute
    beats_per_clip=16,          # Clip length in beats
    use_effects=True,           # Enable effect processor
    effect_intensity=0.7,       # 0.0 to 1.0
    randomize_effects=False     # Random vs. pattern-based
)

pipeline.run()
```

### Detector Tuning

In `cliper/epicdetector.py`, adjust these constants:
```python
MOTION_WEIGHT = 0.55    # Weight for motion detection (0.0-1.0)
AUDIO_WEIGHT = 0.30     # Weight for audio energy (0.0-1.0)
AUDIO_GAIN = 1.0        # Audio amplification factor
SMOOTH_SIGMA = 1.5      # Smoothing factor for scores
```

### Speed Limits

In `editor/ecore.py`:
```python
MIN_SPEED = 0.6    # Minimum clip speed (slower)
MAX_SPEED = 2.2    # Maximum clip speed (faster)
```

## 🔧 Command-Line Interface
```bash
python main.py
```

**Available Commands:**
- `M` - Merge multiple video files
- `S` - Start pipeline (create edit)
- `G` - Apply glitch effects
- `R` - Add audio to video
- `T` - Launch filter tester

## Effect Templates

### Color Grading
- **Cinematic** - Enhanced contrast, slight desaturation
- **Vintage** - Warm tones, reduced saturation
- **Vibrant** - Boosted colors and contrast

### Motion Effects
- **Zoom In** - Smooth zoom effect
- **Zoom Out** - Reverse zoom
- **Pan Left/Right** - Horizontal camera movement
- **Shake** - Subtle shake effect

### Visual Effects
- **Vignette** - Darkened corners
- **Sharpen** - Enhanced detail
- **Glow** - Soft glow effect
- **Blur** - Gaussian blur

### Glitch Effects
- **RGB Split** - Chromatic aberration
- **Tone Swap** - Inverted tones
- **Monochrome Hue** - Single color overlay
- **Line Glitch** - Random horizontal displacement

## How It Works

### Clip Selection Algorithm

1. **Motion Analysis** - Calculates optical flow to detect camera/subject movement
2. **Audio Energy** - Extracts RMS energy from video audio track
3. **Beat Detection** - Identifies beats and tempo in music file
4. **Scoring** - Combines motion, audio, and ML predictions to score each segment
5. **Selection** - Picks best non-overlapping clips that match beat intervals

### Machine Learning

- Uses **Logistic Regression** for binary classification
- Features extracted: motion stats, audio stats, duration, peak ratios
- Balanced class weights to handle imbalanced training data
- Incremental learning through fine-tuning

## Troubleshooting

### FFmpeg Not Found
```
Error: 'ffmpeg' is not recognized...
```
**Solution:** Install FFmpeg and add to system PATH

### No Clips Detected
```
[ED]not found
```
**Solutions:**
- Ensure video has sufficient motion
- Check that audio file is valid
- Try adjusting MOTION_WEIGHT and AUDIO_WEIGHT
- Reduce beats_per_clip for shorter clips

### Memory Errors
**Solutions:**
- Process shorter videos
- Reduce video resolution before processing
- Close other applications
- Increase system RAM

### Model Training Fails
```
Empty training set
```
**Solution:** Label at least a few clips before clicking "Finish training"

### Video/Audio Out of Sync
**Solutions:**
- Check BPM is correct for your music
- Verify audio file matches music_file parameter
- Ensure video has consistent frame rate

## Performance Tips

- **Use shorter videos** - Under 10 minutes processes faster
- **Pre-trim videos** - Remove unwanted sections beforehand
- **Test with low resolution** - Use higher resolution only for final render
- **Train with variety** - Include diverse clips in training data
- **Close background apps** - Free up CPU/RAM during rendering


### Custom Effect Chains

Create your own effects in `editor/effectsp.py`:
```python
@staticmethod
def my_custom_effect() -> str:
    return "eq=contrast=1.2:saturation=0.8,curves=vintage"



## Technical Specifications

- **Video Formats:** MP4, MOV, MKV
- **Audio Formats:** MP3, WAV, OGG, FLAC, AAC, M4A
- **Default Output:** 1920x1080, 30fps, H.264, AAC audio
- **Speed Range:** 0.6x - 2.2x
- **Effect Processing:** Real-time preview, FFmpeg rendering

## Contributing

 Feel free to:
- Fork and modify for your needs
- Report issues or suggestions
- Share improvements

Built with:
- **OpenCV** - Video processing
- **Librosa** - Audio analysis  
- **FFmpeg** - Video rendering
- **PySide6** - GUI framework
- **scikit-learn** - Machine learning
- **NumPy/SciPy** - Numerical computing
