# MIT KEMAR head-related impulse responses (compact set)

`mit_kemar_compact.npz` holds the 368 compact-set impulse responses (128 taps, 44.1 kHz, left and
right ear) from Bill Gardner and Keith Martin, "HRTF Measurements of a KEMAR Dummy-Head
Microphone", MIT Media Lab, 1994 (https://sound.media.mit.edu/resources/KEMAR.html). The
authors allow free use, including commercial use, provided the source is acknowledged.

Repacked from `compact.zip` without changes: `elev` and `azim` (degrees; azimuth 0 to 180 is the
right half, the left half is its mirror with the ears swapped), `hrir` (int16, shape 368 x 128 x
2) and `rate`. Used by metasight/spatial_sound.py.
