# MIT KEMAR head-related impulse responses (compact set)

`mit_kemar_compact.npz` holds the 368 compact-set impulse responses (128 taps, 44.1 kHz, left and
right ear) from Bill Gardner and Keith Martin, "HRTF Measurements of a KEMAR Dummy-Head
Microphone", MIT Media Lab, 1994 (https://sound.media.mit.edu/resources/KEMAR.html). The
authors allow free use, including commercial use, provided the source is acknowledged.

Repacked from `compact.zip` without changes: `elev` and `azim` (degrees; azimuth 0 to 180 is the
right half, the left half is its mirror with the ears swapped), `hrir` (int16, shape 368 x 128 x
2) and `rate`. Used by metasight/spatial_sound.py.

`mit_kemar_large_pinna_el0.npz` holds the 72 horizontal-plane (elevation 0) right-ear responses
of the same measurements' full set (`full.zip`, files `elev0/R0e*a.wav`, 512 taps, 44.1 kHz):
KEMAR's right ear wore the large pinna (DB-065), the left the normal one (DB-061) the compact
set is built from. `azim` (0 to 355, positive to the right), `hrir_right` (int16, 72 x 512),
`rate`. Used only by metasight/spatial_listener.py as the simulated listener's own ears, so
that the listener is not scored with the renderer's ears.
