# README (WIP)

# Running the code
See example.py

# Generating new lookup tables
Use the function build_1d_interp_table to build an interpolation table for each snapshot. You must do this for each density field before running the code. If you don't, it will compute fcoll on the fly for each filter, which is very slow. On lux, you can find interpolation tables for the example snapshots in /data/groups/comp-astro/rmebane/interp_files.

# Halo finding
Instructions for halo finding and implementing it into the reionization model.
