# README (WIP)

# Running the code
See example.py

# Generating new lookup tables
Use the function build_1d_interp_table to build an interpolation table for each snapshot. You must do this for each density field before running the code. If you don't, it will compute fcoll on the fly for each filter, which is very slow. On lux, you can find interpolation tables for the example snapshots in /data/groups/comp-astro/rmebane/interp_files.

NOTE: Since this is computing fcoll weighted by the star formation parameters, you will also need to recompute the tables each time you want to try a new star formation model. If you are just changing the parameter by a constant, you can use the optional "boost" parameter in filter_box.

# Halo finding
Instructions for halo finding and implementing it into the reionization model.
