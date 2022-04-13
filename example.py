from filter import *
from data import *

# example script for filtering a 500 Mpc, 256^3 box with 24 snapshots

out_prefix = "/data/groups/comp-astro/rmebane/output/nion_box_ex"
int_file_prefix = "/data/groups/comp-astro/rmebane/interp_files/nion_table_1d_n"
length = 500 # Mpc

for i in range(1,25):
	# get current snapshot
	den, z = get_snapshot(i)
	# create a zeroed Nrec array if this is our first snapshot
	if(i==1):
		Nrec_box = cupy.zeros_like(den)
		zprev = z
	res = filter_box(z, zprev, den, length, out_prefix + "_n" + str(i), Nrec = Nrec_box, max_R = 50, NR = 50, imax=1, interp_file=int_file_prefix + str(i) + "_rho.dat")
	zprev = z
	Nrec_box = res[1]
