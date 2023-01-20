import numpy as np
import sys
# Quick script to find best HOD parameters from the fitting chains
path_in = sys.argv[1]
max_val_total = -100000
for i in range(1,11):
    try:
        temp = np.load(path_in+"diff_start_low_prior_"+str(i)+"_log_probs.npy")
        max_val = np.max(temp)
        print(max_val)
        if max_val > max_val_total:
            max_val_total = max_val
            best_fit_number = i
    except Exception:
        continue

print(max_val_total)
print(best_fit_number)

best_params = np.genfromtxt(path_in+"diff_start_low_prior_"+str(best_fit_number)+"_best_params_smooth_functions.txt")
np.savetxt("best_params.txt",best_params)

best_chain = np.load(path_in+"diff_start_low_prior_"+str(best_fit_number)+"_params.npy")
best_chain_log_probs = np.load(path_in+"diff_start_low_prior_"+str(best_fit_number)+"_log_probs.npy")

np.save("best_params_chain.npy",best_chain)
np.save("best_params_chain_log_probs.npy",best_chain_log_probs)

