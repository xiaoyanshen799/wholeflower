import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm

# -------------------------------------------------
# 1) load data
# -------------------------------------------------
df = pd.read_csv("/home/ubuntu/wholeflower/femnist/mnist_cpu_theta.csv")

# anchor rows that every client has
anchor_cpu = np.array([1.0,0.6,0.3])

# percentages a% in (0,1) to generate extra points at c = 1 * a
# Use small interval (e.g., 5%) and theta1 = theta0 / a
# where theta0 is theta at c=1 for each client
a_values = np.arange(0.6, 1.0, 0.05)

# power-law + constant model
def power_shift(cpu, alpha, beta, gamma):
    return alpha * cpu**-beta

print("client  θ:α      θ:β     θ:γ     cpu(θ=50)   k:α      k:β     k:γ     k@θ=50")
print("------  -------  ------  ------  ----------  -------  ------  ------  --------")

# colour palette for plotting
cmap   = cm.get_cmap("tab10")
colour = {}

plt.figure(figsize=(10, 6))
cpu_list = np.array([0.4543, 0.4064, 0.6472, 0.5288, 0.2915, 0.2534, 0.3061, 0.2582, 0.3080, 0.2819, 0.8753, 0.7492, 0.8226]) 
theta_list = np.array([30.0441, 30.0569, 31.4793, 30.3318, 28.6419, 31.7776, 29.3117, 30.3779, 29.0103, 32.8483, 29.8898, 29.3561, 29.8699])
k_list = np.array([0.0576, 0.2378, 0.1820, 0.1626, 0.0820, 0.2176, 0.1623, 0.1141, 0.3168, 0.2338, 0.4773, 0.0675, 0.3330])
cpu_list1 = np.array([0.5763, 0.6593, 0.2277, 0.5506, 0.5298, 0.8136, 0.4992, 0.6533, 0.8757, 0.7265])
theta_list1 = np.array([254.7980, 249.2708, 246.2126, 253.9063, 250.0513, 253.3901, 248.8141, 249.71884,252.6335, 250.5417])
k_list1=np.array([0.1333, 1.0547, 0.3996, 0.2106, 0.1420, 0.3057, 0.1132, 1.1509, 0.2441, 0.2643])
added_scaled_legend = False
for i, (cid, sub) in enumerate(df.groupby("id"), start=0):
    colour[cid] = cmap(i % 10)

    # ------ collect anchor values ------------------------------------------
    theta_anchor = []
    k_anchor     = []
    cpu_anchor = anchor_cpu.copy()
    for c in anchor_cpu:
        row = sub[np.isclose(sub["cpu"], c)]
        if row.empty:
            raise ValueError(f"Client {cid}: missing cpu {c:.2f}")
        theta_anchor.append(row.iloc[0]["theta"])
        k_anchor.append(row.iloc[0]["k"])
        
    # cpu_anchor = np.append(anchor_cpu, cpu_list[i])
    # # cpu_anchor = np.append(cpu_anchor, cpu_list1[i])
    # theta_anchor.append(theta_list[i])
    # # theta_anchor.append(theta_list1[i])
    # k_anchor.append(k_list[i])
    # # k_anchor.append(k_list1[i])
    
    theta_anchor = np.asarray(theta_anchor)
    k_anchor     = np.asarray(k_anchor)

    # ------ fit θ(c) --------------------------------------------------------
    p0_theta = (theta_anchor.max(), 1.0, 0.0)
    (a_th, b_th, g_th), _ = curve_fit(
        power_shift, cpu_anchor, theta_anchor,
        p0=p0_theta, bounds=(-10, np.inf), maxfev=20000)

    # ------ fit k(c) --------------------------------------------------------
    # p0_k = (k_anchor.max(), 1.0, 0.0)
    # (a_k, b_k, g_k), _ = curve_fit(
    #     power_shift, cpu_anchor, k_anchor,
    #     p0=p0_k, bounds=(0, np.inf), maxfev=10_000)

    # ------ CPU that yields θ = 50 -----------------------------------------
    target_theta = 30.0
    if a_th <= 0 or b_th == 0 or target_theta <= g_th:
        cpu_50 = np.nan
    else:
        cpu_50 = (a_th / (target_theta - g_th)) ** (1.0 / b_th)

        # k_at_50 = power_shift(cpu_50, a_k, b_k, g_k)

    # Print client id as-is (may be non-numeric like an IP string)
    print(f"{str(cid):>6}  {a_th:7.2f} {b_th:7.4f} {g_th:7.2f} "
          f"{cpu_50:10.4f}  {cpu_anchor} {theta_anchor} ")

    # ------ plot θ fit ------------------------------------------------------
    cpu_grid = np.linspace(sub["cpu"].min(),
                           sub["cpu"].max(), 200)
    plt.scatter(sub["cpu"], sub["theta"],
                s=25, alpha=0.6, color=colour[cid])
    plt.plot(cpu_grid,
             power_shift(cpu_grid, a_th, b_th, g_th),
             '--', lw=2, label=f"θ(c) {cid}", color=colour[cid])

    # ------ add extra scaled points (0 < c < 1) ----------------------------
    # theta0: theta at c0=1 for this client (from anchor data)
    # idx_1 = np.where(np.isclose(cpu_anchor, 1.0))[0]
    # if idx_1.size > 0:
    #     theta0 = theta_anchor[idx_1[0]]
    #     c_extra = 1.0 * a_values
    #     theta_extra = theta0 / a_values
    #     lbl = r"θ1= θ0/a @ c=a" if not added_scaled_legend else "_nolegend_"
    #     plt.scatter(c_extra, theta_extra,
    #                 s=22, marker='x', alpha=0.85,
    #                 color=colour[cid], label=lbl)
    #     added_scaled_legend = True

# plot cosmetics
plt.xlabel("cpu_size")
plt.ylabel(r"$\theta$")
plt.title("K * b + gamma")
plt.grid(alpha=0.3)

# de-duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
uniq = dict(zip(labels, handles))
plt.legend(uniq.values(), uniq.keys(),
           bbox_to_anchor=(1.02, 0.5), loc="center left",
           fontsize=8)
plt.tight_layout()
plt.savefig("theta_cpu_model.png", dpi=300)
plt.show()
