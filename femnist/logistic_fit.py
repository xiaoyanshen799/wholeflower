import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit

# ----------- 读取 CSV 文件 -----------

# CSV 文件路径
csv_file = "/home/xiaoyan/wholeflower/logs/comm_times.csv"
EXCLUDED_CLIENT = "ipv4:10.0.0.4:40254"
# 读取 CSV 文件
df = pd.read_csv(csv_file)
# df = df[df["client_id"].astype(str) != EXCLUDED_CLIENT]

# 数据存储
client_durations = defaultdict(list)
client_colors = {}
empirical_cdfs = {}

MAX_DURATION = 430.0


# 从 CSV 中提取客户端时长信息
for _, row in df.iterrows():
    client_id = str(row["client_id"])  # 客户端ID
    # round_time = row["server_to_client_time"] + row["client_to_server_time"]    # 任务完成时长
    #round_time = row["computing_time"]  
    round_time = row["client_train_s"]  # 任务完成时长
    # round_time = row["client_train_s"] 
    client_durations[client_id].append({"duration": round_time})  # 存储时长


from scipy.stats import lognorm, weibull_min, t
from scipy.interpolate import PchipInterpolator
# ----------- 定义 Logistic 函数 -----------
def logistic_cdf(t, theta, k):
    """2 参数 Logistic 的 CDF."""
    return 1.0 / (1.0 + np.exp(-(t - theta)/k))

def weibull_cdf(t, c, lam):
    return 1 - np.exp(- (t/lam)**c)

def lognorm_cdf(t, s, loc, scale):
    return lognorm.cdf(t, s, loc=loc, scale=scale)

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def aic(n, sse, num_params):
    return n * np.log(sse/n) + 2 * num_params
# 创建图形
plt.figure(figsize=(16, 8))
colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(client_durations)))

# 用于存储每个客户端的 (theta, k) 参数
client_params = {}
weibull_params = {}
t_params = {}
all_durations = []  # 收集所有客户端的所有时长，以便确定全局绘制范围
add = 0
for color, (client, records) in zip(colors, client_durations.items()):
    client_colors[client] = color
    # if client != "1.0":
    #     continue
    # 取出该客户端的 duration 列表
    durations = [
        r["duration"]
        for r in records
        if "duration" in r and r["duration"] <= MAX_DURATION
    ]
    durations.sort()
    if not durations:
        continue

    all_durations.extend(durations)  # 收集到全局

    # 计算经验CDF
    cdf = np.arange(1, len(durations)+1) / float(len(durations))

    # -- 绘制经验CDF（散点或折线都可以） --
    if client == "11.0":
        plt.plot(durations, cdf, "bs", label=f"Client = {client} (empirical)",
                 color=client_colors[client], markersize=3, alpha=0.5)
    else:
        plt.plot(durations, cdf, "o", label=f"Client = {client} (empirical)",
             color=client_colors[client], markersize=3, alpha=1.0)
    
    # -- 做 Logistic 拟合 --
    x_data = np.array(durations)
    y_data = np.array(cdf)
    empirical_cdfs[client] = (x_data, y_data)

    # 初始猜测值 p0
    theta_init = np.median(x_data)  # 中位数
    k_init = 0.5
    p0 = [theta_init, k_init]

    try:
        # data = np.array(durations)   # 直接把所有 duration 扔进去
        # p_log, _   = curve_fit(logistic_cdf, x_data, y_data, p0=[np.median(x_data), 0.1])
        # p_weib, _  = curve_fit(weibull_cdf,  x_data, y_data, p0=[1.5, np.mean(x_data)])
        # p_logn, _  = curve_fit(lognorm_cdf,  x_data, y_data, p0=[1.0, 0.0, np.mean(x_data)])
        # popt, pcov = curve_fit(logistic_cdf, x_data, y_data, p0=p0, maxfev=10000)
        theta0 = np.median(x_data)
        k0 = (np.percentile(x_data, 75) - np.percentile(x_data, 25)) / 4  # IQR/4≈σ
        p0 = [theta0, max(k0, 0.1)]

        bounds = ([min(x_data), 0.05],     # k ≥ 0.05 s
                [max(x_data), 300.0])

        popt, _ = curve_fit(logistic_cdf, x_data, y_data,
                            p0=p0, bounds=bounds,
                            maxfev=20000, loss='soft_l1')
        # n = len(x_data)
        # sse_log  = sse(y_data, logistic_cdf(x_data, *p_log))
        # sse_weib = sse(y_data, weibull_cdf(x_data, *p_weib))
        # sse_logn = sse(y_data, lognorm_cdf(x_data, *p_logn))

        # aic_log  = aic(n, sse_log,  len(p_log))
        # aic_weib = aic(n, sse_weib, len(p_weib))
        # aic_logn = aic(n, sse_logn, len(p_logn))

        # print("Logistic:",  "SSE=", sse_log,  "AIC=", aic_log)
        # print("Weibull:",   "SSE=", sse_weib, "AIC=", aic_weib)
        # print("LogNormal:","SSE=", sse_logn, "AIC=", aic_logn)
        theta_hat, k_hat = popt
        print(f"[Logistic Fit] Client {client} => theta = {theta_hat:.4f}, k = {k_hat:.4f}")

        # 存储到 client_params 里
        client_params[client] = (theta_hat, k_hat)

        # 构造平滑曲线，用来绘制拟合结果
        x_fit = np.linspace(min(x_data), max(x_data), 200)
        y_fit = logistic_cdf(x_fit, theta_hat, k_hat)

        # 在图上绘制光滑的 Logistic 拟合曲线
        plt.plot(
            x_fit,
            y_fit,
            linestyle="-",
            color=color,
            linewidth=1.8,
            label=f"client {client} logistic fit",
        )

        # Student-t Location-Scale 拟合
        try:
            df_hat, loc_hat, scale_hat = t.fit(x_data)
            if scale_hat <= 0:
                raise RuntimeError('scale <= 0')
            t_params[client] = {
                "df": float(df_hat),
                "loc": float(loc_hat),
                "scale": float(scale_hat),
            }
            y_t = t.cdf((x_fit - loc_hat) / scale_hat, df_hat)
            y_t = np.clip(y_t, 0.0, 1.0)
            # plt.plot(
            #     x_fit,
            #     y_t,
            #     linestyle='-.',
            #     color=color,
            #     linewidth=1.4,
            #     alpha=0.9,
            #     label=f"client {client} t-loc-scale fit",
            # )
            # print(
            #     f"[t Fit] Client {client} => df = {df_hat:.4f}, loc = {loc_hat:.4f}, scale = {scale_hat:.4f}"
            # )
        except Exception as t_exc:  # noqa: BLE001
            print(f"[t Fit] Client {client} failed: {t_exc}")

        try:
            shape_hat, loc_hat, scale_hat = weibull_min.fit(x_data, floc=0.0)
            weibull_params[client] = {
                "shape": float(shape_hat),
                "loc": float(loc_hat),
                "scale": float(scale_hat),
            }
            y_weibull = weibull_min.cdf(x_fit, shape_hat, loc=loc_hat, scale=scale_hat)
            # plt.plot(
            #     x_fit,
            #     y_weibull,
            #     linestyle=":",
            #     color=color,
            #     linewidth=1.6,
            #     alpha=0.9,
            #     label=f"client {client} weibull fit",
            # )
        except Exception as weib_exc:  # noqa: BLE001
            print(f"[Weibull Fit] Client {client} failed: {weib_exc}")
    except RuntimeError as e:
        print(f"Client {client} fitting failed: {e}")
        continue

# ========== 计算并绘制“所有客户端都完成”的总 CDF ==========

def F_max(t):
    """所有客户端完成时间最大值的CDF: \prod_i F_i(t)."""
    val = 1.0
    for (theta_hat, k_hat) in client_params.values():
        val *= logistic_cdf(t, theta_hat, k_hat)
    return val

if client_params and all_durations:  # 确保至少有一个客户端成功拟合
    # 在一个全局时间网格上计算总CDF
    t_min = min(all_durations)
    t_max = max(all_durations)
    x_total = np.linspace(t_min, t_max, 300)
    y_total = [F_max(t) for t in x_total]

    # 在图上绘制
    # plt.plot(x_total, y_total, "k-", linewidth=2,
    #          label="All clients finished (max)")
    t_grid = np.linspace(t_min, t_max, 400)
else:
    t_grid = np.array([])

# 2) product of all *fitted* client CDFs
if client_params and t_grid.size:
    y_total_fitted = np.ones_like(t_grid)
    for theta_hat, k_hat in client_params.values():
        y_total_fitted *= logistic_cdf(t_grid, theta_hat, k_hat)
    # plt.plot(t_grid, y_total_fitted, "bs", lw=2,
    #          label="Product – fitted clients")
else:
    y_total_fitted = np.array([])
root_exp             = 9.9089
# 3) product of 13 identical clients (θ=30, k=0.2)
theta_id, k_id, n_id = 30.0, 0.2, 13
y_identical = logistic_cdf(t_grid, theta_id, k_id) ** n_id if t_grid.size else np.array([])
y_root = y_identical ** root_exp if t_grid.size else np.array([])

# # 设置图表样式
# plt.title("CDF of Client Durations")
# plt.xlabel("Time (s)")
# plt.ylabel("Cumulative Probability")
# plt.legend(loc="lower right", fontsize=8, ncol=2)
# plt.grid()
# plt.tight_layout()

# # 保存并显示图表
# plt.savefig("client_durations_cdf1.png")
# plt.show()

theta_new = 28.5                              # 固定 θ
K_FIXED = 0.2                            # 固定的 k

def logistic_gen_cdf(t, gamma):
    return 1.0 / ((1.0 + np.exp(-(t - theta_new) / K_FIXED)) ** gamma)

# ------------------------------------------------------------------
# 2)  Fit γ_i for every client with fixed k
# ------------------------------------------------------------------
gamma_params = {}

for cid in client_params.keys():
    times = np.array([r["duration"] for r in client_durations[cid] if r["duration"] <= MAX_DURATION])
    times.sort()
    if times.size == 0:
        continue
    y_emp = np.arange(1, len(times) + 1) / len(times)

    try:
        popt, _ = curve_fit(
            logistic_gen_cdf,
            times,
            y_emp,
            p0=[1.0],
            bounds=(0.0001, 200000.0),
            maxfev=10000,
        )
        gamma_hat = float(popt[0])
        gamma_params[cid] = gamma_hat
        t_plot = np.linspace(times.min(), times.max(), 200)
        # plt.plot(
        #     t_plot,
        #     logistic_gen_cdf(t_plot, gamma_hat),
        #     "--",
        #     lw=1.5,
        #     label=f"client {cid} γ-fit",
        #     color=client_colors[cid],
        # )
    except RuntimeError as exc:
        print(f"client {cid} γ-fit failed: {exc}")


if all_durations:
    t_grid = np.linspace(min(all_durations), max(all_durations), 400)
    prod_fixed = np.ones_like(t_grid)

    for cid, gamma_hat in gamma_params.items():
        prod_fixed *= logistic_gen_cdf(t_grid, gamma_hat)

    # ------------------------------------------------------------------
    # 3)  Product curves
    # ------------------------------------------------------------------
    t_min = min(all_durations)
    t_max = max(all_durations)
    t_grid = np.linspace(t_min, t_max, 400)

    # (a) product of original 2-param logistics
    prod_original = np.ones_like(t_grid)
    for theta_hat, k_hat in client_params.values():
        prod_original *= logistic_cdf(t_grid, theta_hat, k_hat)

    # (b) product of fixed-k, γ-fitted curves
    prod_gamma = np.ones_like(t_grid)
    for cid, gamma_hat in gamma_params.items():
        prod_gamma *= logistic_gen_cdf(t_grid, gamma_hat)

    # (c) product of t location-scale fits
    prod_t = np.ones_like(t_grid)
    for cid, params in t_params.items():
        prod_t *= t.cdf((t_grid - params["loc"]) / params["scale"], params["df"])

    # (d) product of Weibull fits
    prod_weibull = np.ones_like(t_grid)
    for cid, params in weibull_params.items():
        prod_weibull *= weibull_min.cdf(
            t_grid,
            params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )

    # (e) product of empirical CDF stair-steps  (optional)
    prod_empirical = np.ones_like(t_grid)
    for cid, recs in client_durations.items():
        times = np.array([r["duration"] for r in recs if r["duration"] <= MAX_DURATION])
        times.sort()
        if times.size == 0:
            continue
        y_emp = np.arange(1, len(times) + 1) / len(times)
        F_emp = np.interp(t_grid, times, y_emp, left=0, right=1)
        prod_empirical *= F_emp

    # ------------------------------------------------------------------
    # 4)  Plot
    # ------------------------------------------------------------------
    plt.plot(t_grid, prod_original, label="product of original logistic fits", lw=2, color='black')
    # plt.plot(t_grid, prod_gamma, "-", label=f"product with k={K_FIXED}, γ-fitted", lw=2, color='red')
    # plt.plot(t_grid, prod_t, "-", label="product of t fits", lw=2, color='green')
    # plt.plot(t_grid, prod_weibull, "-", label="product of Weibull fits", lw=2, color='blue')
else:
    t_grid = np.array([])
    prod_fixed = np.array([])

df["_duration"] = df["client_train_s"] + df["client_to_server_ms"] / 1000.0 + df["server_to_client_ms"] / 1000.0

# 尝试自动识别“轮次”的列名
_round_candidates = ["round", "server_round", "round_idx", "global_round", "comm_round", "epoch", "iteration"]
_round_col = next((c for c in _round_candidates if c in df.columns), None)

if _round_col is not None:
    # 每轮内先过滤掉 >35s 的，再取最大
    per_round_time = (
        df.groupby(_round_col)["_duration"]
        .apply(lambda x: x[x <= 430.0].max() if any(x <= 430.0) else np.nan)
        .dropna()
        .sort_index()
        .values
    )
    print("max per-round time (≤430s):", per_round_time)

    if len(per_round_time) > 0:
        # 画经验CDF
        per_round_time_sorted = np.sort(per_round_time)
        per_round_cdf = np.arange(1, len(per_round_time_sorted) + 1) / float(len(per_round_time_sorted))

        plt.plot(
            per_round_time_sorted,
            per_round_cdf,
            "-", linewidth=2.5,
            label="Actual per-round max (≤35s)"
        )


plt.xlabel("Time (s)")
plt.ylabel("Cumulative Probability")
plt.title("server to client")
plt.grid(alpha=0.3)
# plt.legend()
plt.ylim(0, 1.1)
# plt.xlim(left=25, right=35)
plt.tight_layout()
plt.savefig("server_20251022_171102.png")
plt.show()
