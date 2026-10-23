# -*- coding: utf-8 -*-
"""
Step 6b: 稳健性补充
1) 模拟法计算2014年9月月度总量的95%预测区间 (比每日区间加总更合理)
2) 赎回备选模型 SARIMA(3,1,3)(0,1,1)7 的9月预测对比
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings, os

warnings.filterwarnings("ignore")
BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")

df = pd.read_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig", index_col=0, parse_dates=True)

SPECS = {
    ("purchase", "rec"): dict(order=(1,1,2), seasonal_order=(2,1,1,7), trend="n"),
    ("redeem", "rec"): dict(order=(1,1,1), seasonal_order=(0,1,1,7), trend="n"),
    ("redeem", "alt"): dict(order=(3,1,3), seasonal_order=(0,1,1,7), trend="n"),
    ("purchase", "base"): dict(order=(3,1,3), seasonal_order=(0,0,0,0), trend="n"),
    ("redeem", "base"): dict(order=(3,1,3), seasonal_order=(0,0,0,0), trend="c"),
}

H = 30
rows = []
sims_store = {}
for (col, key), spec in SPECS.items():
    y = np.log(df[col])
    m = SARIMAX(y, order=spec["order"], seasonal_order=spec["seasonal_order"],
                trend=spec["trend"], enforce_stationarity=True, enforce_invertibility=True)
    r = m.fit(disp=False)
    fc = r.get_forecast(steps=H)
    mean = np.exp(fc.predicted_mean.values)
    ci = fc.conf_int(alpha=0.05)
    lo, hi = np.exp(ci.iloc[:, 0].values), np.exp(ci.iloc[:, 1].values)

    # 模拟月度总量分布
    try:
        sim = r.simulate(nsimulations=H, anchor="end", repetitions=5000)
        # sim 可能是 DataFrame( repetitions ) 或 ndarray
        sim_arr = np.asarray(sim)
        if sim_arr.ndim == 2 and sim_arr.shape[0] == H:
            tot = np.exp(sim_arr).sum(axis=0)
        elif sim_arr.ndim == 2 and sim_arr.shape[1] == H:
            tot = np.exp(sim_arr).sum(axis=1)
        else:
            tot = None
        if tot is not None:
            m_lo, m_hi = np.percentile(tot, [2.5, 97.5])
        else:
            m_lo, m_hi = lo.sum(), hi.sum()
    except Exception as e:
        m_lo, m_hi = lo.sum(), hi.sum()

    cn = "申购" if col == "purchase" else "赎回"
    label = {"rec": "推荐", "alt": "备选", "base": "ARIMA对照"}[key]
    rows.append({
        "序列": cn, "类型": label,
        "阶数": f"({spec['order'][0]},{spec['order'][1]},{spec['order'][2]})" +
                (f"({spec['seasonal_order'][0]},{spec['seasonal_order'][1]},{spec['seasonal_order'][2]})7" if any(spec['seasonal_order'][:3]) else ""),
        "AIC": r.aic,
        "9月总量预测(亿元)": mean.sum() / 1e8,
        "月度区间下限(亿元)": m_lo / 1e8,
        "月度区间上限(亿元)": m_hi / 1e8,
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "step6b_robustness.csv"), index=False, encoding="utf-8-sig")
print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
