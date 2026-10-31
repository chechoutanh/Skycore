"""Streamlit dashboard for SkyScore.  Run:  streamlit run app.py"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from skyscore.astro import TARGETS  # noqa: E402
from skyscore.pipeline import build_table  # noqa: E402
from skyscore.scoring import Weights, best_window, pareto_front, robust_best_hours  # noqa: E402
from skyscore.seeing_model import train_default  # noqa: E402

st.set_page_config(page_title="SkyScore", page_icon="🔭", layout="wide")


@st.cache_resource
def get_model():
    return train_default()


INTRO_EN = r"""
### What is SkyScore?

SkyScore answers a practical question every astronomy club asks:
**"When tonight should we point the telescope at this target?"**

For every hour of the day it combines four sources of information:

| Criterion | Source | Meaning |
|---|---|---|
| 🌌 Target altitude | Astropy (offline celestial mechanics) | Higher means less atmosphere in the way (lower airmass) |
| 🌙 Moon | Astropy | A bright Moon that is up and close to the target brightens the sky |
| ☁️ Clouds | Open-Meteo (free forecast) | Clouds block the view |
| ✨ Seeing | Random Forest model | Star "twinkle" caused by atmospheric turbulence (arcsec) |

Each criterion is normalized to a **0 (bad) – 1 (good)** scale and combined into a score

$$S(t) = \mathbb{1}[\text{observable}(t)]\sum_i w_i\,g_i(t), \qquad \sum_i w_i = 1.$$

An hour counts as *observable* only when the Sun is below −18° (astronomical night)
and the target is above 15°.

Besides the total score, the app also reports:

* **Best window**: the *L* consecutive hours with the highest mean score.
* **Pareto set**: hours not beaten by any other hour in *all four* criteria, showing trade-offs without choosing weights.
* **Robustness**: 500 random weight sets are tried; if most pick the same hour, the recommendation is reliable.

> ⚠️ **Data honesty:** the seeing model is currently trained on *proxy labels*
> (an approximate physical formula), not real measurements. The MAE/R² shown in the Planner tab
> only prove the pipeline works; they are **not** evidence of real-world forecasting skill.
"""

INTRO_VI = r"""
### SkyScore là gì?

SkyScore trả lời một câu hỏi rất thực tế của mọi CLB thiên văn:
**"Tối nay, mấy giờ thì nên hướng kính thiên văn vào mục tiêu này?"**

Ứng dụng kết hợp bốn nguồn thông tin cho từng giờ trong ngày:

| Thành phần | Nguồn | Ý nghĩa |
|---|---|---|
| 🌌 Độ cao mục tiêu | Astropy (tính toán thiên văn, không cần mạng) | Càng cao càng ít khí quyển cản (airmass nhỏ) |
| 🌙 Mặt Trăng | Astropy | Trăng sáng, đang mọc và gần mục tiêu làm trời sáng lên |
| ☁️ Mây | Open-Meteo (dự báo miễn phí) | Mây che thì không quan sát được |
| ✨ Seeing | Mô hình Random Forest | Độ "rung" của ảnh sao do nhiễu loạn khí quyển (arcsec) |

Mỗi tiêu chí được chuẩn hoá về thang **0 (xấu) – 1 (tốt)** rồi gộp thành điểm

$$S(t) = \mathbb{1}[\text{quan sát được}(t)]\sum_i w_i\,g_i(t), \qquad \sum_i w_i = 1.$$

Một giờ chỉ được tính là *quan sát được* khi Mặt Trời thấp hơn −18° (đêm thiên văn)
và mục tiêu cao hơn 15°.

Ngoài điểm tổng, ứng dụng còn đưa ra:

* **Cửa sổ tốt nhất**: khoảng *L* giờ liên tiếp có điểm trung bình cao nhất.
* **Tập Pareto**: các giờ không bị giờ nào khác vượt trội ở *cả bốn* tiêu chí — xem được sự đánh đổi mà không cần chọn trọng số.
* **Độ bền vững**: thử 500 bộ trọng số ngẫu nhiên; nếu đa số vẫn chọn cùng một giờ thì khuyến nghị đáng tin.

> ⚠️ **Lưu ý trung thực về dữ liệu:** mô hình seeing hiện được huấn luyện trên *nhãn giả lập*
> (công thức vật lý gần đúng), chưa phải số đo thật. Chỉ số MAE/R² hiển thị ở tab Planner chỉ
> chứng minh pipeline hoạt động, **không** chứng minh khả năng dự báo thực tế.
"""

GUIDE_EN = """
### How to use

**1. Pick a site and target (left sidebar)**

* **Latitude / Longitude**: your observing site. Default is Ho Chi Minh City (10.78°, 106.70°).
  Right-click a spot on Google Maps to copy its coordinates.
* **Target**: the object to observe (6 deep-sky objects + 4 planets).
* **Date (UTC)**: the date in UTC. The grid covers 24 hours, 00:00–23:00 UTC.
* **Offline demo**: use when there is no internet; the app then uses *synthetic* weather and says so.

**2. Adjust the weights**

Drag the sliders to tell SkyScore what matters to you. They don't need to sum to 1;
the app normalizes them. Examples:

| Goal | Suggestion |
|---|---|
| Imaging faint galaxies (deep-sky) | Raise *Moon darkness* and *Clear sky* |
| Planetary observing | Raise *Seeing* and *Altitude*, lower *Moon darkness* |
| Public outreach night | Raise *Clear sky*, choose a longer *Window length* |

**Window length** is how many consecutive hours you plan to observe.

**3. Read the results (🔭 Planner tab)**

* **Best window (UTC)**: when to observe. **Vietnam time = UTC + 7**
  (e.g. 13:00 UTC = 8 pm).
* **Mean score**: mean score of that window (0–1). Above ~0.7 is good.
* **Robustness**: share of random weight sets that pick the same best hour. Higher means more certain.
* **Score over time**: zero for hours that are not observable (sky too bright or target too low).
* **Criteria**: the four criterion curves explain *why* an hour scores high or low.
* **Pareto-optimal hours** and **Raw table**: expand for detailed numbers.

**4. FAQ**

* *"Target never observable tonight"*: the target never gets high enough during astronomical night
  on this date; try another date or target.
* *Weather: Synthetic*: the real forecast could not be fetched (no internet or API error); results are illustrative only.
* *How far ahead?* Open-Meteo covers about the next 4 days; beyond that the app falls back to synthetic data.
"""

GUIDE_VI = """
### Hướng dẫn sử dụng

**1. Chọn địa điểm và mục tiêu (thanh bên trái)**

* **Vĩ độ / Kinh độ**: toạ độ nơi quan sát. Mặc định là TP. Hồ Chí Minh (10.78°, 106.70°).
  Lấy toạ độ bằng cách nhấp chuột phải trên Google Maps.
* **Mục tiêu**: thiên thể muốn quan sát (6 thiên thể sâu + 4 hành tinh).
* **Ngày (UTC)**: ngày tính theo giờ UTC. Lưới tính toán là 24 giờ, từ 00:00 đến 23:00 UTC.
* **Chế độ offline**: bật khi không có mạng; ứng dụng dùng thời tiết *giả lập* và ghi rõ trên giao diện.

**2. Điều chỉnh trọng số**

Kéo các thanh trượt để nói cho SkyScore biết điều gì quan trọng với bạn. Tổng không cần bằng 1,
ứng dụng tự chuẩn hoá. Ví dụ:

| Mục đích | Gợi ý |
|---|---|
| Chụp ảnh thiên hà mờ (deep-sky) | Tăng *Trăng tối* và *Trời quang* |
| Quan sát hành tinh | Tăng *Seeing* và *Độ cao*, giảm *Trăng tối* |
| Buổi quan sát công chúng | Tăng *Trời quang*, chọn *Độ dài cửa sổ* dài hơn |

**Độ dài cửa sổ** là số giờ liên tiếp bạn định quan sát.

**3. Đọc kết quả (tab 🔭 Lập lịch)**

* **Cửa sổ tốt nhất (UTC)**: khung giờ nên quan sát. **Giờ Việt Nam = UTC + 7**
  (ví dụ 13:00 UTC = 20:00 tối).
* **Điểm trung bình**: điểm trung bình của khung giờ đó (0–1). Trên ~0.7 là tốt.
* **Độ bền vững**: tỉ lệ bộ trọng số ngẫu nhiên cùng chọn giờ tốt nhất. Càng cao càng chắc chắn.
* **Điểm theo thời gian**: điểm bằng 0 ở những giờ không quan sát được (trời sáng hoặc mục tiêu quá thấp).
* **Các tiêu chí**: bốn đường tiêu chí giúp hiểu *vì sao* một giờ được điểm cao hay thấp.
* **Các giờ tối ưu Pareto** và **Bảng dữ liệu thô**: mở ra để xem số liệu chi tiết.

**4. Câu hỏi thường gặp**

* *"Mục tiêu không quan sát được đêm nay"*: mục tiêu không lên đủ cao trong đêm thiên văn ở ngày này;
  thử ngày khác hoặc mục tiêu khác.
* *Thời tiết: Synthetic*: không lấy được dự báo thật (mất mạng hoặc API lỗi); kết quả chỉ để minh hoạ.
* *Dự báo xa nhất?* Open-Meteo trả về khoảng 4 ngày tới; xa hơn ứng dụng chuyển sang dữ liệu giả lập.
"""

TEXT = {
    "en": {
        "caption": "Multi-objective observing-window planner: astronomy + weather + ML seeing forecast",
        "site": "Site & target",
        "lat": "Latitude (°)",
        "lon": "Longitude (°)",
        "target": "Target",
        "date": "Date (UTC)",
        "offline": "Offline demo (synthetic weather)",
        "weights": "Weights",
        "w_alt": "Altitude",
        "w_moon": "Moon darkness",
        "w_cloud": "Clear sky",
        "w_see": "Seeing",
        "window": "Window length (h)",
        "tabs": ["🔭 Planner", "ℹ️ About", "📖 Guide"],
        "intro": INTRO_EN,
        "guide": GUIDE_EN,
        "zero_weights": "Set at least one weight above 0.",
        "computing": "Computing…",
        "best": "Best window (UTC)",
        "mean": "Mean score",
        "robust": "Robustness of best hour",
        "robust_help": "Share of random weight sets that pick the same best hour",
        "never": "Target never observable tonight (twilight / too low).",
        "score_time": "Score over time",
        "criteria": "Criteria (0 = bad, 1 = good)",
        "seeing": "Predicted seeing (arcsec)",
        "pareto": "Pareto-optimal hours",
        "raw": "Raw table",
        "info": "Weather: **{ws}** · Seeing labels: **{ls}** · CV MAE {mae:.2f}″, R² {r2:.2f}. "
        "Seeing model trained on proxy labels unless replaced with real observing logs.",
    },
    "vi": {
        "caption": "Lập lịch quan sát thiên văn đa mục tiêu: thiên văn + thời tiết + dự báo seeing bằng học máy",
        "site": "Địa điểm & mục tiêu",
        "lat": "Vĩ độ (°)",
        "lon": "Kinh độ (°)",
        "target": "Mục tiêu",
        "date": "Ngày (UTC)",
        "offline": "Chế độ offline (thời tiết giả lập)",
        "weights": "Trọng số",
        "w_alt": "Độ cao",
        "w_moon": "Trăng tối",
        "w_cloud": "Trời quang",
        "w_see": "Seeing",
        "window": "Độ dài cửa sổ (giờ)",
        "tabs": ["🔭 Lập lịch", "ℹ️ Giới thiệu", "📖 Hướng dẫn"],
        "intro": INTRO_VI,
        "guide": GUIDE_VI,
        "zero_weights": "Hãy đặt ít nhất một trọng số lớn hơn 0.",
        "computing": "Đang tính toán…",
        "best": "Cửa sổ tốt nhất (UTC)",
        "mean": "Điểm trung bình",
        "robust": "Độ bền vững của giờ tốt nhất",
        "robust_help": "Tỉ lệ bộ trọng số ngẫu nhiên cùng chọn giờ tốt nhất",
        "never": "Mục tiêu không quan sát được đêm nay (trời chưa đủ tối / mục tiêu quá thấp).",
        "score_time": "Điểm theo thời gian",
        "criteria": "Các tiêu chí (0 = xấu, 1 = tốt)",
        "seeing": "Seeing dự đoán (arcsec)",
        "pareto": "Các giờ tối ưu Pareto",
        "raw": "Bảng dữ liệu thô",
        "info": "Thời tiết: **{ws}** · Nhãn seeing: **{ls}** · CV MAE {mae:.2f}″, R² {r2:.2f}. "
        "Mô hình seeing học từ nhãn giả lập cho tới khi được thay bằng nhật ký quan sát thật.",
    },
}

with st.sidebar:
    lang = st.radio(
        "🌐 Language / Ngôn ngữ",
        ["en", "vi"],
        format_func={"en": "English", "vi": "Tiếng Việt"}.get,
        horizontal=True,
    )
    T = TEXT[lang]

    st.header(T["site"])
    lat = st.number_input(T["lat"], -90.0, 90.0, 10.78, 0.01, key="lat")
    lon = st.number_input(T["lon"], -180.0, 180.0, 106.70, 0.01, key="lon")
    target = st.selectbox(T["target"], TARGETS, key="target")
    date = st.date_input(T["date"], pd.Timestamp.utcnow().date(), key="date")
    offline = st.checkbox(T["offline"], value=False, key="offline")

    st.header(T["weights"])
    w_alt = st.slider(T["w_alt"], 0.0, 1.0, 0.35, 0.05, key="w_alt")
    w_moon = st.slider(T["w_moon"], 0.0, 1.0, 0.20, 0.05, key="w_moon")
    w_cloud = st.slider(T["w_cloud"], 0.0, 1.0, 0.30, 0.05, key="w_cloud")
    w_see = st.slider(T["w_see"], 0.0, 1.0, 0.15, 0.05, key="w_see")
    window = st.slider(T["window"], 1, 6, 3, key="window")

st.title("🔭 SkyScore")
st.caption(T["caption"])

tab_plan, tab_intro, tab_guide = st.tabs(T["tabs"])

# Static tabs are filled first so they still render if the planner calls st.stop().
with tab_intro:
    st.markdown(T["intro"])
with tab_guide:
    st.markdown(T["guide"])

with tab_plan:
    if w_alt + w_moon + w_cloud + w_see == 0:
        st.warning(T["zero_weights"])
        st.stop()

    weights = Weights(w_alt, w_moon, w_cloud, w_see)
    with st.spinner(T["computing"]):
        df, meta = build_table(lat, lon, target, pd.Timestamp(date), weights, offline, get_model())

    c1, c2, c3 = st.columns(3)
    if df["observable"].any():
        start, end, mean_s = best_window(df, window)
        c1.metric(T["best"], f"{start:%H:%M} – {end + pd.Timedelta(hours=1):%H:%M}")
        c2.metric(T["mean"], f"{mean_s:.2f}")
        freq = robust_best_hours(df)
        c3.metric(T["robust"], f"{freq.max():.0%}", help=T["robust_help"])
    else:
        c1.error(T["never"])

    st.subheader(T["score_time"])
    st.line_chart(df[["score"]])

    left, right = st.columns(2)
    with left:
        st.subheader(T["criteria"])
        st.line_chart(df[["g_altitude", "g_moon", "g_cloud", "g_seeing"]])
    with right:
        st.subheader(T["seeing"])
        st.line_chart(df[["seeing_arcsec"]])

    with st.expander(T["pareto"]):
        st.dataframe(pareto_front(df).round(2))
    with st.expander(T["raw"]):
        st.dataframe(df.round(2))

    st.info(
        T["info"].format(
            ws=meta["weather_source"],
            ls=meta["seeing_label_source"],
            mae=meta["seeing_cv"].get("cv_mae_arcsec", float("nan")),
            r2=meta["seeing_cv"].get("cv_r2", float("nan")),
        )
    )
