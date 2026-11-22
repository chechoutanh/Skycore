"""Streamlit dashboard for SkyScore.  Run:  streamlit run app.py"""

import sys
from datetime import time as dtime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from skyscore.astro import TARGETS  # noqa: E402
from skyscore.evaluation import add_predictions, evaluate, load_logs  # noqa: E402
from skyscore.pipeline import build_table  # noqa: E402
from skyscore.scoring import Weights, best_window, pareto_front, robust_best_hours  # noqa: E402
from skyscore.seeing_model import train_default  # noqa: E402

st.set_page_config(page_title="SkyScore", page_icon="🔭", layout="wide")


DATA = Path(__file__).parent / "data"
EVENTS, OBS = DATA / "events.csv", DATA / "observations.csv"


@st.cache_resource
def get_model():
    return train_default()


def save_log(event: dict, obs: dict) -> str:
    """Append one observed hour to the logs; reuse the event when date, target and site match."""
    ev = pd.read_csv(EVENTS)
    same = ev[
        (ev["date"] == event["date"]) & (ev["target"] == event["target"])
        & ((ev["lat"] - event["lat"]).abs() < 0.01) & ((ev["lon"] - event["lon"]).abs() < 0.01)
    ]
    if len(same):
        eid = same["event_id"].iloc[0]
    else:
        n = pd.to_numeric(ev["event_id"].str[1:], errors="coerce").max()
        eid = f"E{int(0 if pd.isna(n) else n) + 1:02d}"
        pd.DataFrame([{**event, "event_id": eid}]).reindex(columns=ev.columns).to_csv(EVENTS, mode="a", header=False, index=False)
    cols = pd.read_csv(OBS, nrows=0).columns
    pd.DataFrame([{**obs, "event_id": eid}]).reindex(columns=cols).to_csv(OBS, mode="a", header=False, index=False)
    return eid


def night_chart(df: pd.DataFrame, cols: list[str], labels: list[str]):
    """Hourly lines on local-time labels; hours that are not observable are shaded grey."""
    d = df[cols].reset_index(drop=True).assign(time=labels, observable=df["observable"].to_numpy())
    x = alt.X("time:N", sort=labels, title=None, axis=alt.Axis(labelAngle=0, values=labels[::2]))
    shade = alt.Chart(d[~d["observable"]]).mark_rect(color="gray", opacity=0.15).encode(x=x)
    lines = (
        alt.Chart(d.melt(["time", "observable"], cols, "series", "value"))
        .mark_line(point=True)
        .encode(
            x=x,
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format=".2~f")),
            color=alt.Color("series:N", title=None, scale=alt.Scale(scheme="category10"), legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=["time", "series", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    return (shade + lines).properties(height=280)


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

An hour counts as *observable* only when the target is above 15° and the sky is dark enough for it:
Sun below −18° for galaxies, nebulae and clusters, −12° for naked-eye constellations, −6° for bright planets.
The Sun itself is a daytime target. Moonlight does not count against the Sun or the Moon.

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

Một giờ chỉ được tính là *quan sát được* khi mục tiêu cao hơn 15° và trời đủ tối cho mục tiêu đó:
Mặt Trời thấp hơn −18° với thiên hà, tinh vân, cụm sao; −12° với chòm sao nhìn bằng mắt thường; −6° với hành tinh sáng.
Bản thân Mặt Trời là mục tiêu ban ngày. Ánh trăng không bị tính là bất lợi khi mục tiêu là Mặt Trời hoặc Mặt Trăng.

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
* **Target**: the object to observe: 6 deep-sky objects, the Moon, 7 planets, the moons of Jupiter and Saturn, the Sun and 10 well-known constellations. A planet's moons stay within 0.2° of it, so they use the planet's position.
* **Date (evening)**: the local date of the evening you plan to observe. The grid covers 24 hours from local noon (from midnight for the Sun); times are shown in local time (see *UTC offset*).
* **UTC offset (h)**: your time zone, e.g. 7 for Vietnam, −5 for New York (winter).
* **Offline demo**: use when there is no internet; the app then uses *synthetic* weather and says so.

**2. Choose what you want to observe**

Pick a goal in **What do you want to observe?** and SkyScore sets the weights for you.
Want more control? Open **Fine-tune weights (advanced)** and drag the sliders to tell SkyScore what matters to you. They don't need to sum to 1;
the app normalizes them. Examples:

| Goal | Suggestion |
|---|---|
| Imaging faint galaxies (deep-sky) | Raise *Moon darkness* and *Clear sky* |
| Planetary observing | Raise *Seeing* and *Altitude*, lower *Moon darkness* |
| Public outreach night | Raise *Clear sky*, choose a longer *Window length* |

**Window length** is how many consecutive hours you plan to observe.

**3. Read the results (🔭 Planner tab)**

* **Best window**: when to observe, in local time. Grey bands on the charts are hours when the target is not observable.
* **Mean score**: mean score of that window (0–1). Above ~0.7 is good.
* **Confidence**: share of random weight sets that pick the same best hour. Higher means more certain.
* **Score over time**: zero for hours that are not observable (sky too bright or target too low).
* **Criteria**: the four criterion curves explain *why* an hour scores high or low.
* **Other good hours to consider**: hours that are the best at something (e.g. clearest sky), handy if your priorities differ. **Raw table**: every number.

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
* **Mục tiêu**: thiên thể muốn quan sát: 6 thiên thể sâu, Mặt Trăng, 7 hành tinh, các vệ tinh của Sao Mộc và Sao Thổ, Mặt Trời và 10 chòm sao phổ biến. Vệ tinh luôn ở cách hành tinh dưới 0,2°, nên dùng vị trí của hành tinh.
* **Ngày (buổi tối)**: ngày địa phương của buổi tối định quan sát. Lưới tính toán là 24 giờ tính từ trưa địa phương (từ nửa đêm nếu mục tiêu là Mặt Trời); mọi giờ hiển thị theo giờ địa phương (xem *Múi giờ*).
* **Múi giờ (UTC+)**: ví dụ 7 cho Việt Nam, −5 cho New York (mùa đông).
* **Chế độ offline**: bật khi không có mạng; ứng dụng dùng thời tiết *giả lập* và ghi rõ trên giao diện.

**2. Chọn mục đích quan sát**

Chọn ở ô **Bạn muốn quan sát gì?**, SkyScore sẽ tự đặt trọng số.
Muốn chỉnh kỹ hơn? Mở **Tinh chỉnh trọng số (nâng cao)** và kéo các thanh trượt để nói cho SkyScore biết điều gì quan trọng với bạn. Tổng không cần bằng 1,
ứng dụng tự chuẩn hoá. Ví dụ:

| Mục đích | Gợi ý |
|---|---|
| Chụp ảnh thiên hà mờ (deep-sky) | Tăng *Trăng tối* và *Trời quang* |
| Quan sát hành tinh | Tăng *Seeing* và *Độ cao*, giảm *Trăng tối* |
| Buổi quan sát công chúng | Tăng *Trời quang*, chọn *Độ dài cửa sổ* dài hơn |

**Độ dài cửa sổ** là số giờ liên tiếp bạn định quan sát.

**3. Đọc kết quả (tab 🔭 Lập lịch)**

* **Cửa sổ tốt nhất**: khung giờ nên quan sát, theo giờ địa phương. Vùng tô xám trên biểu đồ là những giờ không quan sát được.
* **Điểm trung bình**: điểm trung bình của khung giờ đó (0–1). Trên ~0.7 là tốt.
* **Độ tin cậy**: tỉ lệ bộ trọng số ngẫu nhiên cùng chọn giờ tốt nhất. Càng cao càng chắc chắn.
* **Điểm theo thời gian**: điểm bằng 0 ở những giờ không quan sát được (trời sáng hoặc mục tiêu quá thấp).
* **Các tiêu chí**: bốn đường tiêu chí giúp hiểu *vì sao* một giờ được điểm cao hay thấp.
* **Các giờ tốt khác đáng cân nhắc**: những giờ tốt nhất ở một mặt nào đó (ví dụ trời quang nhất), hữu ích khi ưu tiên của bạn khác. **Bảng dữ liệu thô**: toàn bộ số liệu.

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
        "date": "Date (evening)",
        "offline": "Offline demo (synthetic weather)",
        "weights": "Weights",
        "w_alt": "Height in sky",
        "w_moon": "Moon darkness",
        "w_cloud": "Clear sky",
        "w_see": "Steady air (seeing)",
        "window": "Window length (h)",
        "tabs": ["🔭 Planner", "📝 Observing log", "ℹ️ About", "📖 Guide"],
        "log_intro": "After each session, note how the sky **really** was. With 20–30 sessions we can measure how accurate SkyScore is. "
        "Log bad hours too (clouds, blurry stars): they matter just as much.",
        "log_from_sidebar": "Site, target and date come from the sidebar: **{target}**, {date}, ({lat:.2f}, {lon:.2f}), UTC{tz:+g}.",
        "log_site": "Place name", "log_club": "Club / observer",
        "log_time": "Time observed (local)", "log_duration": "How long (h)",
        "log_seeing": "How steady were the stars?", "log_seeing_help": "1 = boiling, very blurry · 5 = rock steady, sharp",
        "log_quality": "Overall, how good was it?", "log_quality_help": "1 = useless · 5 = excellent",
        "log_cloud": "Sky covered by cloud (%)",
        "log_wind": "Wind", "wind": {"calm": "Calm", "light": "Light", "strong": "Strong (shook the telescope)"},
        "log_notes": "Notes (optional)", "log_submit": "💾 Save this entry",
        "log_saved": "Saved to session {eid}. Logging another hour of the same night? Change the time and save again.",
        "log_cloud_warn": "Note: on a hosted app (e.g. Streamlit Cloud) saved entries can disappear on restart. Download the CSV files below to keep them.",
        "log_table": "Logged sessions ({n} entries)",
        "log_dl_events": "⬇️ events.csv", "log_dl_obs": "⬇️ observations.csv",
        "log_eval": "📊 How accurate is SkyScore so far?",
        "log_eval_err": "Could not evaluate: {e}",
        "ev_rating": "Real rating at SkyScore's best hour", "ev_rating_delta": "{d:+.2f} vs a random logged hour",
        "ev_cloud": "Cloud forecast error", "ev_cloud_delta": "{d:+.0f} pts vs guessing the average",
        "ev_seeing": "Seeing agreement",
        "ev_seeing_help": "Rank correlation between predicted and logged seeing: 1 = perfect, 0 = no relation, negative = backwards.",
        "ev_note": "Based on **{events}** sessions / **{hours}** logged hours. Under ~20 sessions these numbers are mostly noise. "
        "The 'best hour' check only means something for nights where you logged several hours.",
        "ev_missing": "Not enough data yet",
        "intro": INTRO_EN,
        "guide": GUIDE_EN,
        "zero_weights": "Set at least one weight above 0.",
        "computing": "Computing…",
        "best": "Best window (local time)",
        "tz": "UTC offset (h)",
        "mean": "Mean score",
        "robust": "Confidence",
        "robust_help": "How often the same best hour wins when we try 500 random priority settings. Higher = more certain.",
        "never": "Target not observable on this date (sky not dark enough / target too low).",
        "sun_warning": "Never look at the Sun through a telescope, binoculars or with the naked eye without a certified solar filter.",
        "score_time": "Score over time",
        "criteria": "Criteria (0 = bad, 1 = good)",
        "seeing": "Predicted seeing (arcsec)",
        "pareto": "Other good hours to consider (trade-offs)",
        "pareto_help": "Each hour below is **unbeatable in at least one way**: no other hour is better on everything at once. "
        "Use it if your priorities differ from the sliders: e.g. pick the hour that is best for *Clear sky* if clouds worry you most. "
        "⭐ = inside the recommended window.",
        "col_time": "Time", "col_score": "Score", "col_alt": "Height in sky", "col_cloud": "Clouds",
        "col_seeing": "Seeing", "col_moon": "Moon darkness", "col_best": "Best hour for", "mix": "Good balance",
        "verdict": "👉 **Go out {start} – {end}.** Conditions look **{q}** (score {s:.2f} / 1).",
        "quality": ["poor", "fair", "good"],
        "goal": "What do you want to observe?",
        "goal_help": "Picks sensible weights for you. Fine-tune them below if you like.",
        "presets": {"balanced": "Balanced (default)", "deep": "Faint galaxies & nebulae", "planets": "Planets & the Moon", "public": "Public / casual night"},
        "fine_tune": "Fine-tune weights (advanced)",
        "names": {"g_altitude": "Height in sky", "g_moon": "Moon darkness", "g_cloud": "Clear sky", "g_seeing": "Steady air"},
        "seeing_help": "Lower is better: how much stars blur from air turbulence (arcseconds).",
        "date_help": "The evening you plan to observe.",
        "date_text": "…or type / paste a date",
        "date_bad": "Could not read that date. Try 22/11/2026 or 2026-11-22.",
        "tz_help": "Hours ahead of UTC. Vietnam = 7.",
        "lat_help": "Right-click a spot in Google Maps to copy its coordinates.",
        "window_help": "How many hours in a row you plan to observe.",
        "raw": "Raw table",
        "save": "💾 Save tonight's predictions (CSV)",
        "info": "Weather: **{ws}** · Seeing labels: **{ls}** · CV MAE {mae:.2f}″, R² {r2:.2f}. "
        "Seeing model trained on proxy labels unless replaced with real observing logs.",
    },
    "vi": {
        "caption": "Lập lịch quan sát thiên văn đa mục tiêu: thiên văn + thời tiết + dự báo seeing bằng học máy",
        "site": "Địa điểm & mục tiêu",
        "lat": "Vĩ độ (°)",
        "lon": "Kinh độ (°)",
        "target": "Mục tiêu",
        "date": "Ngày (buổi tối)",
        "offline": "Chế độ offline (thời tiết giả lập)",
        "weights": "Trọng số",
        "w_alt": "Độ cao",
        "w_moon": "Trăng tối",
        "w_cloud": "Trời quang",
        "w_see": "Khí quyển ổn định (seeing)",
        "window": "Độ dài cửa sổ (giờ)",
        "tabs": ["🔭 Lập lịch", "📝 Nhật ký quan sát", "ℹ️ Giới thiệu", "📖 Hướng dẫn"],
        "log_intro": "Sau mỗi buổi, ghi lại bầu trời **thực tế** ra sao. Có khoảng 20–30 buổi là đo được SkyScore chuẩn đến đâu. "
        "Hãy ghi cả những giờ xấu (mây, sao nhoè): chúng quan trọng không kém.",
        "log_from_sidebar": "Địa điểm, mục tiêu và ngày lấy từ thanh bên: **{target}**, {date}, ({lat:.2f}, {lon:.2f}), UTC{tz:+g}.",
        "log_site": "Tên địa điểm", "log_club": "CLB / người quan sát",
        "log_time": "Giờ quan sát (địa phương)", "log_duration": "Kéo dài (giờ)",
        "log_seeing": "Sao có ổn định không?", "log_seeing_help": "1 = rung mạnh, rất nhoè · 5 = đứng yên, rất nét",
        "log_quality": "Đánh giá chung buổi quan sát", "log_quality_help": "1 = không xem được gì · 5 = tuyệt vời",
        "log_cloud": "Phần trời bị mây che (%)",
        "log_wind": "Gió", "wind": {"calm": "Lặng", "light": "Nhẹ", "strong": "Mạnh (làm rung kính)"},
        "log_notes": "Ghi chú (không bắt buộc)", "log_submit": "💾 Lưu mục này",
        "log_saved": "Đã lưu vào buổi {eid}. Muốn ghi thêm giờ khác của cùng đêm? Đổi giờ rồi lưu tiếp.",
        "log_cloud_warn": "Lưu ý: nếu app chạy trên máy chủ (ví dụ Streamlit Cloud), dữ liệu đã lưu có thể mất khi khởi động lại. Hãy tải các file CSV bên dưới để giữ lại.",
        "log_table": "Các buổi đã ghi ({n} mục)",
        "log_dl_events": "⬇️ events.csv", "log_dl_obs": "⬇️ observations.csv",
        "log_eval": "📊 SkyScore chuẩn đến đâu?",
        "log_eval_err": "Không đánh giá được: {e}",
        "ev_rating": "Điểm thực tế ở giờ SkyScore chọn", "ev_rating_delta": "{d:+.2f} so với một giờ bất kỳ",
        "ev_cloud": "Sai số dự báo mây", "ev_cloud_delta": "{d:+.0f} điểm so với đoán trung bình",
        "ev_seeing": "Độ khớp seeing",
        "ev_seeing_help": "Tương quan thứ hạng giữa seeing dự đoán và ghi nhận: 1 = khớp hoàn toàn, 0 = không liên quan, âm = ngược.",
        "ev_note": "Dựa trên **{events}** buổi / **{hours}** giờ đã ghi. Dưới ~20 buổi thì các con số này chủ yếu là nhiễu. "
        "Phép thử 'giờ tốt nhất' chỉ có ý nghĩa với những đêm bạn ghi nhiều giờ.",
        "ev_missing": "Chưa đủ dữ liệu",
        "intro": INTRO_VI,
        "guide": GUIDE_VI,
        "zero_weights": "Hãy đặt ít nhất một trọng số lớn hơn 0.",
        "computing": "Đang tính toán…",
        "best": "Cửa sổ tốt nhất (giờ địa phương)",
        "tz": "Múi giờ (UTC+)",
        "mean": "Điểm trung bình",
        "robust": "Độ tin cậy",
        "robust_help": "Tỉ lệ 500 bộ ưu tiên ngẫu nhiên vẫn chọn cùng giờ tốt nhất. Càng cao càng chắc chắn.",
        "never": "Mục tiêu không quan sát được vào ngày này (trời chưa đủ tối / mục tiêu quá thấp).",
        "sun_warning": "Tuyệt đối không nhìn Mặt Trời qua kính thiên văn, ống nhòm hay bằng mắt thường nếu không có kính lọc Mặt Trời đạt chuẩn.",
        "score_time": "Điểm theo thời gian",
        "criteria": "Các tiêu chí (0 = xấu, 1 = tốt)",
        "seeing": "Seeing dự đoán (arcsec)",
        "pareto": "Các giờ tốt khác đáng cân nhắc (đánh đổi)",
        "pareto_help": "Mỗi giờ dưới đây **không thể bị vượt mặt ở ít nhất một mặt**: không có giờ nào khác tốt hơn ở mọi tiêu chí cùng lúc. "
        "Dùng bảng này nếu ưu tiên của bạn khác với thanh trượt: ví dụ lo mây nhất thì chọn giờ tốt nhất cho *Trời quang*. "
        "⭐ = nằm trong khung giờ được khuyến nghị.",
        "col_time": "Giờ", "col_score": "Điểm", "col_alt": "Độ cao", "col_cloud": "Mây",
        "col_seeing": "Seeing", "col_moon": "Trăng tối", "col_best": "Tốt nhất về", "mix": "Cân bằng tốt",
        "verdict": "👉 **Nên quan sát từ {start} – {end}.** Điều kiện **{q}** (điểm {s:.2f} / 1).",
        "quality": ["kém", "tạm được", "tốt"],
        "goal": "Bạn muốn quan sát gì?",
        "goal_help": "Tự chọn trọng số phù hợp. Có thể tinh chỉnh bên dưới.",
        "presets": {"balanced": "Cân bằng (mặc định)", "deep": "Thiên hà & tinh vân mờ", "planets": "Hành tinh & Mặt Trăng", "public": "Buổi quan sát công chúng"},
        "fine_tune": "Tinh chỉnh trọng số (nâng cao)",
        "names": {"g_altitude": "Độ cao", "g_moon": "Trăng tối", "g_cloud": "Trời quang", "g_seeing": "Khí quyển ổn định"},
        "seeing_help": "Càng thấp càng tốt: mức nhoè của sao do nhiễu loạn không khí (giây cung).",
        "date_help": "Buổi tối bạn định quan sát.",
        "date_text": "…hoặc gõ / dán ngày",
        "date_bad": "Không đọc được ngày này. Thử 22/11/2026 hoặc 2026-11-22.",
        "tz_help": "Số giờ chênh với UTC. Việt Nam = 7.",
        "lat_help": "Nhấp chuột phải trên Google Maps để lấy toạ độ.",
        "window_help": "Số giờ liên tiếp bạn định quan sát.",
        "raw": "Bảng dữ liệu thô",
        "save": "💾 Lưu dữ liệu dự đoán đêm nay (CSV)",
        "info": "Thời tiết: **{ws}** · Nhãn seeing: **{ls}** · CV MAE {mae:.2f}″, R² {r2:.2f}. "
        "Mô hình seeing học từ nhãn giả lập cho tới khi được thay bằng nhật ký quan sát thật.",
    },
}

PRESETS = {  # altitude, moon, cloud, seeing
    "balanced": (0.35, 0.20, 0.30, 0.15),
    "deep": (0.25, 0.35, 0.30, 0.10),
    "planets": (0.35, 0.05, 0.25, 0.35),
    "public": (0.25, 0.10, 0.55, 0.10),
}
W_KEYS = ["w_alt", "w_moon", "w_cloud", "w_see"]
for k, v in zip(W_KEYS, PRESETS["balanced"]):
    st.session_state.setdefault(k, v)


def apply_preset(widget_key):
    st.session_state.preset = st.session_state[widget_key]
    st.session_state.update(zip(W_KEYS, PRESETS[st.session_state.preset]))


def sync_date():
    """Typed or pasted date → the calendar widget. Day first (22/11/2026); ISO 2026-11-22 also works."""
    s = st.session_state.date_text.strip()
    st.session_state.date_bad = False
    if s:
        try:
            st.session_state.date = pd.to_datetime(s, dayfirst=True).date()
        except (ValueError, OverflowError):
            st.session_state.date_bad = True


with st.sidebar:
    lang = st.radio(
        "🌐 Language / Ngôn ngữ",
        ["en", "vi"],
        format_func={"en": "English", "vi": "Tiếng Việt"}.get,
        horizontal=True,
    )
    T = TEXT[lang]

    st.header(T["site"])
    lat = st.number_input(T["lat"], -90.0, 90.0, 10.78, 0.01, key="lat", help=T["lat_help"])
    lon = st.number_input(T["lon"], -180.0, 180.0, 106.70, 0.01, key="lon", help=T["lat_help"])
    target = st.selectbox(T["target"], TARGETS, key="target")
    date = st.date_input(T["date"], key="date", help=T["date_help"])
    st.text_input(T["date_text"], key="date_text", on_change=sync_date, placeholder="22/11/2026")
    if st.session_state.get("date_bad"):
        st.error(T["date_bad"])
    tz = st.number_input(T["tz"], -12.0, 14.0, 7.0, 0.5, key="tz", help=T["tz_help"])
    offline = st.checkbox(T["offline"], value=False, key="offline")

    st.header(T["weights"])
    # One widget per language: Streamlit keeps the old label on screen when format_func changes.
    st.selectbox(
        T["goal"], list(PRESETS), index=list(PRESETS).index(st.session_state.get("preset", "balanced")),
        format_func=T["presets"].get, key=f"preset_{lang}", on_change=apply_preset, args=(f"preset_{lang}",), help=T["goal_help"],
    )
    window = st.slider(T["window"], 1, 6, 3, key="window", help=T["window_help"])
    with st.expander(T["fine_tune"]):
        w_alt = st.slider(T["w_alt"], 0.0, 1.0, step=0.05, key="w_alt")
        w_moon = st.slider(T["w_moon"], 0.0, 1.0, step=0.05, key="w_moon")
        w_cloud = st.slider(T["w_cloud"], 0.0, 1.0, step=0.05, key="w_cloud")
        w_see = st.slider(T["w_see"], 0.0, 1.0, step=0.05, key="w_see", help=T["seeing_help"])

st.title("🔭 SkyScore")
st.caption(T["caption"])

tab_plan, tab_log, tab_intro, tab_guide = st.tabs(T["tabs"])

# Other tabs are filled first so they still render if the planner calls st.stop().
with tab_log:
    st.markdown(T["log_intro"])
    st.caption(T["log_from_sidebar"].format(target=target, date=date, lat=lat, lon=lon, tz=tz))
    with st.form("log", clear_on_submit=True):
        a, b = st.columns(2)
        site = a.text_input(T["log_site"])
        club = b.text_input(T["log_club"])
        obs_time = a.time_input(T["log_time"], dtime(21, 0), step=1800)
        duration = b.number_input(T["log_duration"], 0.5, 12.0, 1.0, 0.5)
        seeing = a.select_slider(T["log_seeing"], [1, 2, 3, 4, 5], 3, help=T["log_seeing_help"])
        quality = b.select_slider(T["log_quality"], [1, 2, 3, 4, 5], 3, help=T["log_quality_help"])
        cloud = a.slider(T["log_cloud"], 0, 100, 0, 10)
        wind = b.radio(T["log_wind"], list(T["wind"]), format_func=T["wind"].get, horizontal=True)
        notes = st.text_input(T["log_notes"])
        if st.form_submit_button(T["log_submit"], type="primary"):
            eid = save_log(
                {"date": f"{date:%Y-%m-%d}", "club": club, "site": site, "lat": lat, "lon": lon, "utc_offset": tz, "target": target},
                {"time": f"{obs_time:%H:%M}", "duration_h": duration, "seeing_value": seeing, "seeing_method": "scale_1_5",
                 "cloud_observed": cloud, "quality_rating": quality, "wind_feel": wind, "notes": notes},
            )
            st.success(T["log_saved"].format(eid=eid))
    st.caption(T["log_cloud_warn"])

    logs = load_logs(EVENTS, OBS)
    with st.expander(T["log_table"].format(n=len(logs))):
        st.dataframe(logs[["event_id", "date", "time", "site", "target", "seeing_value", "cloud_observed", "quality_rating", "notes"]], hide_index=True)
    a, b = st.columns(2)
    a.download_button(T["log_dl_events"], EVENTS.read_bytes(), "events.csv", "text/csv")
    b.download_button(T["log_dl_obs"], OBS.read_bytes(), "observations.csv", "text/csv")

    if st.button(T["log_eval"]):
        try:
            with st.spinner(T["computing"]):
                m = evaluate(add_predictions(logs, get_model(), offline=offline))
        except Exception as e:  # network or bad rows: show it, keep the rest of the app usable
            st.error(T["log_eval_err"].format(e=e))
        else:
            a, b, c = st.columns(3)
            if "rating_at_skyscore_hour" in m:
                a.metric(T["ev_rating"], f"{m['rating_at_skyscore_hour']:.2f} / 5",
                         T["ev_rating_delta"].format(d=m["rating_at_skyscore_hour"] - m["rating_at_random_hour"]))
            if "cloud_mae_forecast" in m:
                b.metric(T["ev_cloud"], f"{m['cloud_mae_forecast']:.0f}%",
                         T["ev_cloud_delta"].format(d=m["cloud_mae_forecast"] - m["cloud_mae_climatology"]), delta_color="inverse")
            rho = m.get("seeing_scale_1_5_spearman")
            c.metric(T["ev_seeing"], T["ev_missing"] if rho is None or pd.isna(rho) else f"{rho:.2f}", help=T["ev_seeing_help"])
            st.info(T["ev_note"].format(events=m["events"], hours=m["hours"]))

with tab_intro:
    st.markdown(T["intro"])
with tab_guide:
    st.markdown(T["guide"])

with tab_plan:
    if w_alt + w_moon + w_cloud + w_see == 0:
        st.warning(T["zero_weights"])
        st.stop()

    if target == "Sun":
        st.warning("⚠️ " + T["sun_warning"])
    weights = Weights(w_alt, w_moon, w_cloud, w_see)
    with st.spinner(T["computing"]):
        df, meta = build_table(lat, lon, target, pd.Timestamp(date), weights, offline, get_model())
    to_local = pd.Timedelta(hours=tz)
    labels = [f"{t + to_local:%H:%M}" for t in df.index]

    c1, c2, c3 = st.columns(3)
    if df["observable"].any():
        start, end, mean_s = best_window(df, window)
        c1.metric(T["best"], f"{start + to_local:%H:%M} – {end + to_local + pd.Timedelta(hours=1):%H:%M}")
        c2.metric(T["mean"], f"{mean_s:.2f}")
        freq = robust_best_hours(df)
        c3.metric(T["robust"], f"{freq.max():.0%}", help=T["robust_help"])
        q = T["quality"][int(mean_s >= 0.5) + int(mean_s >= 0.7)]
        span = f"{start + to_local:%H:%M}", f"{end + to_local + pd.Timedelta(hours=1):%H:%M}"
        st.success(T["verdict"].format(start=span[0], end=span[1], q=q, s=mean_s))
    else:
        c1.error(T["never"])

    st.subheader(T["score_time"])
    st.altair_chart(night_chart(df.rename(columns={"score": T["col_score"]}), [T["col_score"]], labels), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader(T["criteria"])
        names = T["names"]
        st.altair_chart(night_chart(df.rename(columns=names), list(names.values()), labels), use_container_width=True)
    with right:
        st.subheader(T["seeing"])
        st.altair_chart(night_chart(df.rename(columns={"seeing_arcsec": "Seeing"}), ["Seeing"], labels), use_container_width=True)

    pf = pareto_front(df)
    if len(pf):  # empty when nothing is observable, so start/end below are defined
        with st.expander(T["pareto"]):
            st.markdown(T["pareto_help"])
            crit = list(T["names"])
            is_best = pf[crit].eq(pf[crit].max())
            in_win = (pf.index >= start) & (pf.index <= end)
            pct = st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")
            st.dataframe(
                pd.DataFrame({
                    T["col_time"]: [("⭐ " if w else "") + f"{t + to_local:%H:%M}" for t, w in zip(pf.index, in_win)],
                    T["col_score"]: pf["score"].to_numpy(),
                    T["col_best"]: [", ".join(T["names"][c] for c in crit if r[c]) or T["mix"] for _, r in is_best.iterrows()],
                    T["col_alt"]: pf["target_alt"].to_numpy(),
                    T["col_cloud"]: pf["cloud_cover"].to_numpy(),
                    T["col_seeing"]: pf["seeing_arcsec"].to_numpy(),
                    T["col_moon"]: pf["g_moon"].to_numpy(),
                }),
                hide_index=True,
                use_container_width=True,
                column_config={
                    T["col_score"]: pct,
                    T["col_moon"]: pct,
                    T["col_alt"]: st.column_config.NumberColumn(format="%.0f°"),
                    T["col_cloud"]: st.column_config.NumberColumn(format="%.0f%%"),
                    T["col_seeing"]: st.column_config.NumberColumn(format="%.1f″", help=T["seeing_help"]),
                },
            )
    with st.expander(T["raw"]):
        st.dataframe(df.round(2))

    # Full hourly table plus the settings that produced it, so a saved night can be reproduced.
    export = df.assign(
        local_time=labels, lat=lat, lon=lon, target=target, utc_offset=tz,
        w_altitude=w_alt, w_moon=w_moon, w_cloud=w_cloud, w_seeing=w_see,
        weather_source=meta["weather_source"], seeing_label_source=meta["seeing_label_source"],
    )
    st.download_button(
        T["save"],
        export.to_csv(index_label="time_utc").encode("utf-8"),
        file_name=f"skyscore_{date:%Y-%m-%d}_{target.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.info(
        T["info"].format(
            ws=meta["weather_source"],
            ls=meta["seeing_label_source"],
            mae=meta["seeing_cv"].get("cv_mae_arcsec", float("nan")),
            r2=meta["seeing_cv"].get("cv_r2", float("nan")),
        )
    )
