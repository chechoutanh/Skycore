"""Streamlit dashboard for SkyScore.  Run:  streamlit run app.py"""

import hmac
import sys
from datetime import time as dtime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from skyscore.astro import TARGETS  # noqa: E402
from skyscore.evaluation import add_predictions, evaluate, load_logs, session_midpoint  # noqa: E402
from skyscore.pipeline import build_table  # noqa: E402
from skyscore.scoring import SEEING_BEST, SEEING_WORST, Weights, best_window, minmax, pareto_front, robust_best_hours  # noqa: E402
from skyscore.seeing_model import train_default  # noqa: E402

st.set_page_config(page_title="SkyScore", page_icon="🔭", layout="wide")


DATA = Path(__file__).parent / "data"
EVENTS, OBS = DATA / "events.csv", DATA / "observations.csv"
PHOTOS = Path(__file__).parent / "photos"  # photos/<event_id>/*.jpg, made with `python -m skyscore.photos`
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@st.cache_resource
def get_model():
    return train_default()


def log_unlocked(T) -> bool:
    """Owner-only gate: the log is shown only after entering `log_password` from Streamlit secrets."""
    try:
        pw = st.secrets.get("log_password")
    except FileNotFoundError:  # no secrets.toml at all
        pw = None
    if not pw:
        st.info(T["log_no_pw"])
        return False
    if st.session_state.get("log_ok"):
        return True
    entered = st.text_input(T["log_pw"], type="password", key="log_pw")
    if entered and hmac.compare_digest(entered.encode(), str(pw).encode()):
        st.session_state.log_ok = True
        st.rerun()
    if entered:
        st.error(T["log_pw_bad"])
    return False


@st.cache_data(show_spinner=False)
def predicted_logs(stamp: tuple, offline: bool) -> pd.DataFrame:
    """Logged hours plus what SkyScore forecast for them; `stamp` (file mtimes) refreshes it after a save."""
    return add_predictions(load_logs(EVENTS, OBS), get_model(), offline=offline)


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

#### How the numbers are calculated

* **Position of the target.** Planets, the Moon, the Sun and planetary moons: Astropy computes their sky position for each hour from an ephemeris (`get_body`), as seen from your latitude/longitude. Deep-sky objects and constellations are fixed points (RA/Dec). Each position is converted to the local sky: **altitude** $h$ above the horizon.
* **Visibility (is it blocked?).** A target is hidden by the horizon when $h<0°$, and too low to be useful below 15° (thick air, buildings, trees). The Sun must also be low enough: below −18°, −12° or −6° depending on the target (see above). Outside these limits the hour scores 0.
* **Altitude score.** $g_{\text{alt}}=\max(0,\sin h)$, a proxy for $1/\text{airmass}$: the target at the zenith scores 1, at 30° it scores 0.5.
* **Moon.** The illuminated fraction is $k=(1+\cos i)/2$, where $i$ is the Sun–Moon phase angle. The penalty grows with $k$ and shrinks with the Moon–target angular separation $\theta$ (as $e^{-\theta/40°}$); it applies only while the Moon is above the horizon.
* **Clouds and seeing.** Cloud cover comes from the Open-Meteo forecast ($g=1-c/100$). Seeing (arcsec) comes from the Random Forest model, then $g_{\text{see}}$ maps 0.8″ (best) to 3.5″ (worst) onto 1 to 0.

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

#### Các con số được tính như thế nào

* **Vị trí mục tiêu.** Hành tinh, Mặt Trăng, Mặt Trời và các vệ tinh hành tinh: Astropy tính vị trí trên bầu trời cho từng giờ từ lịch thiên văn (`get_body`), theo vĩ độ/kinh độ của bạn. Thiên thể xa (thiên hà, tinh vân, cụm sao) và chòm sao là các điểm cố định (RA/Dec). Mỗi vị trí được đổi sang bầu trời địa phương: **độ cao** $h$ so với đường chân trời.
* **Độ khuất (có bị che không).** Mục tiêu bị đường chân trời che khi $h<0°$, và quá thấp để dùng được khi dưới 15° (khí quyển dày, nhà cửa, cây cối). Mặt Trời cũng phải đủ thấp: dưới −18°, −12° hoặc −6° tuỳ mục tiêu (xem trên). Ngoài các giới hạn này, giờ đó được 0 điểm.
* **Điểm độ cao.** $g_{\text{alt}}=\max(0,\sin h)$, đại diện cho $1/\text{airmass}$: mục tiêu ở thiên đỉnh được 1, ở 30° được 0,5.
* **Mặt Trăng.** Phần được chiếu sáng là $k=(1+\cos i)/2$, với $i$ là góc pha Mặt Trời–Mặt Trăng. Mức phạt tăng theo $k$ và giảm theo góc cách $\theta$ giữa Mặt Trăng và mục tiêu (như $e^{-\theta/40°}$); chỉ áp dụng khi Mặt Trăng ở trên đường chân trời.
* **Mây và seeing.** Mây lấy từ dự báo Open-Meteo ($g=1-c/100$). Seeing (arcsec) do mô hình Random Forest dự đoán, rồi $g_{\text{see}}$ ánh xạ 0,8″ (tốt nhất) đến 3,5″ (tệ nhất) sang 1 đến 0.

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
        "tabs": ["🔭 Planner", "🌟 Our stargazing nights", "📝 Observing log", "ℹ️ About", "📖 Guide"],
        "nights_intro": "Public stargazing sessions run by our club: who came, where, and what we looked at.",
        "nights_events": "Sessions", "nights_people": "Reached by the club", "nights_places": "Places",
        "nights_people_help": "People who joined the club's own activities (approx.).",
        "nights_crowd": "Total event attendance", "nights_crowd_help": "Everyone at the events, as reported by the organisers (approx.).",
        "nights_post": "📎 Read the club's post", "nights_window": " Suggested window: {w}.",
        "nights_used": "🔭 SkyScore was used to pick the time for this session.",
        "nights_people_one": "{n} participants",
        "nights_empty": "No photos yet. Put photos for a session in `photos/<event_id>/` "
        "(for example `photos/E01/`) with `python -m skyscore.photos <photo folder> E01`: "
        "it shrinks them and removes GPS location.",
        "log_intro": "After each session, note how the sky **really** was. With 20–30 sessions we can measure how accurate SkyScore is. "
        "Log bad hours too (clouds, blurry stars): they matter just as much.",
        "log_pw": "🔒 Owner password", "log_pw_bad": "Wrong password.",
        "log_no_pw": "🔒 Saved entries are private. To view them, the owner sets `log_password` in `.streamlit/secrets.toml` "
        "(or in the app's Secrets on Streamlit Cloud).",
        "f": {
            "sec_event": "#### 1. About the session",
            "event_name": "Event name", "event_name_ph": "e.g. Mid-Autumn Moon night",
            "date": "Date", "date_help": "Local date of the EVENING the session started (a session past midnight keeps that evening's date).",
            "club": "Club *", "site": "Place *",
            "lat": "Latitude (°)", "lon": "Longitude (°)",
            "target": "Main target", "target_help": "Not in the list? Pick the closest one you observed and write the real one under Other targets.",
            "other_targets": "Other targets", "equipment": "Equipment",
            "equipment_help": "Telescope, focal length (mm), camera. Focal length and camera are needed if seeing was measured from images.",
            "participants": "People reached by the club", "participants_help": "People who joined the club's own activities. Approximate is fine: 40+",
            "event_participants": "Total event attendance", "event_participants_help": "Everyone at the event, as published by the organisers. Same as above for the club's own sessions.",
            "prediction_saved": "Was SkyScore's suggestion saved BEFORE the session?",
            "prediction_saved_help": "yes = saved or screenshotted before; no = SkyScore was run afterwards. Please be honest.",
            "yesno": {"yes": "Yes", "no": "No"},
            "recommended_window": "Window SkyScore suggested (local)",
            "source_url": "Link to the club's post (optional)",
            "sec_obs": "#### 2. Conditions you observed",
            "start": "Start (local)", "end": "End (local)", "end_help": "An end before the start means the session ran past midnight.",
            "seeing_method": "How seeing was measured",
            "methods": {"scale_1_5": "By eye, 1–5 (5 = steadiest)", "pickering_1_10": "Pickering scale 1–10 (10 = best)",
                        "fwhm_arcsec": "From images, FWHM in arcsec (smaller = better)"},
            "seeing_value": "Seeing value", "seeing_value_help": "On the scale chosen on the left. FWHM to arcsec = FWHM_px × 206.265 × pixel_µm ÷ focal_mm.",
            "transparency": "Transparency (optional)", "transparency_help": "Faintest naked-eye star magnitude, or an SQM reading.",
            "err_required": "Please fill in Club and Place.", "err_seeing": "Seeing value must be between {lo:g} and {hi:g} on this scale.",
            "err_url": "The link must start with https://", "err_number": "Transparency must be a number (use a dot: 5.5).",
        },
        "log_quality": "Overall, how good was it?", "log_quality_help": "1 = useless · 5 = excellent",
        "log_cloud": "Sky covered by cloud (%)",
        "log_wind": "Wind", "wind": {"calm": "Calm", "light": "Light", "strong": "Strong (shook the telescope)"},
        "log_notes": "Notes (optional)", "log_submit": "💾 Save this entry",
        "log_saved": "Saved to session {eid}. Logging another time block of the same session? Keep the details, change the times and save again.",
        "log_cloud_warn": "Note: on a hosted app (e.g. Streamlit Cloud) saved entries can disappear on restart. Download the CSV files below to keep them.",
        "log_table": "🔒 Saved entries ({n}), owner only",
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
        "cmp_title": "Forecast vs what you saw",
        "cmp_help": "Each row is a logged hour. Bars far apart = the cloud forecast was wrong that night. "
        "Seeing and SkyScore's score are converted to the same 1–5 scale as your ratings so they can be compared side by side.",
        "cmp_cols": {"hour": "Session", "date": "Date", "site": "Place", "cloud_f": "Cloud forecast", "cloud_o": "Cloud seen",
                     "see_f": "Seeing forecast", "see_o": "Seeing seen", "q_f": "SkyScore score", "q_o": "Your rating"},
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
        "tabs": ["🔭 Lập lịch", "🌟 Những buổi ngắm sao", "📝 Nhật ký quan sát", "ℹ️ Giới thiệu", "📖 Hướng dẫn"],
        "nights_intro": "Các buổi ngắm sao cộng đồng do CLB tổ chức: ai đã tham gia, ở đâu, và đã quan sát gì.",
        "nights_events": "Số buổi", "nights_people": "Người CLB tiếp cận", "nights_places": "Số địa điểm",
        "nights_people_help": "Số người tham gia trực tiếp hoạt động của CLB (ước tính).",
        "nights_crowd": "Tổng người dự sự kiện", "nights_crowd_help": "Toàn bộ người dự sự kiện, theo số liệu ban tổ chức công bố (ước tính).",
        "nights_post": "📎 Xem bài đăng của CLB", "nights_window": " Khung giờ gợi ý: {w}.",
        "nights_used": "🔭 Buổi này đã dùng SkyScore để chọn giờ.",
        "nights_people_one": "{n} người tham gia",
        "nights_empty": "Chưa có ảnh. Đặt ảnh của mỗi buổi vào `photos/<mã buổi>/` "
        "(ví dụ `photos/E01/`) bằng lệnh `python -m skyscore.photos <thư mục ảnh> E01`: "
        "lệnh này giảm dung lượng và xoá vị trí GPS trong ảnh.",
        "log_intro": "Sau mỗi buổi, ghi lại bầu trời **thực tế** ra sao. Có khoảng 20–30 buổi là đo được SkyScore chuẩn đến đâu. "
        "Hãy ghi cả những giờ xấu (mây, sao nhoè): chúng quan trọng không kém.",
        "log_pw": "🔒 Mật khẩu chủ sở hữu", "log_pw_bad": "Sai mật khẩu.",
        "log_no_pw": "🔒 Dữ liệu đã lưu là riêng tư. Để xem, chủ sở hữu đặt `log_password` trong `.streamlit/secrets.toml` "
        "(hoặc mục Secrets của app trên Streamlit Cloud).",
        "f": {
            "sec_event": "#### 1. Thông tin buổi",
            "event_name": "Tên sự kiện", "event_name_ph": "ví dụ: Trung thu Lãm nguyệt",
            "date": "Ngày", "date_help": "Ngày địa phương của BUỔI TỐI bắt đầu (buổi kéo qua nửa đêm vẫn ghi ngày của buổi tối).",
            "club": "CLB *", "site": "Địa điểm *",
            "lat": "Vĩ độ (°)", "lon": "Kinh độ (°)",
            "target": "Mục tiêu chính", "target_help": "Không có trong danh sách? Chọn mục tiêu gần nhất đã quan sát và ghi mục tiêu thật ở Mục tiêu khác.",
            "other_targets": "Mục tiêu khác", "equipment": "Thiết bị",
            "equipment_help": "Kính, tiêu cự (mm), camera. Cần tiêu cự và camera nếu đo seeing từ ảnh.",
            "participants": "Số người CLB tiếp cận", "participants_help": "Số người tham gia trực tiếp hoạt động của CLB. Ghi ước lượng cũng được: 40+",
            "event_participants": "Tổng người dự sự kiện", "event_participants_help": "Toàn bộ người dự sự kiện, theo số ban tổ chức công bố. Buổi CLB tự tổ chức thì ghi giống ô bên trái.",
            "prediction_saved": "Gợi ý của SkyScore có được lưu TRƯỚC buổi quan sát không?",
            "prediction_saved_help": "Có = đã lưu/chụp màn hình trước buổi; Không = chạy SkyScore sau buổi. Ghi trung thực.",
            "yesno": {"yes": "Có", "no": "Không"},
            "recommended_window": "Khung giờ SkyScore gợi ý (giờ địa phương)",
            "source_url": "Link bài đăng của CLB (không bắt buộc)",
            "sec_obs": "#### 2. Điều kiện quan sát thực tế",
            "start": "Bắt đầu (giờ địa phương)", "end": "Kết thúc (giờ địa phương)", "end_help": "Giờ kết thúc sớm hơn giờ bắt đầu = buổi kéo qua nửa đêm.",
            "seeing_method": "Cách đo seeing",
            "methods": {"scale_1_5": "Bằng mắt, thang 1–5 (5 = ổn định nhất)", "pickering_1_10": "Thang Pickering 1–10 (10 = tốt nhất)",
                        "fwhm_arcsec": "Từ ảnh, FWHM giây cung (càng nhỏ càng tốt)"},
            "seeing_value": "Giá trị seeing", "seeing_value_help": "Theo thang đã chọn bên trái. FWHM ra giây cung = FWHM_px × 206.265 × pixel_µm ÷ tiêu_cự_mm.",
            "transparency": "Độ trong (không bắt buộc)", "transparency_help": "Cấp sao mờ nhất thấy bằng mắt thường, hoặc số đo SQM.",
            "err_required": "Hãy điền CLB và Địa điểm.", "err_seeing": "Giá trị seeing phải từ {lo:g} đến {hi:g} theo thang này.",
            "err_url": "Link phải bắt đầu bằng https://", "err_number": "Độ trong phải là số (dùng dấu chấm: 5.5).",
        },
        "log_quality": "Đánh giá chung buổi quan sát", "log_quality_help": "1 = không xem được gì · 5 = tuyệt vời",
        "log_cloud": "Phần trời bị mây che (%)",
        "log_wind": "Gió", "wind": {"calm": "Lặng", "light": "Nhẹ", "strong": "Mạnh (làm rung kính)"},
        "log_notes": "Ghi chú (không bắt buộc)", "log_submit": "💾 Lưu mục này",
        "log_saved": "Đã lưu vào buổi {eid}. Muốn ghi thêm khung giờ khác của cùng buổi? Giữ nguyên thông tin, đổi giờ rồi lưu tiếp.",
        "log_cloud_warn": "Lưu ý: nếu app chạy trên máy chủ (ví dụ Streamlit Cloud), dữ liệu đã lưu có thể mất khi khởi động lại. Hãy tải các file CSV bên dưới để giữ lại.",
        "log_table": "🔒 Dữ liệu đã lưu ({n} mục), chỉ chủ sở hữu",
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
        "cmp_title": "Dự báo so với thực tế",
        "cmp_help": "Mỗi dòng là một giờ đã ghi. Hai cột cách xa nhau = đêm đó dự báo mây sai nhiều. "
        "Seeing và điểm SkyScore được quy về cùng thang 1–5 như đánh giá của bạn để đặt cạnh nhau so sánh.",
        "cmp_cols": {"hour": "Buổi", "date": "Ngày", "site": "Địa điểm", "cloud_f": "Mây dự báo", "cloud_o": "Mây thực tế",
                     "see_f": "Seeing dự báo", "see_o": "Seeing thực tế", "q_f": "Điểm SkyScore", "q_o": "Bạn đánh giá"},
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

tab_plan, tab_nights, tab_log, tab_intro, tab_guide = st.tabs(T["tabs"])

# Other tabs are filled first so they still render if the planner calls st.stop().
with tab_nights:
    st.markdown(T["nights_intro"])
    ev = pd.read_csv(EVENTS).sort_values("date", ascending=False)
    def count(col):  # "~40+" -> 40; missing column or text -> NaN
        return pd.to_numeric(ev.get(col, pd.Series(index=ev.index, dtype=str)).astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    people, crowd = count("participants"), count("event_participants")
    a, b, c, d = st.columns(4)
    a.metric(T["nights_events"], len(ev))
    b.metric(T["nights_people"], f"{int(people.sum())}+", help=T["nights_people_help"])
    c.metric(T["nights_crowd"], f"{int(crowd.sum())}+", help=T["nights_crowd_help"])
    d.metric(T["nights_places"], ev["site"].nunique())
    shown = 0
    for (_, e), n in zip(ev.iterrows(), people):
        pics = sorted(p for p in (PHOTOS / str(e["event_id"])).glob("*") if p.suffix.lower() in IMG_EXTS)
        if not pics:
            continue
        shown += 1
        st.divider()
        name = e.get("event_name")
        has_name = isinstance(name, str) and name.strip()
        st.subheader(name if has_name else e["site"])
        looked_at = ", ".join(x for x in (e["target"], e["other_targets"]) if isinstance(x, str) and x.strip())
        facts = [e["date"], e["site"] if has_name else None, e["club"],
                 T["nights_people_one"].format(n=e["participants"]) if pd.notna(n) else None,
                 f"🔭 {looked_at}" if looked_at else None]
        st.caption(" · ".join(str(x) for x in facts if isinstance(x, str) and x))
        url = e.get("source_url")
        if isinstance(url, str) and url.startswith("https://"):
            st.markdown(f"[{T['nights_post']}]({url})")
        if str(e["prediction_saved"]).strip().lower() == "yes":
            w = e["recommended_window"]
            st.success(T["nights_used"] + (T["nights_window"].format(w=w) if isinstance(w, str) and w.strip() else ""))
        cols = st.columns(3)
        for i, p in enumerate(pics):
            cols[i % 3].image(str(p), width="stretch")
    if not shown:
        st.info(T["nights_empty"])

with tab_log:
    st.markdown(T["log_intro"])
    F = T["f"]
    # Same fields as the club's Google Sheet (tabs events + observations); sidebar values are only defaults.
    with st.form("log"):
        st.markdown(F["sec_event"])
        a, b = st.columns(2)
        ev_name = a.text_input(F["event_name"], placeholder=F["event_name_ph"])
        ev_date = b.date_input(F["date"], date, help=F["date_help"])
        club = a.text_input(F["club"])
        site = b.text_input(F["site"])
        ev_lat = a.number_input(F["lat"], -90.0, 90.0, float(lat), format="%.4f", help=T["lat_help"])
        ev_lon = b.number_input(F["lon"], -180.0, 180.0, float(lon), format="%.4f", help=T["lat_help"])
        ev_tz = a.number_input(T["tz"], -12.0, 14.0, float(tz), 0.5, help=T["tz_help"])
        ev_target = b.selectbox(F["target"], TARGETS, index=TARGETS.index(target), help=F["target_help"])
        others = st.text_input(F["other_targets"], placeholder="Jupiter; M45")
        equipment = st.text_input(F["equipment"], placeholder="Dobson 8 inch, f=1200 mm, ZWO ASI585MC", help=F["equipment_help"])
        a, b = st.columns(2)
        people = a.text_input(F["participants"], placeholder="30", help=F["participants_help"])
        crowd = b.text_input(F["event_participants"], placeholder="300+", help=F["event_participants_help"])
        saved = a.radio(F["prediction_saved"], ["yes", "no"], index=1, format_func=F["yesno"].get, horizontal=True,
                        help=F["prediction_saved_help"])
        window_txt = b.text_input(F["recommended_window"], placeholder="20:00-23:00")
        url = st.text_input(F["source_url"], placeholder="https://www.facebook.com/...")

        st.markdown(F["sec_obs"])
        a, b = st.columns(2)
        t0 = a.time_input(F["start"], dtime(19, 0), step=900)
        t1 = b.time_input(F["end"], dtime(21, 0), step=900, help=F["end_help"])
        method = a.selectbox(F["seeing_method"], list(F["methods"]), format_func=F["methods"].get)
        seeing = b.number_input(F["seeing_value"], 0.0, 10.0, 3.0, 0.1, help=F["seeing_value_help"])
        cloud = a.slider(T["log_cloud"], 0, 100, 0, 5)
        quality = b.slider(T["log_quality"], 1.0, 5.0, 3.0, 0.1, help=T["log_quality_help"])
        transparency = a.text_input(F["transparency"], help=F["transparency_help"])
        wind = b.radio(T["log_wind"], list(T["wind"]), format_func=T["wind"].get, horizontal=True)
        notes = st.text_input(T["log_notes"])
        submitted = st.form_submit_button(T["log_submit"], type="primary")
    if submitted:
        lo, hi = {"scale_1_5": (1, 5), "pickering_1_10": (1, 10), "fwhm_arcsec": (0.3, 10)}[method]
        errors = [F["err_required"]] if not (club.strip() and site.strip()) else []
        if not lo <= seeing <= hi:
            errors.append(F["err_seeing"].format(lo=lo, hi=hi))
        if url.strip() and not url.strip().startswith("https://"):
            errors.append(F["err_url"])
        if transparency.strip() and pd.isna(pd.to_numeric(transparency.strip(), errors="coerce")):
            errors.append(F["err_number"])
        for err in errors:
            st.error(err)
        if not errors:
            mid, hours = session_midpoint(f"{t0:%H:%M}", f"{t1:%H:%M}")
            eid = save_log(
                {"date": f"{ev_date:%Y-%m-%d}", "club": club.strip(), "site": site.strip(), "lat": ev_lat, "lon": ev_lon,
                 "utc_offset": ev_tz, "target": ev_target, "other_targets": others.strip(), "equipment": equipment.strip(),
                 "participants": people.strip(), "event_participants": crowd.strip(),
                 "recommended_window": window_txt.strip(), "prediction_saved": saved,
                 "source_url": url.strip(), "event_name": ev_name.strip()},
                {"time": mid, "duration_h": hours, "seeing_value": seeing, "seeing_method": method,
                 "cloud_observed": cloud, "quality_rating": quality, "transparency": transparency.strip(),
                 "wind_feel": wind, "notes": notes.strip()},
            )
            st.success(T["log_saved"].format(eid=eid))

    logs = load_logs(EVENTS, OBS)
    stamp = (EVENTS.stat().st_mtime, OBS.stat().st_mtime)
    if st.button(T["log_eval"]):
        try:
            with st.spinner(T["computing"]):
                m = evaluate(predicted_logs(stamp, offline))
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

    # Everyone can add entries and see the aggregate check; the raw entries are owner-only.
    with st.expander(T["log_table"].format(n=len(logs))):
        if log_unlocked(T):
            st.dataframe(logs[["event_id", "date", "time", "site", "target", "seeing_value", "cloud_observed", "quality_rating", "notes"]], hide_index=True)
            a, b = st.columns(2)
            a.download_button(T["log_dl_events"], EVENTS.read_bytes(), "events.csv", "text/csv")
            b.download_button(T["log_dl_obs"], OBS.read_bytes(), "observations.csv", "text/csv")
            st.caption(T["log_cloud_warn"])

            st.subheader(T["cmp_title"])
            try:
                with st.spinner(T["computing"]):
                    p = predicted_logs(stamp, offline)
            except Exception as e:  # network or bad rows: report, keep the table above usable
                st.error(T["log_eval_err"].format(e=e))
            else:
                st.markdown(T["cmp_help"])
                C = T["cmp_cols"]
                cmp = pd.DataFrame({
                    C["hour"]: p["event_id"] + " " + p["time"],
                    C["date"]: p["date"],
                    C["site"]: p["site"],
                    C["cloud_f"]: p["cloud_forecast"], C["cloud_o"]: p["cloud_observed"],
                    # Predicted arcsec mapped onto the 1–5 scale observers use (5 = steadiest).
                    C["see_f"]: 1 + 4 * minmax(p["seeing_pred"], SEEING_BEST, SEEING_WORST, invert=True),
                    C["see_o"]: p["seeing_value"].where(p["seeing_method"] == "scale_1_5"),
                    C["q_f"]: 1 + 4 * p["score"], C["q_o"]: p["quality_rating"],
                })
                st.bar_chart(cmp, x=C["hour"], y=[C["cloud_f"], C["cloud_o"]], stack=False, y_label="%", x_label="")
                pct = st.column_config.NumberColumn(format="%.0f%%")
                five = st.column_config.NumberColumn(format="%.1f / 5")
                st.dataframe(cmp, hide_index=True, column_config={
                    C["cloud_f"]: pct, C["cloud_o"]: pct,
                    C["see_f"]: five, C["see_o"]: five, C["q_f"]: five, C["q_o"]: five,
                })

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
    st.altair_chart(night_chart(df.rename(columns={"score": T["col_score"]}), [T["col_score"]], labels), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader(T["criteria"])
        names = T["names"]
        st.altair_chart(night_chart(df.rename(columns=names), list(names.values()), labels), width="stretch")
    with right:
        st.subheader(T["seeing"])
        st.altair_chart(night_chart(df.rename(columns={"seeing_arcsec": "Seeing"}), ["Seeing"], labels), width="stretch")

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
                width="stretch",
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
