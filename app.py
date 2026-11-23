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

SkyScore answers one question: **"Tonight, at what time should we point the telescope at this target?"**

Think of it as a judge who gives **every hour of the night a mark from 0 to 1**, then tells you the best few hours in a row.
You don't need to know any astronomy to use it: pick a place, a date and a target, and read the green verdict in the Planner tab.

#### How it works, in 5 steps

**1. Lay out the night.** SkyScore looks at 24 hours, from local noon on your date to noon the next day,
so the whole evening and night are covered. (For the Sun it looks at the day instead: midnight to midnight.)

**2. Work out where everything is in the sky.** For each hour it calculates how high your target is,
where the Sun is and where the Moon is and how full it is. This is pure geometry, like a clock: it needs no internet
and is accurate to a fraction of a degree.

**3. Throw out hours that are impossible.** An hour gets **0 points** if:

* the target is **lower than 15°** above the horizon (it is hidden behind trees and buildings, and you look through too much air), or
* the sky is **not dark enough** for that kind of target. Faint galaxies and nebulae need full darkness;
  constellations need the sky "mostly dark"; bright planets and the Moon can be seen soon after sunset.

**4. Mark each remaining hour on four things**, each from 0 (bad) to 1 (good):

| | Criterion | What it means in plain words | Where the data comes from |
|---|---|---|---|
| 🌌 | **Height in sky** | The higher the target, the less air you look through, so it looks sharper and brighter. Overhead = 1, halfway up (30°) = 0.5. | Calculated (Astropy) |
| 🌙 | **Moon darkness** | A bright Moon lights up the sky like a street lamp and washes out faint objects. It hurts most when it is full, up, and close to your target. A new Moon or a Moon below the horizon = 1. | Calculated (Astropy) |
| ☁️ | **Clear sky** | Clouds block everything. No cloud = 1, fully overcast = 0. | Weather forecast (Open-Meteo, free) |
| ✨ | **Steady air (seeing)** | Even on a clear night, moving air makes stars "boil" and twinkle, so details blur. Steady air matters most for planets and the Moon. | Predicted by a machine-learning model from the forecast |

**5. Add the marks up, with your priorities.** Not every goal cares about the same things, so each criterion has a weight
(how much it counts). Picking **What do you want to observe?** sets the weights for you:

| Goal | Height | Moon | Clouds | Seeing |
|---|---|---|---|---|
| Balanced (default) | 35% | 20% | 30% | 15% |
| Faint galaxies & nebulae | 25% | **35%** | 30% | 10% |
| Planets & the Moon | 35% | 5% | 25% | **35%** |
| Public / casual night | 25% | 10% | **55%** | 10% |

The total mark of an hour is the weighted average of its four marks, so it is also between 0 and 1.
As a rule of thumb: **above 0.7 is good, 0.5–0.7 is fair, below 0.5 is poor.**

#### What you get back

* **Best window**: the block of consecutive hours (you choose how long) with the highest average mark. This is the "go out at…" answer.
* **Confidence**: SkyScore re-does the choice 500 times with random priorities. If the same hour wins most of the time,
  the answer does not depend on how you set the sliders, so you can trust it. A low value means "it depends on what you care about".
* **Other good hours (trade-offs)**: hours that are unbeatable in at least one way, e.g. the clearest hour, or the hour with the
  target highest. Useful if your priorities differ from the sliders.
* **Charts**: the score and the four criteria hour by hour, so you can see *why* an hour is good or bad. Grey bands = impossible hours.

#### Where the data comes from

* **Sky positions**: Astropy, a standard astronomy library, works fully offline.
* **Weather**: the free Open-Meteo forecast (cloud, humidity, temperature, pressure, wind near the ground and at ~10 km high).
  It reaches about 4 days ahead. If there is no internet or the date is too far away, SkyScore uses **made-up (synthetic) weather**
  and says so clearly under the charts: results are then only a demo.
* **Seeing**: there is no free seeing forecast for most places, so SkyScore estimates it with a Random Forest model
  (a set of decision trees) that looks at the things that make air restless: strong wind high up (the jet stream),
  wind near the ground, fast temperature and pressure changes, and humidity.

#### How we check it is right

After each session the club writes down in the **📝 Observing log** tab what the sky was really like (cloud, seeing, an overall 1–5 rating).
SkyScore then replays the forecast it *would have had before* that night (an archived forecast, not hindsight)
and compares. It is judged against a simple baseline, "always guess the average", so we can tell real skill from luck.
Around 20–30 sessions are needed before the numbers mean much.

> ⚠️ **Honest limits.** The seeing model has not yet learned from real measurements: it was trained on a physics-based
> *estimate* of seeing, so its seeing numbers are an educated guess. Cloud comes from a real forecast and is the most
> reliable part. Light pollution, local obstacles and dew on the optics are not included.
"""

INTRO_TECH_EN = r"""
**Total score.** For each hour $t$, with goodness values $g_i(t)\in[0,1]$ and weights $w_i$ normalized so that $\sum_i w_i=1$:

$$S(t) = \mathbb{1}[\text{observable}(t)]\sum_i w_i\,g_i(t).$$

**Observable.** Target altitude $h>15°$ **and** Sun altitude below a limit that depends on the target:
−18° (astronomical night) for galaxies, nebulae, clusters, Uranus and Neptune; −12° (nautical twilight) for constellations
and Saturn's moons; −6° (civil twilight) for the Moon, Mercury, Venus, Mars, Jupiter, Saturn and Jupiter's moons;
no limit for the Sun.

**Positions.** Solar-system bodies come from Astropy's built-in ephemeris (`get_body`) as seen from your latitude/longitude;
planetary moons use their planet's position (they stay within 0.2°). Deep-sky objects and constellations are fixed RA/Dec (J2000).
Everything is converted to local altitude/azimuth for each hour.

**Height in sky.** $g_{\text{alt}}=\max(0,\sin h)$, a stand-in for $1/\text{airmass}$.

**Moon.** Illuminated fraction $k=(1+\cos i)/2$, with $i$ the Sun–Moon phase angle. With $\theta$ the Moon–target separation:

$$g_{\text{moon}} = 1 - k\cdot\mathbb{1}[\text{Moon up}]\cdot\left(0.4 + 0.6\,e^{-\theta/40°}\right).$$

A full Moon that is up always costs at least 0.4, up to 1.0 when it sits right next to the target.
$g_{\text{moon}}=1$ when the target is the Sun or the Moon.

**Clouds.** $g_{\text{cloud}} = 1 - c/100$, with $c$ the forecast cloud cover (%).

**Seeing.** The Random Forest (200 trees) predicts seeing in arcseconds from 7 features: wind at 250 hPa (~10 km),
wind at 10 m, |temperature change over 3 h|, |pressure change over 3 h|, temperature − dew point, relative humidity and cloud cover.
It is trained on 240 days of synthetic weather labelled with a physics-inspired formula (the "proxy"), and checked with
time-ordered cross-validation (no peeking into the future); the MAE/R² under the Planner charts come from that check, so they only
show the model learned the proxy. Then $g_{\text{see}}$ maps 0.8″ (excellent) → 1 and 3.5″ (poor) → 0, clipped.

**Best window.** The rolling mean of $S$ over $L$ consecutive hours; the highest one wins.

**Trade-off hours (Pareto set).** Observable hours that no other hour beats on all four $g_i$ at once.

**Confidence.** 500 weight vectors drawn uniformly at random (flat Dirichlet); the share that picks the same best hour.

**Evaluation.** For each logged session, cloud and seeing forecasts are averaged over the session's hours using
Open-Meteo's *historical forecast* archive. Metrics: cloud mean absolute error vs. always guessing the observed average;
Spearman rank correlation between predicted and observed seeing; the 1–5 rating at SkyScore's top hour vs. a random logged hour.
"""

INTRO_VI = r"""
### SkyScore là gì?

SkyScore trả lời một câu hỏi: **"Tối nay, mấy giờ thì nên hướng kính thiên văn vào mục tiêu này?"**

Hãy hình dung SkyScore như một giám khảo **chấm điểm từng giờ trong đêm, từ 0 đến 1**, rồi chỉ ra vài giờ liên tiếp tốt nhất.
Bạn không cần biết thiên văn để dùng: chọn địa điểm, ngày và mục tiêu, rồi đọc dòng kết luận màu xanh ở tab Lập lịch.

#### Cách hoạt động, trong 5 bước

**1. Trải ra cả đêm.** SkyScore xét 24 giờ, từ trưa ngày bạn chọn đến trưa hôm sau, nên bao trọn buổi tối và cả đêm.
(Với Mặt Trời thì xét ban ngày: từ nửa đêm đến nửa đêm.)

**2. Tính xem mọi thứ đang ở đâu trên trời.** Với từng giờ, ứng dụng tính mục tiêu cao bao nhiêu,
Mặt Trời ở đâu, Mặt Trăng ở đâu và tròn bao nhiêu. Đây thuần là hình học, giống như đồng hồ: không cần mạng
và chính xác đến từng phần nhỏ của một độ.

**3. Loại những giờ không thể quan sát.** Một giờ bị **0 điểm** nếu:

* mục tiêu **thấp hơn 15°** so với đường chân trời (bị cây, nhà che, và phải nhìn xuyên qua quá nhiều lớp không khí), hoặc
* trời **chưa đủ tối** cho loại mục tiêu đó. Thiên hà, tinh vân mờ cần trời tối hẳn;
  chòm sao cần trời "gần tối"; hành tinh sáng và Mặt Trăng thì xem được ngay sau hoàng hôn một chút.

**4. Chấm mỗi giờ còn lại theo bốn tiêu chí**, mỗi tiêu chí từ 0 (xấu) đến 1 (tốt):

| | Tiêu chí | Nói đơn giản | Dữ liệu lấy từ đâu |
|---|---|---|---|
| 🌌 | **Độ cao** | Mục tiêu càng cao thì càng ít lớp không khí cản, nên nhìn rõ và sáng hơn. Ở đỉnh đầu = 1, ở lưng chừng (30°) = 0,5. | Tính toán (Astropy) |
| 🌙 | **Trăng tối** | Trăng sáng giống như đèn đường, làm cả bầu trời sáng lên và "nuốt" mất các vật mờ. Hại nhất khi trăng tròn, đang ở trên trời và nằm gần mục tiêu. Trăng non hoặc trăng đã lặn = 1. | Tính toán (Astropy) |
| ☁️ | **Trời quang** | Mây che thì không thấy gì. Không mây = 1, mây kín trời = 0. | Dự báo thời tiết (Open-Meteo, miễn phí) |
| ✨ | **Khí quyển ổn định (seeing)** | Kể cả khi trời quang, không khí chuyển động làm sao "sôi" và lấp lánh, chi tiết bị nhoè. Quan trọng nhất khi xem hành tinh và Mặt Trăng. | Mô hình học máy dự đoán từ dự báo thời tiết |

**5. Cộng điểm theo ưu tiên của bạn.** Mỗi mục đích quan tâm những thứ khác nhau, nên mỗi tiêu chí có một trọng số
(mức độ quan trọng). Chọn **Bạn muốn quan sát gì?** là ứng dụng tự đặt trọng số:

| Mục đích | Độ cao | Trăng | Mây | Seeing |
|---|---|---|---|---|
| Cân bằng (mặc định) | 35% | 20% | 30% | 15% |
| Thiên hà & tinh vân mờ | 25% | **35%** | 30% | 10% |
| Hành tinh & Mặt Trăng | 35% | 5% | 25% | **35%** |
| Buổi quan sát công chúng | 25% | 10% | **55%** | 10% |

Điểm tổng của một giờ là trung bình có trọng số của bốn điểm trên, nên cũng nằm trong khoảng 0 đến 1.
Quy ước dễ nhớ: **trên 0,7 là tốt, 0,5–0,7 là tạm được, dưới 0,5 là kém.**

#### Kết quả bạn nhận được

* **Cửa sổ tốt nhất**: khối giờ liên tiếp (bạn chọn dài bao lâu) có điểm trung bình cao nhất. Đây chính là câu trả lời "nên ra lúc…".
* **Độ tin cậy**: SkyScore chọn lại 500 lần với các bộ ưu tiên ngẫu nhiên. Nếu phần lớn vẫn ra cùng một giờ, kết quả không
  phụ thuộc vào cách bạn kéo thanh trượt, nên rất đáng tin. Giá trị thấp nghĩa là "tuỳ bạn ưu tiên điều gì".
* **Các giờ tốt khác (đánh đổi)**: những giờ không thể bị vượt mặt ở ít nhất một mặt, ví dụ giờ trời quang nhất, hoặc giờ
  mục tiêu lên cao nhất. Hữu ích khi ưu tiên của bạn khác với thanh trượt.
* **Biểu đồ**: điểm và bốn tiêu chí theo từng giờ, để thấy *vì sao* một giờ tốt hay xấu. Vùng xám = giờ không quan sát được.

#### Dữ liệu lấy từ đâu

* **Vị trí thiên thể**: Astropy, thư viện thiên văn chuẩn, chạy hoàn toàn không cần mạng.
* **Thời tiết**: dự báo miễn phí của Open-Meteo (mây, độ ẩm, nhiệt độ, áp suất, gió sát mặt đất và gió ở độ cao ~10 km).
  Dự báo xa được khoảng 4 ngày. Nếu không có mạng hoặc ngày quá xa, SkyScore dùng **thời tiết giả lập**
  và ghi rõ bên dưới biểu đồ: khi đó kết quả chỉ để minh hoạ.
* **Seeing**: hầu hết các nơi không có dự báo seeing miễn phí, nên SkyScore ước lượng bằng mô hình Random Forest
  (một "rừng" cây quyết định) dựa trên những thứ làm không khí xáo động: gió mạnh trên cao (dòng tia),
  gió sát mặt đất, nhiệt độ và áp suất thay đổi nhanh, và độ ẩm.

#### Làm sao biết SkyScore đúng hay sai

Sau mỗi buổi, CLB ghi lại trong tab **📝 Nhật ký quan sát** bầu trời thực tế ra sao (mây, seeing, điểm chung 1–5).
SkyScore sau đó dựng lại dự báo mà nó *lẽ ra đã có trước* đêm đó (dự báo lưu trữ, không phải "biết trước kết quả")
rồi so sánh. Nó được so với một cách đoán đơn giản, "lúc nào cũng đoán bằng mức trung bình", để phân biệt giỏi thật với may mắn.
Cần khoảng 20–30 buổi thì các con số mới có ý nghĩa.

> ⚠️ **Giới hạn cần biết.** Mô hình seeing chưa được học từ số đo thật: nó được huấn luyện trên một *ước lượng*
> seeing theo công thức vật lý, nên con số seeing chỉ là phỏng đoán có cơ sở. Mây lấy từ dự báo thật và là phần
> đáng tin nhất. Ô nhiễm ánh sáng, vật cản tại chỗ và hơi sương đọng trên kính chưa được tính đến.
"""

INTRO_TECH_VI = r"""
**Điểm tổng.** Với mỗi giờ $t$, điểm thành phần $g_i(t)\in[0,1]$ và trọng số $w_i$ đã chuẩn hoá sao cho $\sum_i w_i=1$:

$$S(t) = \mathbb{1}[\text{quan sát được}(t)]\sum_i w_i\,g_i(t).$$

**Quan sát được.** Độ cao mục tiêu $h>15°$ **và** độ cao Mặt Trời thấp hơn một ngưỡng tuỳ mục tiêu:
−18° (đêm thiên văn) cho thiên hà, tinh vân, cụm sao, Sao Thiên Vương, Sao Hải Vương; −12° (chạng vạng hàng hải) cho chòm sao
và vệ tinh Sao Thổ; −6° (chạng vạng dân sự) cho Mặt Trăng, Sao Thuỷ, Sao Kim, Sao Hoả, Sao Mộc, Sao Thổ và vệ tinh Sao Mộc;
không giới hạn với Mặt Trời.

**Vị trí.** Thiên thể trong Hệ Mặt Trời lấy từ lịch thiên văn có sẵn của Astropy (`get_body`), nhìn từ vĩ độ/kinh độ của bạn;
vệ tinh hành tinh dùng vị trí của hành tinh (luôn cách dưới 0,2°). Thiên thể sâu và chòm sao là toạ độ RA/Dec cố định (J2000).
Tất cả được đổi sang độ cao/phương vị địa phương cho từng giờ.

**Độ cao.** $g_{\text{alt}}=\max(0,\sin h)$, đại diện cho $1/\text{airmass}$.

**Mặt Trăng.** Phần được chiếu sáng $k=(1+\cos i)/2$, với $i$ là góc pha Mặt Trời–Mặt Trăng. Với $\theta$ là góc cách giữa Mặt Trăng và mục tiêu:

$$g_{\text{moon}} = 1 - k\cdot\mathbb{1}[\text{Trăng trên trời}]\cdot\left(0{,}4 + 0{,}6\,e^{-\theta/40°}\right).$$

Trăng tròn đang ở trên trời luôn trừ ít nhất 0,4, tối đa 1,0 khi nằm sát mục tiêu.
$g_{\text{moon}}=1$ khi mục tiêu là Mặt Trời hoặc Mặt Trăng.

**Mây.** $g_{\text{cloud}} = 1 - c/100$, với $c$ là lượng mây dự báo (%).

**Seeing.** Random Forest (200 cây) dự đoán seeing theo giây cung từ 7 đặc trưng: gió ở mực 250 hPa (~10 km),
gió ở 10 m, |thay đổi nhiệt độ trong 3 giờ|, |thay đổi áp suất trong 3 giờ|, nhiệt độ − điểm sương, độ ẩm tương đối và lượng mây.
Mô hình học trên 240 ngày thời tiết giả lập được gắn nhãn bằng công thức lấy cảm hứng từ vật lý (nhãn "giả lập"), và được kiểm tra
bằng kiểm định chéo theo thứ tự thời gian (không nhìn trước tương lai); MAE/R² dưới biểu đồ ở tab Lập lịch lấy từ phép kiểm tra này,
nên chỉ cho thấy mô hình đã học được công thức giả lập. Sau đó $g_{\text{see}}$ ánh xạ 0,8″ (rất tốt) → 1 và 3,5″ (kém) → 0, có cắt biên.

**Cửa sổ tốt nhất.** Trung bình trượt của $S$ trên $L$ giờ liên tiếp; cửa sổ cao nhất được chọn.

**Giờ đánh đổi (tập Pareto).** Các giờ quan sát được mà không giờ nào khác vượt trội ở cả bốn $g_i$ cùng lúc.

**Độ tin cậy.** 500 bộ trọng số rút ngẫu nhiên đều (phân phối Dirichlet phẳng); tỉ lệ bộ chọn cùng giờ tốt nhất.

**Đánh giá.** Với mỗi buổi đã ghi, dự báo mây và seeing được lấy trung bình trên các giờ của buổi, dùng kho
*dự báo lịch sử* của Open-Meteo. Chỉ số: sai số tuyệt đối trung bình của mây so với cách luôn đoán bằng trung bình quan sát;
tương quan thứ hạng Spearman giữa seeing dự đoán và quan sát; điểm 1–5 ở giờ SkyScore chọn so với một giờ đã ghi bất kỳ.
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
        "intro_tech_title": "🔬 For the curious: the exact formulas",
        "intro_tech": INTRO_TECH_EN,
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
        "intro_tech_title": "🔬 Dành cho người tò mò: công thức chi tiết",
        "intro_tech": INTRO_TECH_VI,
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
    # Hover zoom for the photo grid (st.image is only used here).
    st.markdown("""<style>
    [data-testid="stImage"] img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; transition: transform .25s ease, box-shadow .25s ease; position: relative; }
    [data-testid="stImage"] img:hover { transform: scale(1.8); z-index: 100; box-shadow: 0 8px 32px rgba(0,0,0,.6); border-radius: 6px; }
    [data-testid="stColumn"]:has(img:hover), [data-testid="stImage"]:has(img:hover) { z-index: 100; overflow: visible; }
    </style>""", unsafe_allow_html=True)
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
        name = e.get("event_name_en") if lang == "en" and isinstance(e.get("event_name_en"), str) else e.get("event_name")
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
    with st.expander(T["intro_tech_title"]):
        st.markdown(T["intro_tech"])
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
