# 📖 Hướng dẫn sử dụng SkyScore

SkyScore giúp bạn chọn **khung giờ tốt nhất trong đêm** để quan sát một thiên thể, dựa trên độ cao mục tiêu, Mặt Trăng, mây và seeing (độ ổn định khí quyển).

## 1. Cài đặt và chạy

Yêu cầu: Python 3.10 trở lên.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Trình duyệt sẽ tự mở `http://localhost:8501`. Nếu không, hãy mở địa chỉ đó bằng tay.

Kiểm thử (không bắt buộc):

```bash
pytest
```

## 2. Các bước sử dụng

> 🌐 Chọn **English / Tiếng Việt** ở đầu thanh bên trái để đổi ngôn ngữ giao diện. Bảng dưới dùng tên tiếng Anh; ở chế độ tiếng Việt, các ô có tên tương ứng (Vĩ độ, Mục tiêu, Trọng số…).

### Bước 1: Nhập địa điểm và mục tiêu (thanh bên trái)

| Ô nhập | Ý nghĩa |
|---|---|
| Latitude / Longitude | Toạ độ nơi quan sát. Mặc định là TP. Hồ Chí Minh. Lấy toạ độ bằng cách nhấp chuột phải trên Google Maps. |
| Target | Thiên thể muốn quan sát: M31, M42, M45, M13, M8, M104, Sao Mộc, Sao Thổ, Sao Hoả, Sao Kim. |
| Date (UTC) | Ngày tính theo giờ UTC. Ứng dụng tính 24 giờ, từ 00:00 đến 23:00 UTC. |
| Offline demo | Bật khi không có mạng. Thời tiết sẽ là dữ liệu **giả lập**. |

### Bước 2: Chỉnh trọng số

Mỗi thanh trượt cho biết tiêu chí đó quan trọng thế nào với bạn. Tổng không cần bằng 1, ứng dụng tự chuẩn hoá.

| Mục đích | Gợi ý trọng số |
|---|---|
| Chụp thiên hà, tinh vân mờ | Tăng **Moon darkness** và **Clear sky** |
| Quan sát hành tinh | Tăng **Seeing** và **Altitude**, giảm **Moon darkness** |
| Buổi quan sát công chúng | Tăng **Clear sky**, tăng **Window length** |

**Window length**: số giờ liên tiếp bạn định quan sát (1–6 giờ).

### Bước 3: Đọc kết quả (tab 🔭 Planner)

| Mục | Cách hiểu |
|---|---|
| Best window (UTC) | Khung giờ nên quan sát. **Giờ Việt Nam = UTC + 7** (13:00 UTC = 20:00 tối). |
| Mean score | Điểm trung bình khung giờ, từ 0 đến 1. Trên khoảng 0.7 là tốt. |
| Robustness of best hour | Tỉ lệ trong 500 bộ trọng số ngẫu nhiên cùng chọn giờ tốt nhất. Cao = chắc chắn. |
| Score over time | Điểm theo giờ. Bằng 0 khi trời chưa tối hẳn hoặc mục tiêu thấp hơn 15°. |
| Criteria | Bốn tiêu chí (0 = xấu, 1 = tốt), giải thích vì sao một giờ được điểm cao hay thấp. |
| Predicted seeing | Seeing dự đoán (arcsec). Càng nhỏ càng tốt; dưới 1.5″ là tốt. |
| Pareto-optimal hours | Các giờ không bị giờ nào khác vượt trội ở cả bốn tiêu chí. |
| Raw table | Toàn bộ số liệu. |

Dòng thông tin cuối trang cho biết nguồn thời tiết (thật hay giả lập) và nguồn nhãn của mô hình seeing.

## 3. Câu hỏi thường gặp

**"Target never observable tonight"?**
Mục tiêu không lên cao quá 15° trong lúc trời tối hẳn (Mặt Trời dưới −18°). Hãy thử ngày khác hoặc mục tiêu khác. Ví dụ M42 dễ quan sát vào các tháng 11 đến 3.

**Thấy chữ "Synthetic (offline demo data)" dù không bật Offline?**
Ứng dụng không lấy được dự báo từ Open-Meteo (mất mạng, hoặc ngày chọn ngoài khoảng dự báo ~4 ngày tới). Kết quả lúc đó chỉ để minh hoạ.

**Seeing dự đoán có chính xác không?**
Chưa. Mô hình hiện học từ nhãn giả lập. Để chính xác hơn, CLB cần ghi nhật ký seeing thực tế qua các buổi quan sát, xem mục "Seeing model" trong [README](../README.md).

**Lần đầu mở ứng dụng hơi chậm?**
Lần chạy đầu, mô hình seeing được huấn luyện (khoảng vài giây đến vài chục giây). Các lần sau dùng bản đã lưu trong bộ nhớ đệm.
