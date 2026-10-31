# 🚀 Đưa SkyScore lên web

SkyScore là ứng dụng Streamlit thuần Python, không cần API key hay cơ sở dữ liệu, nên có thể deploy miễn phí.

## Cách 1 (khuyên dùng): Streamlit Community Cloud

Miễn phí, nhanh nhất, được link dạng `https://<ten-app>.streamlit.app`.

1. **Đưa code lên GitHub** (repo public, hoặc private nếu tài khoản cho phép):

   ```bash
   git init
   git add .
   git commit -m "SkyScore"
   git branch -M main
   git remote add origin https://github.com/<username>/skyscore.git
   git push -u origin main
   ```

2. Vào <https://share.streamlit.io>, đăng nhập bằng GitHub.
3. Bấm **Create app** → **Deploy a public app from GitHub**, chọn:
   * Repository: `<username>/skyscore`
   * Branch: `main`
   * Main file path: `app.py`
   * (Advanced settings) Python version: 3.11 hoặc 3.12
4. Bấm **Deploy**. Lần đầu mất vài phút để cài thư viện.

Mỗi lần `git push`, ứng dụng tự cập nhật.

**Lưu ý:**
* App miễn phí sẽ "ngủ" nếu lâu không ai truy cập; người vào sau chỉ cần bấm nút đánh thức.
* RAM giới hạn khoảng 1 GB, đủ cho SkyScore (mô hình Random Forest nhỏ, huấn luyện một lần rồi cache).
* Open-Meteo cho phép gọi miễn phí cho mục đích phi thương mại.

## Cách 2: Hugging Face Spaces

1. Tạo Space mới tại <https://huggingface.co/new-space>, chọn SDK **Docker** hoặc **Streamlit** (tuỳ lựa chọn đang có).
2. Upload toàn bộ thư mục dự án (hoặc `git push` vào repo của Space).
3. Link có dạng `https://huggingface.co/spaces/<username>/skyscore`.

## Cách 3: Chạy trong mạng LAN của CLB

Trên một máy trong mạng:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Các máy khác mở `http://<IP-máy-chạy>:8501`. Có thể cần mở cổng 8501 trên Windows Firewall.

## Kiểm tra trước khi deploy

* `requirements.txt` đã đủ thư viện (`pytest` không cần cho web nhưng để lại cũng không sao).
* Chạy `pytest` và `streamlit run app.py` trên máy mình thấy ổn.
* Không commit file bí mật; `.streamlit/secrets.toml` đã nằm trong `.gitignore`.
