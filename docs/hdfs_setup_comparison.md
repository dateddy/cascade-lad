# So sánh setup HDFS: DeepLog gốc, Le & Zhang 2022 và cascade-lad

Nguồn:
- DeepLog: Du, Li, Zheng, Srikumar. *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*. CCS 2017, Section 5.1.2–5.1.3, Table 3–4.
- Le & Zhang: *Log-based Anomaly Detection with Deep Learning: How Far Are We?* ICSE 2022 (arXiv 2202.04301), Section 3.1.1, 3.3–3.5, 4.1–4.2, Table 1, 4.
- Code của Le & Zhang: <https://github.com/LogIntelligence/LogADEmpirical> (nhánh `main`).

## Bảng so sánh

| | DeepLog gốc (CCS 2017) | Le & Zhang (ICSE 2022) | cascade-lad (hiện tại) |
|---|---|---|---|
| Dữ liệu đầu vào | HDFS, 11.197.954 dòng log | HDFS, 11.175.629 dòng log | `HDFS.log` raw và `anomaly_label.csv` |
| Gom nhóm | Session theo `block_id` | Session theo `block_id` | Session theo `block_id` |
| Cách chọn train | 100.000 dòng log đầu tiên (thứ tự trong file, tức theo thời gian) | Xáo trộn ngẫu nhiên các session, chia 80:20 ("session (random)") | Sắp xếp block theo `start`, cắt 0,7 / 0,1 / 0,2, purge các block cắt ngang ranh giới |
| Train | 4.855 session normal (<1%) | 460.048 session | Chỉ gồm block normal |
| Test | 553.366 normal, 15.200 anomaly (toàn bộ phần còn lại) | 115.013 session, 3.317 anomaly | 20% block cuối |
| Validation | Không có | Không ghi | 10%, dùng cho early stopping (val loss) và chọn g |
| Số log key | 29 | 48 (Drain, tham số mặc định) | Drain chỉ fit trên train, template mới gặp → `<UNK>` |
| LSTM | L=2, α=64 | 2 lớp, 128 neuron | Chưa có config |
| h / g | h=10, g=9 (cố định) | Không ghi | g chọn trên val (invariant #5) |
| Mức đánh giá | Session là anomaly nếu có ít nhất một key ngoài top-g | Theo session | Theo window (= block) |
| Kết quả HDFS | FP 833, FN 619, F1 ≈ 0,96 | P 0,835, R 0,994, F1 0,908 | — |

## Ghi chú

1. **Le & Zhang không lặp lại setup HDFS của DeepLog.** Họ thay đổi cách chia, kích thước train, số template và kích thước LSTM. Hai con số F1 0,96 và 0,908 vì thế không đo cùng điều kiện.

2. **Nhận định "HDFS không có timestamp" không đúng với dữ liệu raw.** Le & Zhang, Section 3.4 (RQ1):
   > "As the HDFS dataset does not contain timestamp information, the chronological selection cannot be applied to HDFS, and thus it is not used in this RQ."

   Thực tế, `HDFS.log` có timestamp `yymmdd HHMMSS` ở mọi dòng (từ 081109 203518 đến 081111 111628). DeepLog gốc chọn train bằng cách lấy các dòng log đầu tiên của file, và chính Le & Zhang (Section 3.1.1) cũng xếp DeepLog vào nhóm chọn dữ liệu "based on the timestamp of logs".

3. **Code của Le & Zhang chia ngẫu nhiên.** Trong `logadempirical/logdeep/dataset/session.py`, hàm `session_window` gom log theo block theo thứ tự trong file, không dùng cột thời gian, và kết thúc bằng `results = shuffle(results)` (sklearn, không đặt seed). Sau đó `logadempirical/dataset/data_process.py` lấy `window_df[:n_train]` làm train.

   Lý do thật có lẽ là cách chia chronological của họ: lấy 80% dòng log đầu tiên rồi mới gom theo fixed window. Cách này không áp dụng được khi gom theo block, vì các block chạy đan xen nhau. Đây là suy luận của chúng ta, paper không viết như vậy.

4. **cascade-lad chia theo thời gian ở mức block.** Mỗi block có thời điểm `start = min(timestamp)`. Các block được sắp xếp theo `(start, group_id)` rồi cắt. Block train nào kết thúc sau khi block val đầu tiên bắt đầu sẽ bị purge, và tương tự giữa val với test ([splitting.py](../src/cascadelad/data/splitting.py)). Số block bị purge được in trong mục `[data]` của log mỗi lần chạy, và cần báo cáo con số này trong luận văn.

## Điểm khác so với DeepLog cần ghi trong luận văn

- **Phạm vi dữ liệu.** DeepLog train trên 100k dòng đầu và test trên toàn bộ phần còn lại. cascade-lad với `max_lines: 100000` chia 70/10/20 bên trong 100k dòng đó. Muốn chạy trên toàn bộ HDFS thì đặt `max_lines: null` trong [hdfs.yaml](../configs/dataset/hdfs.yaml).
- **Chọn g.** DeepLog cố định g=9 và khảo sát ảnh hưởng của g trên test. cascade-lad chọn g trên val, dùng cùng thủ tục cho mọi method.
- **Session ở ranh giới.** DeepLog không nói rõ cách xử lý block bắt đầu trong 100k dòng đầu nhưng còn tiếp tục về sau. cascade-lad xử lý chuyện này rõ ràng bằng purge.
- **Timestamp chỉ chính xác tới giây.** Nhiều block có cùng `start`; thứ tự giữa chúng được quyết định bởi `group_id`. Cách này ổn định nhưng tùy ý, nên ghi vào phần hạn chế.
