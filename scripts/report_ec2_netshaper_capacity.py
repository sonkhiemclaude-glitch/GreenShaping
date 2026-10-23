"""Write the evidence-based Vietnamese assessment from completed campaigns."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import ec2_runner as r


def report():
    root = r.EVIDENCE
    first = root / "capacity_01"
    retry = root / "capacity_recovery_01"
    rows = json.loads((retry / "summary.json").read_text())
    assert json.loads((retry / "status.json").read_text())["state"] == "complete"
    assert len(rows) == 16, "All prespecified recovery trials must be retained"
    raw = json.loads((first / "summary.json").read_text())
    analysis = json.loads((retry / "analysis.json").read_text())
    telemetry = json.loads((retry / "telemetry_analysis.json").read_text())
    lines = [
        "# Đánh giá NetShaper với mục tiêu 10 Gbps trên bốn EC2 hiện có",
        "",
        "Đã hoàn tất bộ thử trên đúng bốn `m6i.2xlarge` người dùng cung cấp. "
        "Mục tiêu 10 Gbps xuyên qua NetShaper **chưa đạt** trong các cấu hình đã thử. "
        "Mạng VPC nhiều luồng đạt trên 12 Gbps trong các phép đo ngắn; kết quả này không xác nhận băng thông duy trì dài hạn. Thứ tự case cố định và credit burst không được kiểm soát trực tiếp, "  # noqa: E501
        "nên không suy ra riêng tác động nhân quả của shaping từ chênh lệch hai mode.",
        "",
        "## Mạng trực tiếp",
        "",
        "| Chặng / cấu hình | Receiver Gbps | Số lần |",
        "|---|---:|---:|",
    ]
    for label in dict.fromkeys(x["label"] for x in raw if x["label"].startswith("raw_")):
        values = [x["receiver_bps"] / 1e9 for x in raw if x["label"] == label and x["receiver_bps"] is not None and not x["error"]]
        lines.append(f"| {label} | {statistics.median(values):.6f} | {len(values)} |")
    lines += [
        "",
        "Mỗi lần baseline đo 8 giây sau 2 giây warm-up. `P4` là bốn luồng TCP. "
        "Ở target 10 Gbps, `-b` được chia thành 2,5 Gbps/stream; không nhầm thành tổng tải 40 Gbps. "
        "Ba lần direct T1→T4 ở target 10 Gbps đều không có TCP retransmission.",
        "",
        "## NetShaper và tiêu chí đánh giá",
        "",
        "T1 dùng TCP tới T2:8000; T2/T3 trao đổi qua QUIC UDP:4567; T3 kết nối tới T4:5555. "
        "Tất cả dùng private IP trong VPC. SSH quản trị không nằm trong đường dữ liệu. "
        "Kiểm tra chức năng trước đó đã xác minh tuyến và 36/36 payload đúng nội dung; iperf3 ở lượt throughput không kiểm tra hash toàn bộ payload.",  # noqa: E501
        "",
        "Lượt xác minh sau lỗi dùng container 16 GiB, queue 8 MiB, tối đa 16 stream, "
        "shaper interval 1 ms, noise multiplier 1, sensitivity 65536 byte, max decision 16 MiB. "
        "Các tham số này phục vụ kiểm tra hiệu năng; chưa có epsilon/accounting để so sánh privacy với Green Shaping. "
        "Mỗi phép đo khởi động một cặp middlebox mới, dùng 4 TCP stream, 2 giây warm-up và 5 giây phát tải. "
        "Tải 3 Gbps chạy một lần mỗi mode/chiều; tải 10 Gbps chạy ba lần mỗi mode/chiều, tổng 16 phép đo.",
        "",
        "Số chính dưới đây lấy từ `iperf3 end.sum_received.bits_per_second`, chỉ dùng lần có exit code 0 và không có lỗi. "
        "Trường `reached_10Gbps` yêu cầu receiver ≥10.000.000.000 bps. Không đổi ngưỡng sau khi xem kết quả.",
        "",
        "| Mode / chiều ở target 10 Gbps | Lần hoàn tất hợp lệ | Median receiver Gbps | Min–max Gbps | Đạt ≥10 Gbps |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in analysis["groups"]:
        if not group["label"].endswith("10G"):
            continue
        value = f"{group['median_Gbps']:.3f}" if group["median_Gbps"] is not None else "—"
        span = f"{group['min_Gbps']:.3f}–{group['max_Gbps']:.3f}" if group["min_Gbps"] is not None else "—"
        lines.append(
            f"| {group['label']} | {group['successful']}/{group['n']} | {value} | {span} | {group['passes_10Gbps']}/{group['n']} |"
        )
    lines += [
        "",
        "**Cần phân biệt cửa sổ phát và cửa sổ nhận.** Khi proxy tích lũy backlog, iperf3 có thể chờ receiver drain lâu hơn thời gian phát 5 giây. "  # noqa: E501
        "Dữ liệu gửi trong warm-up cũng có thể còn trong hàng đợi sau khi reset bộ đếm. "
        "Vì vậy số receiver ở bảng là trung bình do iperf3 báo, gồm ảnh hưởng drain; không được coi là trần thông lượng ổn định hoặc tỷ lệ mất byte. "  # noqa: E501
        "CSV/analysis giữ riêng số giây và byte của sender/receiver. Các lần timeout có số byte hoặc tốc độ tạm thời vẫn được đánh dấu lỗi và không đưa vào median.",  # noqa: E501
        "",
        "| Mode / chiều ở target 3 Gbps | Receiver Gbps | Trạng thái |",
        "|---|---:|---|",
    ]
    for row in rows:
        if row["offered_aggregate_bps"] != 3_000_000_000:
            continue
        value = f"{row['receiver_bps'] / 1e9:.3f}" if row["receiver_bps"] is not None else "—"
        lines.append(f"| {row['label']} | {value} | {'Hoàn tất' if not row['error'] and row['returncode'] == 0 else 'Lỗi'} |")
    bulk_folder = root / "bulk_verify_01"
    bulk = json.loads((bulk_folder / "summary.json").read_text())
    assert json.loads((bulk_folder / "status.json").read_text())["state"] == "complete"
    assert len(bulk) == 5
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    values = [x["receiver_bps"] / 1e9 if x["receiver_bps"] is not None else 0 for x in bulk]
    bars = ax.bar(range(5), values, color=["gray", "tab:blue", "tab:blue", "tab:orange", "tab:orange"])
    ax.set_xticks(range(5), ["Direct\nforward", "No shaping\nforward", "No shaping\nreverse", "Shaping\nforward", "Shaping\nreverse"])
    ax.axhline(10, color="black", linestyle="--", linewidth=1, label="10 Gbps target")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
    ax.set(ylim=(0, 11.5), ylabel="Receiver goodput (Gbps)", title="Fixed 4 GiB transfer, CRC32 checked — one trial per condition")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(bulk_folder / "bulk_goodput.png", dpi=180)
    fig.savefig(bulk_folder / "bulk_goodput.pdf")
    plt.close(fig)
    lines += [
        "",
        "## Đối chứng truyền cố định có CRC32",
        "",
        "Bổ sung một lần truyền cho mỗi cấu hình: 4 stream ×1 GiB, tổng 4 GiB, "
        "mục tiêu phát tổng 10 Gbps. Mỗi stream tự handshake trước khi phát và xác nhận sau "
        "khi nhận đủ; không có stream điều khiển iperf3 riêng chờ giữa quá trình truyền. "
        "Receiver kiểm CRC32 của toàn bộ payload và đo từ byte đầu tiên tới byte cuối cùng "
        "trên cùng đồng hồ của máy nhận. Đây là kiểm tra chéo độc lập, không có warm-up/reset "
        "bộ đếm hay cộng dữ liệu warm-up vào kết quả.",
        "",
        "| Cấu hình | Receiver Gbps | Đủ 4 GiB và đúng CRC32 |",
        "|---|---:|---|",
    ]
    for item in bulk:
        value = f"{item['receiver_bps'] / 1e9:.3f}" if item["receiver_bps"] is not None else "—"
        good = item["returncode"] == 0 and item["crc_valid"] and item["total_bytes"] == 4 * 1073741824
        lines.append(f"| {item['label']} | {value} | {'Đúng' if good else 'Lỗi'} |")
    lines += [
        "",
        "Chỉ một lần đối chứng mỗi cấu hình, payload toàn byte zero, không phải workload bài báo "
        "hoặc bằng chứng băng thông duy trì. Case hoàn tất ở bulk nhưng timeout với iperf3 "
        "cho thấy không nên quy kết timeout cho mọi kiểu ứng dụng. Source/receiver CRC32 cũng "
        "tiêu thụ CPU; baseline direct được đo lại ngay trước các lượt bulk để đối chiếu.",
        "",
    ]
    oom = []
    for file in retry.glob("*_runtime.json"):
        if file.name == "final_runtime.json":
            continue
        obj = json.loads(file.read_text())
        if isinstance(obj, dict) and (inspect := obj.get("inspect")) and inspect[0]["State"]["OOMKilled"]:
            oom.append(file.name.removesuffix("_runtime.json"))
    abnormal = []
    for file in retry.glob("*_runtime.json"):
        if file.name == "final_runtime.json":
            continue
        obj = json.loads(file.read_text())
        if isinstance(obj, dict) and (inspect := obj.get("inspect")) and inspect[0]["State"]["ExitCode"] == 139:
            abnormal.append(file.name.removesuffix("_runtime.json"))
    peaks = [x["peak_GiB"] for x in telemetry["memory"] if x["peak_GiB"] is not None]
    lines += [
        "",
        "## Lỗi tải cao và telemetry",
        "",
        "Lượt đầu `capacity_01` đặt cap 4 GiB/container: T2 bị OOM-kill trong phép thử 10 Gbps đầu tiên. "
        "Hai lần connection-refused kế tiếp kế thừa middlebox đã chết, không phải hai phép đo trần độc lập. "
        "Đã dừng lượt này và giữ nguyên bằng chứng. Lượt `capacity_recovery_01` tăng cap lên 16 GiB và tạo cặp middlebox mới cho từng phép đo.",  # noqa: E501
        "",
        f"Trong lượt 16 GiB, Docker xác nhận OOMKilled ở {len(oom)} container đã thu được inspect. "
        f"Đỉnh `memory.peak` quan sát được là {max(peaks):.3f} GiB. "
        "Không suy ra leak hay nguyên nhân độc nhất chỉ từ hiện tượng này; cần profiling/bounded buffering riêng để xác định và sửa.",
        "",
        *["- " + name for name in oom],
        "",
        f"Có {len(abnormal)} container exit 139 sau các case lỗi; log có thông báo dumped core. "
        "Phải phân biệt thời điểm đóng kết nối/dừng container với thời điểm đang phát tải "
        "trước khi kết luận nguyên nhân segfault.",
        "Runtime log có cảnh báo fromShaped queue đầy lặp lại. Source tham chiếu "
        "`hardware/src/peer1/ShapedClient.cpp:285–293` lặp chờ push vào queue trong receive callback "
        "và log mỗi vòng; ở no-shaping không có sleep của nhánh SHAPING. "
        "Đây là điểm cần profiling về backpressure, worker blocking và chi phí log; "
        "chưa chứng minh binary image tương ứng chính xác source commit hoặc đây là nguyên nhân duy nhất.",
        "",
        "",
        "Một lần thu Docker log của reverse no-shaping bị SSH timeout sau khi đã lưu kết quả thử. "
        "Đã lưu trạng thái gián đoạn, xác minh container được dọn và tiếp tục các case chưa chạy; không chạy đè các case đã lưu. "
        "Những lần sau chỉ thu 150 dòng log cuối/container. Inspect T2 của reverse no-shaping lần 1 không được lưu thành công; không suy diễn trạng thái OOM của container đó.",  # noqa: E501
        "",
        "Telemetry gồm CPU và ENA trước/sau từng phép đo; cgroup memory/CPU và NIC counters 1 Hz trong lượt retry. "
        "NIC counters gồm header và traffic khác trên ens5, không thay thế application goodput. "
        "Trace NIC bắt đầu sau hai lần 10 Gbps forward đầu tiên; case thiếu mẫu được ghi null, không nội suy. "
        "Monitor được khởi động lại có ghi biên bản để gia hạn thời lượng; giữ các khoảng trống thay vì nội suy. "
        "Biểu đồ throughput giữ cả khoảng min–max ba lần; không dùng ba lần lặp để tuyên bố ý nghĩa thống kê.",
        "",
        "## Giới hạn nền tảng và cách diễn giải",
        "",
        "AWS công bố m6i.2xlarge baseline mạng 3,125 Gbps và burst tới 12,5 Gbps. "
        "Phép đo ngắn ở đây nằm trong khả năng burst, không chứng minh duy trì 10 Gbps sau khi hết credit. "
        "[Thông số AWS](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html).",
        "",
        "AWS giới hạn một TCP/UDP flow ở 5 Gbps khi hai instance không cùng cluster placement group; "
        "cùng cluster placement group có thể đạt tới 10 Gbps/flow. Chưa xác minh membership của bốn VM. "
        "Bốn TCP stream ở hai đầu vẫn đi trong một kết nối QUIC giữa middlebox. "
        "Vì vậy không thể lấy baseline P4 >12 Gbps làm bằng chứng một QUIC tunnel sẽ đạt 10 Gbps. "
        "[Quy tắc băng thông AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-network-bandwidth.html).",
        "",
        "Đây là đánh giá runtime upstream bằng image ID đã cố định, không phải binary đã chứng minh build từ source commit tham chiếu. "
        "Không đo joule, không có chứng nhận privacy, không chạy attacker hoặc workload bài báo được khớp budget. "
        "Kết quả bổ sung bằng chứng về network/runtime và chưa hoàn tất E1–E5 của bản thảo.",
        "",
        "## Biểu đồ",
        "",
        "![Fixed-length bulk goodput](../outputs/ec2_netshaper_20261020/bulk_verify_01/bulk_goodput.png)",
        "",
        "![iperf3 receiver averages including drain](../outputs/ec2_netshaper_20261020/capacity_recovery_01/throughput.png)",
        "",
        "![NIC and memory telemetry](../outputs/ec2_netshaper_20261020/capacity_recovery_01/telemetry.png)",
        "",
        "## Bằng chứng và chạy lại",
        "",
        "- `outputs/ec2_netshaper_20261020/capacity_01/`: baseline mạng, ramp ban đầu và lỗi cap 4 GiB.",
        "- `outputs/ec2_netshaper_20261020/capacity_recovery_01/`: 16 case, cấu hình, raw iperf JSON, server logs, Docker inspect, CSV, analysis và telemetry.",  # noqa: E501
        "- `capacity_recovery_01/throughput.png` và `telemetry.png`: biểu đồ; có bản PDF để xuất.",
        "- `outputs/ec2_netshaper_20261020/bulk_verify_01/`: đối chứng CRC32, cấu hình, client/server records.",
        "- `manifest.json` trong mỗi run: SHA-256 của các artifact.",
        "- Chạy mới: `.venv/bin/python scripts/run_ec2_netshaper_capacity_recovery.py --run-name <tên_mới>`.",
        "- Cấu hình mạng, vai trò máy, digest và kiểm tra chức năng: `docs/ec2_netshaper_setup_20261020.md`.",
        "",
        "Đã dừng container, iperf server và các tiến trình monitor của phép thử; giữ nguyên bốn EC2 running và SG đã cấu hình. "
        "Không thay instance type hoặc tạo thêm máy. Archive không chứa khóa SSH, .env hoặc TLS private key.",
        "",
    ]
    path = Path(__file__).resolve().parents[1] / "docs/ec2_netshaper_capacity_20261020.md"
    path.write_text("\n".join(lines))
    print(path)


if __name__ == "__main__":
    report()
