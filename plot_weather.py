import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
CSV_PATH = WORKSPACE / "weather_20251008_14.csv"
TEMPERATURE_IMAGE_PATH = WORKSPACE / "temperature_20251008_14.png"
HUMIDITY_IMAGE_PATH = WORKSPACE / "humidity_20251008_14.png"


def load_hourly_series(csv_path: Path, value_column: str) -> pd.Series:
    # CSV의 시각과 지정된 수치를 변환·정렬하고 누락된 1시간 간격을 보강합니다.
    frame = pd.read_csv(csv_path)
    required_columns = {"datetime", value_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")
    if frame.empty:
        raise ValueError("CSV에 그래프로 표시할 자료가 없습니다.")

    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    invalid_timestamps = timestamps.isna()
    if invalid_timestamps.any():
        raise ValueError(f"datetime을 변환할 수 없는 행이 {int(invalid_timestamps.sum())}개 있습니다.")

    raw_values = frame[value_column]
    values = pd.to_numeric(raw_values, errors="coerce")
    invalid_values = raw_values.notna() & values.isna()
    if invalid_values.any():
        raise ValueError(f"{value_column}이 숫자가 아닌 행이 {int(invalid_values.sum())}개 있습니다.")

    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps), name=value_column)
    series = series.groupby(level=0, sort=True).mean()
    hourly_index = pd.date_range(
        start=series.index.min().floor("h"),
        end=series.index.max().ceil("h"),
        freq="h",
    )
    complete_index = hourly_index.union(series.index).sort_values()
    return series.reindex(complete_index)


def save_and_show_plot(
    values: pd.Series,
    image_path: Path,
    title: str,
    y_label: str,
) -> None:
    # 결측 구간은 NaN으로 둬 선을 끊고, 그래프를 저장한 뒤 화면에 표시합니다.
    figure, axis = plt.subplots(figsize=(10, 5))
    try:
        axis.plot(values.index, values.to_numpy(), marker="o")
        axis.set_title(title)
        axis.set_xlabel("Time (KST)")
        axis.set_ylabel(y_label)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(image_path)
        plt.show()
    finally:
        plt.close(figure)


def main() -> int:
    try:
        temperatures = load_hourly_series(CSV_PATH, "temperature_C")
        humidity = load_hourly_series(CSV_PATH, "humidity_pct")
        save_and_show_plot(
            temperatures,
            TEMPERATURE_IMAGE_PATH,
            "Temperature over time",
            "Temperature (°C)",
        )
        save_and_show_plot(
            humidity,
            HUMIDITY_IMAGE_PATH,
            "Humidity over time",
            "Humidity (%)",
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"그래프를 만들 수 없습니다: {error}", file=sys.stderr)
        return 1

    print(f"기온 그래프 저장 위치: {TEMPERATURE_IMAGE_PATH}")
    print(f"습도 그래프 저장 위치: {HUMIDITY_IMAGE_PATH}")
    print(f"시간 범위: {temperatures.index.min()} ~ {temperatures.index.max()}")
    print(f"누락 기온 시각 수: {int(temperatures.isna().sum())}")
    print(f"누락 습도 시각 수: {int(humidity.isna().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
