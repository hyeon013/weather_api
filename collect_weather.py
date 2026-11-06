import getpass
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus

import pandas as pd
import requests


API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
TIMEOUT = (5, 30)
WORKSPACE = Path(__file__).resolve().parent
OUTPUT_PATH = WORKSPACE / "weather_20251008_14.csv"
RAW_RESPONSE_PATH = WORKSPACE / "weather_response_20251008_14.txt"
PREVIOUS_OUTPUT_PATH = WORKSPACE / "weather.csv"
OUTPUT_COLUMNS = ["datetime", "temperature_C", "humidity_pct", "wind_speed_ms"]
REQUEST_START = "202510080000"
REQUEST_END = "202510142300"

# API help 출력의 1부터 시작하는 필드 번호: TM(1), WS(4), TA(12), HM(14)
FIELD_POSITIONS = {"TM": 1, "WS": 4, "TA": 12, "HM": 14}
TEMPERATURE_MISSING_VALUES = {-99.0, -999.0, -9999.0}


def response_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()

    if isinstance(payload, dict):
        message = payload.get("message")
        if message is not None:
            return str(message)
    return body.strip()


def redact_auth_key(text: str, auth_key: str) -> str:
    redacted = text.replace(auth_key, "[인증키 숨김]")
    for encoded_key in {quote(auth_key, safe=""), quote_plus(auth_key)}:
        if encoded_key != auth_key:
            redacted = redacted.replace(encoded_key, "[인증키 숨김]")
    return re.sub(
        r"(?i)(authKey=)[^&\s\"'<>]+",
        r"\1[인증키 숨김]",
        redacted,
    )


def no_data_reason(body: str) -> str:
    # 주석 설명에서 자료 부재 사유를 찾아 안내하고, 없으면 응답에 사유가 없음을 알립니다.
    no_data_patterns = (
        r"no\s+data",
        r"자료\s*가?\s*없",
        r"데이터\s*가?\s*없",
        r"조회\s*(?:된\s*)?결과\s*가?\s*없",
        r"존재하지\s*않",
    )
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        explanation = stripped.lstrip("#").strip()
        if explanation and any(
            re.search(pattern, explanation, re.IGNORECASE)
            for pattern in no_data_patterns
        ):
            return explanation
    return "응답 본문에 자료 부재 사유가 명시되어 있지 않습니다."


def parse_weather_table(body: str) -> tuple[pd.DataFrame, dict[str, int]]:
    # 주석과 공백 구분 자료 행을 나누고, 변환 실패 수를 세어 표로 만듭니다.
    raw_records: list[dict[str, str]] = []
    malformed_rows = 0
    for line_number, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        fields = stripped.split()
        if len(fields) < max(FIELD_POSITIONS.values()):
            malformed_rows += 1
            print(
                f"응답 {line_number}행을 제외했습니다: 필드 {len(fields)}개 "
                f"(최소 {max(FIELD_POSITIONS.values())}개 필요)",
                file=sys.stderr,
            )
            continue

        raw_records.append(
            {
                name: fields[position - 1]
                for name, position in FIELD_POSITIONS.items()
            }
        )

    raw = pd.DataFrame(raw_records, columns=FIELD_POSITIONS)
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {
            "malformed_rows": malformed_rows,
            "invalid_datetime": 0,
            "invalid_temperature": 0,
            "invalid_humidity": 0,
            "invalid_wind_speed": 0,
        }

    timestamps = pd.to_datetime(raw["TM"], format="%Y%m%d%H%M", errors="coerce")
    invalid_datetime = int(timestamps.isna().sum())
    numeric = {
        "temperature_C": pd.to_numeric(raw["TA"], errors="coerce"),
        "humidity_pct": pd.to_numeric(raw["HM"], errors="coerce"),
        "wind_speed_ms": pd.to_numeric(raw["WS"], errors="coerce"),
    }
    conversion_failures = {
        column: int((raw[source].notna() & numeric[column].isna()).sum())
        for column, source in (
            ("temperature_C", "TA"),
            ("humidity_pct", "HM"),
            ("wind_speed_ms", "WS"),
        )
    }

    # 비유한 수치와 물리적으로 음수가 될 수 없는 관측값은 결측으로 둡니다.
    for column in numeric:
        numeric[column] = numeric[column].replace([float("inf"), float("-inf")], pd.NA)
    numeric["temperature_C"] = numeric["temperature_C"].mask(
        numeric["temperature_C"].isin(TEMPERATURE_MISSING_VALUES)
    )
    numeric["humidity_pct"] = numeric["humidity_pct"].mask(numeric["humidity_pct"] < 0)
    numeric["wind_speed_ms"] = numeric["wind_speed_ms"].mask(numeric["wind_speed_ms"] < 0)

    table = pd.DataFrame(
        {
            "datetime": timestamps,
            "temperature_C": numeric["temperature_C"],
            "humidity_pct": numeric["humidity_pct"],
            "wind_speed_ms": numeric["wind_speed_ms"],
        }
    )
    table = table.dropna(subset=["datetime"]).sort_values("datetime")
    failures = {
        "malformed_rows": malformed_rows,
        "invalid_datetime": invalid_datetime,
        "invalid_temperature": conversion_failures["temperature_C"],
        "invalid_humidity": conversion_failures["humidity_pct"],
        "invalid_wind_speed": conversion_failures["wind_speed_ms"],
    }
    return table, failures


def save_weather_csv(table: pd.DataFrame) -> None:
    # 정리된 표를 UTF-8-SIG CSV로 저장하고 pandas 행 인덱스는 제외합니다.
    table.loc[:, OUTPUT_COLUMNS].to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )


def print_period_comparison(table: pd.DataFrame) -> None:
    # 기존 10월 1~7일 CSV와 새 기간의 기온 평균을 비교해 한 문장으로 출력합니다.
    try:
        previous = pd.read_csv(PREVIOUS_OUTPUT_PATH)
        if not {"datetime", "temperature_C"}.issubset(previous.columns):
            raise ValueError("기존 CSV에 datetime 또는 temperature_C 열이 없습니다.")

        previous_temperatures = pd.to_numeric(previous["temperature_C"], errors="coerce").dropna()
        current_temperatures = table["temperature_C"].dropna()
        if previous_temperatures.empty or current_temperatures.empty:
            print("기간 비교: 한 기간 이상에 유효한 기온 자료가 없어 평균을 비교할 수 없습니다.")
            return

        previous_mean = previous_temperatures.mean()
        current_mean = current_temperatures.mean()
        difference = current_mean - previous_mean
        direction = "높습니다" if difference > 0 else "낮습니다" if difference < 0 else "같습니다"
        print(
            "기간 비교: 2025-10-08~14의 평균기온은 "
            f"{current_mean:.2f}°C로 2025-10-01~07의 "
            f"{previous_mean:.2f}°C보다 {abs(difference):.2f}°C {direction}."
        )
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"기간 비교를 건너뛰었습니다: {error}", file=sys.stderr)


# 원본 응답을 보관하고 자료를 변환·저장한 뒤 처리 결과를 출력합니다.
def main() -> int:
    auth_key = getpass.getpass("기상청 API 인증키: ")
    if not auth_key:
        print("인증키를 입력해야 합니다.", file=sys.stderr)
        return 1

    params = {
        "tm1": REQUEST_START,
        "tm2": REQUEST_END,
        "stn": "108",
        "help": "1",
        "authKey": auth_key,
    }

    try:
        response = requests.get(API_URL, params=params, timeout=TIMEOUT)
    except requests.RequestException:
        print("HTTP 요청에 실패했습니다. 네트워크와 API 주소를 확인해 주세요.", file=sys.stderr)
        return 1

    print(f"HTTP 상태 코드: {response.status_code}")
    try:
        RAW_RESPONSE_PATH.write_bytes(response.content)
    except OSError as error:
        print(f"원본 응답 저장에 실패했습니다: {error}", file=sys.stderr)
        return 1

    encoding = response.encoding or response.apparent_encoding or "utf-8"
    try:
        body = response.content.decode(encoding, errors="replace")
    except LookupError:
        body = response.content.decode("utf-8", errors="replace")

    if not response.ok:
        message = redact_auth_key(response_message(body), auth_key)
        print(f"API 요청 실패: {message or '응답 본문이 비어 있습니다.'}", file=sys.stderr)
        return 1

    safe_lines = [redact_auth_key(line, auth_key) for line in body.splitlines()]
    print("\n응답 본문 앞 15줄:")
    print("\n".join(safe_lines[:15]))

    data_rows = [line for line in safe_lines if line.strip() and not line.lstrip().startswith("#")]
    print("\n실제 데이터 행 3개:")
    if data_rows:
        print("\n".join(data_rows[:3]))
    else:
        reason = redact_auth_key(no_data_reason(body), auth_key)
        print(f"자료행이 없어 {OUTPUT_PATH.name}을(를) 저장하지 않았습니다. 원인: {reason}")
        print(f"원본 응답 저장 위치: {RAW_RESPONSE_PATH}")
        return 1

    try:
        table, conversion_failures = parse_weather_table(body)
    except (OSError, ValueError, TypeError) as error:
        print(f"날씨 자료 처리에 실패했습니다: {error}", file=sys.stderr)
        return 1

    print(f"변환 실패 수: {conversion_failures}")
    if table.empty:
        print(
            f"유효한 시각의 자료행이 없어 {OUTPUT_PATH.name}을(를) 저장하지 않았습니다. "
            "변환 실패 건수를 확인해 주세요."
        )
        print(f"원본 응답 저장 위치: {RAW_RESPONSE_PATH}")
        return 1

    duplicate_counts = table["datetime"].value_counts()
    duplicate_rows = int((duplicate_counts - 1).clip(lower=0).sum())
    duplicate_times = int((duplicate_counts > 1).sum())
    print(f"중복 시각 그룹 수: {duplicate_times}")
    print(f"중복 초과 행 수 (행은 유지): {duplicate_rows}")
    missing_counts = table.isna().sum()
    expected_times = pd.date_range(
        start=pd.to_datetime(REQUEST_START, format="%Y%m%d%H%M"),
        end=pd.to_datetime(REQUEST_END, format="%Y%m%d%H%M"),
        freq="h",
    )
    observed_times = pd.DatetimeIndex(table["datetime"].drop_duplicates())
    missing_hours = len(expected_times.difference(observed_times))
    print(f"시간 누락 수 (요청 구간): {missing_hours}")

    try:
        save_weather_csv(table)
    except OSError as error:
        print(f"CSV 저장에 실패했습니다: {error}", file=sys.stderr)
        return 1

    first_time = table["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    last_time = table["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n행 수: {len(table)}")
    print(f"열별 결측 수: {missing_counts.to_dict()}")
    print(f"첫 시각: {first_time}")
    print(f"마지막 시각: {last_time}")
    temperature = table["temperature_C"].dropna()
    if temperature.empty:
        print("기온 범위·평균: 기온 자료 없음")
    else:
        print(
            "기온 범위: "
            f"{temperature.min():.1f}~{temperature.max():.1f}°C"
        )
        print(f"기온 평균: {temperature.mean():.2f}°C")
    print(f"CSV 저장 위치: {OUTPUT_PATH}")
    print(f"원본 응답 저장 위치: {RAW_RESPONSE_PATH}")
    print_period_comparison(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
